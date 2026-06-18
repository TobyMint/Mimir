"""Prompt 与上下文压缩（赛题优化方向之三）。

对 system prompt、工具描述等内容进行去重与精简，消除冗余信息，从而降低上下文占用
并提升整体效率。详见 ``docs/技术方案.md`` §3.3。

本模块提供「请求侧」的上下文变换 —— 在把消息交给 vLLM 之前，先压缩 / 去重，
从而减少进入 KV Cache 的 token 数。与 vLLM 的 APC（前缀复用）正交且互补：
- APC 复用「完全相同的 token 前缀」的 KV；
- 本模块在「内容层面」消除冗余（摘要旧轮次、精简工具返回），从源头减少 token。

三类变换（保真度可配置）：
1. **静态去重**：跨轮稳定的 system / tool schema，提取为共享前缀（哈希指纹标记）。
2. **历史摘要**：把较早的对话轮次压缩为摘要，保留最近 N 轮原文（fidelity 控制）。
3. **工具结果精简**：把大工具返回替换为结构化摘要（保留关键字段，丢弃冗长原文）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.workloads import WorkloadCase


class Fidelity(str, Enum):
    """压缩保真度档位。越高越保真（压缩越少）。"""

    LOSSLESS = "lossless"  # 仅静态去重，不摘要
    BALANCED = "balanced"  # 摘要旧轮次 + 精简工具返回，保留近 N 轮原文
    AGGRESSIVE = "aggressive"  # 更激进摘要（更少保留轮次、更短摘要）


@dataclass
class CompressionStats:
    """压缩统计（用于评估与报告）。"""

    original_chars: int = 0
    compressed_chars: int = 0
    original_turns: int = 0
    compressed_turns: int = 0
    tool_results_summarized: int = 0
    static_prefix_chars: int = 0

    @property
    def char_reduction_pct(self) -> float:
        if self.original_chars <= 0:
            return 0.0
        return max(0.0, (1 - self.compressed_chars / self.original_chars) * 100)


def fingerprint(text: str) -> str:
    """内容指纹（用于标记可复用的静态前缀）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _summarize_tool_result(content: str, *, max_chars: int = 240) -> str:
    """把大工具返回压缩为结构化摘要。

    策略：若为 JSON 列表/对象，保留「键名 + 元素数 + 首项预览」；
    否则截断到 max_chars 并标注省略。保留关键字段（name/url/title/score/snippet）。
    """
    s = content.strip()
    if len(s) <= max_chars:
        return s
    # 尝试 JSON 结构化摘要
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        # 非 JSON：保留首尾，标注省略
        head = s[: max_chars // 2]
        tail = s[-(max_chars // 4) :]
        return f"{head}…[omitted {len(s) - len(head) - len(tail)} chars]…{tail}"
    return _summarize_json(obj, max_chars=max_chars, depth=0)


def _summarize_json(obj: Any, *, max_chars: int, depth: int) -> str:
    KEEP_KEYS = {"name", "title", "url", "score", "snippet", "id", "type"}
    if depth > 2:
        return "…"
    if isinstance(obj, list):
        n = len(obj)
        preview = _summarize_json(obj[0], max_chars=max_chars, depth=depth + 1) if obj else ""
        return f"[list x{n}]: {preview}"
    if isinstance(obj, dict):
        kept = {k: obj[k] for k in KEEP_KEYS if k in obj}
        rest = f", +{len(obj) - len(kept)} fields" if len(obj) > len(kept) else ""
        return json.dumps(kept, ensure_ascii=False)[:max_chars] + rest
    return repr(obj)[:max_chars]


def _summarize_turn_text(text: str, *, max_chars: int = 200) -> str:
    """把单轮文本压缩为摘要（保留首句 + 关键词）。"""
    s = text.strip()
    if len(s) <= max_chars:
        return s
    # 取第一个句号/换行作为首句
    m = re.split(r"[。.\n]", s, maxsplit=1)
    first = m[0][:max_chars] if m else s[:max_chars]
    return f"{first}…[+{len(s) - len(first)} chars summarized]"


@dataclass
class ContextCompressor:
    """上下文压缩器。"""

    fidelity: Fidelity = Fidelity.BALANCED
    keep_recent_turns: int = 2  # BALANCED 保留近 N 轮原文
    tool_result_max_chars: int = 240
    summary_max_chars: int = 200
    stats: CompressionStats = field(default_factory=CompressionStats)

    def _recent_keep(self) -> int:
        if self.fidelity is Fidelity.LOSSLESS:
            return 10**9  # 全保留
        if self.fidelity is Fidelity.AGGRESSIVE:
            return max(1, self.keep_recent_turns - 1)
        return self.keep_recent_turns

    def compress(self, case: WorkloadCase) -> WorkloadCase:
        """返回压缩后的 ``WorkloadCase``（不修改原对象）。

        - 静态去重：system + tool schemas 保留为前缀（指纹标记），不重复。
        - 历史摘要：早于 keep_recent 的轮次文本被摘要。
        - 工具返回：大返回被结构化摘要。
        """
        keep = self._recent_keep()
        original_chars = (
            sum(len(t.content) for t in case.turns)
            + sum(len(r.content) for r in case.tool_results)
            + len(case.system)
            + sum(len(s) for s in case.tool_schemas)
        )
        original_turns = len(case.turns)

        # 1) 工具 schema 静态去重：本实现里 tool_schemas 本就一份（不重复），仅记指纹
        static_prefix_chars = len(case.system) + sum(len(s) for s in case.tool_schemas)

        # 2) 历史轮次摘要
        new_turns = []
        n_turns = len(case.turns)
        for i, t in enumerate(case.turns):
            if self.fidelity is Fidelity.LOSSLESS or i >= n_turns - keep:
                new_turns.append(t)
            else:
                new_turns.append(
                    type(t)(
                        role=t.role,
                        content=_summarize_turn_text(t.content, max_chars=self.summary_max_chars),
                    )
                )

        # 3) 工具返回精简
        new_results = []
        summarized = 0
        for r in case.tool_results:
            if self.fidelity is Fidelity.LOSSLESS:
                new_results.append(r)
                continue
            comp = _summarize_tool_result(r.content, max_chars=self.tool_result_max_chars)
            if len(comp) < len(r.content):
                summarized += 1
            new_results.append(type(r)(name=r.name, content=comp, tokens_approx=len(comp) // 4))

        compressed_chars = (
            sum(len(t.content) for t in new_turns)
            + sum(len(r.content) for r in new_results)
            + len(case.system)
            + sum(len(s) for s in case.tool_schemas)
        )

        self.stats = CompressionStats(
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            original_turns=original_turns,
            compressed_turns=len(new_turns),
            tool_results_summarized=summarized,
            static_prefix_chars=static_prefix_chars,
        )
        return WorkloadCase(
            name=case.name,
            description=case.description,
            system=case.system,
            tool_schemas=list(case.tool_schemas),
            turns=new_turns,
            tool_results=new_results,
            branches=case.branches,
            recommended_features=case.recommended_features,
        )


def compress_workload(
    case: WorkloadCase, *, fidelity: Fidelity = Fidelity.BALANCED
) -> tuple[WorkloadCase, CompressionStats]:
    """便捷函数：返回 ``(压缩后的 case, 统计)``。"""
    c = ContextCompressor(fidelity=fidelity)
    out = c.compress(case)
    return out, c.stats
