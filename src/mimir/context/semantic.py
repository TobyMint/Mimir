"""基于 LLM 的语义上下文压缩（新优化方向）。

``mimir.context.compressor`` 是启发式压缩（截断/结构摘要），保真度有限。本模块提供
**LLM 驱动**的语义压缩：用推理模型本身把「较早的对话轮次」摘要成一段自然语言，
保留语义与关键信息，再喂回上下文。比启发式更保真，可按需调用（成本高，配额制）。

策略
----
- 把早于 ``keep_recent`` 的连续若干轮合并为一段，请求模型生成一段「对话摘要」。
- 摘要替换原轮次，进入上下文（token 数大幅下降且保留语义）。
- 摘要带缓存（同输入→同摘要），避免重复调用。
- fidelity 档位控制：LOSSLESS 不摘要、BALANCED 摘要旧轮、AGGRESSIVE 更早摘要更多轮。

与启发式的关系：二者可组合 —— 先 LLM 摘要（语义保真），再启发式精简工具返回（结构）。
本模块聚焦 LLM 摘要路径。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from benchmarks.workloads import ConversationTurn, WorkloadCase

from mimir.context.compressor import Fidelity


@dataclass
class SemanticSummaryStats:
    """LLM 摘要统计。"""

    original_chars: int = 0
    summarized_chars: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    summarized_turn_groups: int = 0

    @property
    def reduction_pct(self) -> float:
        if self.original_chars <= 0:
            return 0.0
        return max(0.0, (1 - self.summarized_chars / self.original_chars) * 100)


# 摘要指令（保真导向：保留事实、决策、关键数值，丢弃寒暄）
SUMMARIZE_INSTRUCTION = (
    "Summarize the following agent conversation turns into a compact paragraph. "
    "Preserve: key facts, decisions made, important numbers/identifiers, and the "
    "current goal. Drop pleasantries and redundancy. Keep it faithful and concise."
)


def _group_key(turns_text: str) -> str:
    return hashlib.sha1(turns_text.encode("utf-8")).hexdigest()[:16]


class LLMSemanticCompressor:
    """LLM 驱动的语义压缩器。

    ``summarize_fn(text) -> str``：注入实际 LLM 调用（如 vLLM 引擎的 chat）。
    未注入时退化为「截断摘要」（便于无卡单测）。
    """

    def __init__(
        self,
        *,
        summarize_fn: Callable[[str], str] | None = None,
        fidelity: Fidelity = Fidelity.BALANCED,
        keep_recent_turns: int = 2,
        turns_per_summary: int = 3,
        summary_max_chars: int = 400,
    ) -> None:
        self.summarize_fn = summarize_fn
        self.fidelity = fidelity
        self.keep_recent_turns = keep_recent_turns
        self.turns_per_summary = turns_per_summary
        self.summary_max_chars = summary_max_chars
        self._cache: dict[str, str] = {}
        self.stats = SemanticSummaryStats()

    def _summarize_group(self, group_text: str) -> str:
        """摘要一组轮次（带缓存）。无 summarize_fn 时退化为截断。"""
        key = _group_key(group_text)
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        if self.summarize_fn is None:
            # 退化：截断
            summary = group_text[: self.summary_max_chars]
            if len(group_text) > self.summary_max_chars:
                summary += f"…[+{len(group_text) - self.summary_max_chars} chars]"
        else:
            prompt = f"{SUMMARIZE_INSTRUCTION}\n\n---\n{group_text}\n---"
            summary = self.summarize_fn(prompt)[: self.summary_max_chars]
            self.stats.llm_calls += 1
        self._cache[key] = summary
        return summary

    def compress(self, case: WorkloadCase) -> tuple[WorkloadCase, SemanticSummaryStats]:
        """返回 ``(压缩后 case, 统计)``。LOSSLESS 直接返回原样。"""
        self.stats = SemanticSummaryStats()
        if self.fidelity is Fidelity.LOSSLESS or len(case.turns) <= self.keep_recent_turns:
            self.stats.original_chars = sum(len(t.content) for t in case.turns)
            self.stats.summarized_chars = self.stats.original_chars
            return case, self.stats

        keep = (
            self.keep_recent_turns
            if self.fidelity is Fidelity.BALANCED
            else max(1, self.keep_recent_turns - 1)
        )
        old_turns = case.turns[: len(case.turns) - keep]
        recent = case.turns[len(case.turns) - keep :]

        orig_chars = sum(len(t.content) for t in old_turns)
        # 分组摘要
        summary_parts: list[str] = []
        for i in range(0, len(old_turns), self.turns_per_summary):
            group = old_turns[i : i + self.turns_per_summary]
            group_text = "\n".join(f"[{t.role}] {t.content}" for t in group)
            self.stats.summarized_turn_groups += 1
            summary_parts.append(self._summarize_group(group_text))

        summary_text = " summaries:\n- " + "\n- ".join(summary_parts)
        self.stats.summarized_chars = len(summary_text)

        new_turns = [ConversationTurn(role="system", content=summary_text)] + list(recent)
        self.stats.original_chars = orig_chars + sum(len(t.content) for t in recent)
        self.stats.summarized_chars += 0  # summary_text 已计入

        new_case = WorkloadCase(
            name=case.name,
            description=case.description,
            system=case.system,
            tool_schemas=list(case.tool_schemas),
            turns=new_turns,
            tool_results=list(case.tool_results),
            branches=case.branches,
            recommended_features=case.recommended_features,
        )
        return new_case, self.stats


def make_vllm_summarize_fn(engine: Any) -> Callable[[str], str]:
    """用 vLLM 引擎构造摘要函数（注入 LLMSemanticCompressor）。"""

    def _summarize(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text, _n = engine.chat(msgs, max_tokens=256, temperature=0.0)
        return text

    return _summarize
