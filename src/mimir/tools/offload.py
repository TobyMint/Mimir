"""工具调用数据优化（赛题优化方向之四）。

对工具调用过程中产生的大规模中间数据进行结构化存储或按需加载，避免其完全进入 KV Cache。
详见 ``docs/技术方案.md`` §3.4。

核心思想
--------
工具返回的大对象（长 JSON / 表格 / 搜索结果）存入外部 ``ToolDataStore``，**不直接进入
上下文**。在消息中仅放入一个轻量「引用 + 摘要」；当后续推理步骤真正需要完整数据时，再按需
materialize（lazy load）。这样大块中间数据不进入 KV Cache，显著降低显存与 prefill 成本。

与上下文压缩（``mimir.context``）正交：
- 上下文压缩：对「进入上下文的内容」做摘要；
- 工具外置：让大工具结果「根本不进入上下文」，只留引用。
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 工具返回「够小」就直接放进上下文，不必外置的阈值（字符数）
DEFAULT_INLINE_THRESHOLD = 512


@dataclass(frozen=True)
class ToolDataRef:
    """对已外置工具数据的引用（轻量，进入上下文）。"""

    ref_id: str
    tool_name: str
    summary: str  # 短摘要（保留关键字段）
    full_chars: int  # 原始数据字符数（用于统计）
    tokens_approx: int  # 原始数据粗估 token 数

    def as_context_text(self) -> str:
        """放入上下文的文本：摘要 + 引用标记（不含完整数据）。"""
        return (
            f"[TOOL_RESULT ref={self.ref_id} tool={self.tool_name} "
            f"len={self.full_chars}]\n{self.summary}\n[/TOOL_RESULT]"
        )


def _short_summary(content: str, *, max_chars: int = 200) -> str:
    """生成短摘要：JSON 取结构/计数，否则截断。"""
    s = content.strip()
    if len(s) <= max_chars:
        return s
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        head = s[: max_chars // 2]
        return f"{head}…[+{len(s) - len(head)} chars offloaded]"
    if isinstance(obj, list):
        n = len(obj)
        first = json.dumps(obj[0], ensure_ascii=False)[: max_chars - 40] if obj else ""
        return f"[list x{n}] {first}"
    if isinstance(obj, dict):
        keys = list(obj.keys())[:8]
        return f"[dict keys={keys} len={len(s)}]"
    return repr(obj)[:max_chars]


def _ref_id(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]


class ToolDataStore:
    """外置工具数据存储（进程内 dict + 可选落盘）。

    线程安全。支持按需 ``materialize`` 取回完整数据。
    """

    def __init__(self, *, disk_dir: str | Path | None = None) -> None:
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        self._disk_dir = Path(disk_dir) if disk_dir else None
        if self._disk_dir:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        # 统计
        self.offloaded_count = 0
        self.offloaded_chars = 0
        self.inline_count = 0

    def put(
        self,
        tool_name: str,
        content: str,
        *,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
        summary_max_chars: int = 200,
    ) -> str:
        """登记一条工具返回。

        - 若 ``len(content) <= inline_threshold``：直接返回原文（不外置）。
        - 否则：外置存储，返回 ``ToolDataRef.as_context_text()``（含摘要 + 引用）。

        返回值即「应放入上下文的消息内容」。
        """
        if len(content) <= inline_threshold:
            with self._lock:
                self.inline_count += 1
            return content
        rid = _ref_id(content)
        with self._lock:
            self._store[rid] = content
            self.offloaded_count += 1
            self.offloaded_chars += len(content)
        # 可选落盘（冷数据外存）
        if self._disk_dir:
            (self._disk_dir / f"{rid}.json").write_text(content, encoding="utf-8")
        ref = ToolDataRef(
            ref_id=rid,
            tool_name=tool_name,
            summary=_short_summary(content, max_chars=summary_max_chars),
            full_chars=len(content),
            tokens_approx=len(content) // 4,
        )
        return ref.as_context_text()

    def materialize(self, ref_id: str) -> str | None:
        """按需取回完整数据（lazy load）。"""
        with self._lock:
            if ref_id in self._store:
                return self._store[ref_id]
        if self._disk_dir:
            p = self._disk_dir / f"{ref_id}.json"
            if p.exists():
                return p.read_text(encoding="utf-8")
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "offloaded_count": self.offloaded_count,
            "offloaded_chars": self.offloaded_chars,
            "inline_count": self.inline_count,
            "store_size": len(self._store),
        }


@dataclass
class OffloadStats:
    """工具外置统计（汇总）。"""

    original_chars: int = 0
    in_context_chars: int = 0
    offloaded_count: int = 0

    @property
    def reduction_pct(self) -> float:
        if self.original_chars <= 0:
            return 0.0
        return max(0.0, (1 - self.in_context_chars / self.original_chars) * 100)


def offload_workload_tool_results(
    store: ToolDataStore,
    tool_results: list[Any],
    *,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
) -> tuple[list[str], OffloadStats]:
    """把一组工具返回外置，返回 ``(进入上下文的文本列表, 统计)``。

    ``tool_results`` 元素需有 ``.name`` 与 ``.content``（如 ``benchmarks.workloads.ToolResult``）。
    """
    texts: list[str] = []
    stats = OffloadStats()
    for r in tool_results:
        content = r.content
        stats.original_chars += len(content)
        ctx = store.put(r.name, content, inline_threshold=inline_threshold)
        stats.in_context_chars += len(ctx)
        if len(ctx) < len(content):
            stats.offloaded_count += 1
        texts.append(ctx)
    return texts, stats
