"""评测指标采集器。

提供统一的指标采集上下文，覆盖赛题口径（对应 ``docs/测试报告.md`` §4）：
peak GPU memory、TTFT、端到端延迟、吞吐、任务成功率。

设计要点
--------
- **显存峰值用 torch.cuda.max_memory_allocated / max_memory_reserved**：
  - 按「本进程」计量，在 ``CUDA_VISIBLE_DEVICES`` 下索引正确；
  - 不受同卡其他用户干扰（比 nvidia-smi 全卡值更干净、更可比）；
  - 同时记录 allocated（净分配）与 reserved（含缓存池），后者反映 vLLM 预分配的 KV 池。
- 无 GPU 时优雅返回 None，便于无卡环境跑逻辑测试。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _gib(num_bytes: int | None) -> float | None:
    """字节 → GiB。"""
    if num_bytes is None:
        return None
    return num_bytes / (1024**3)


@dataclass
class RunMetrics:
    """单次运行的评测结果。"""

    label: str = ""
    peak_gpu_mem_alloc_gib: float | None = None
    peak_gpu_mem_reserved_gib: float | None = None
    ttft_ms: float | None = None
    e2e_latency_s: float | None = None
    throughput_tok_per_s: float | None = None
    task_success: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetrics:
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})


class MetricsCollector:
    """采集一次运行（一条工作流的一次执行）的指标。

    用法::

        with MetricsCollector(device=0).track("baseline") as c:
            ...  # prefill / generate
            c.mark_first_token()       # 记 TTFT 起点
            c.add_output_tokens(120)   # 累计输出 token
            c.success = True            # 任务是否成功
        m = c.metrics()  # -> RunMetrics
    """

    def __init__(self, device: int = 0) -> None:
        self.device = device
        self._t0: float | None = None
        self._t_first: float | None = None
        self._out_tokens = 0
        self.success: bool | None = None
        self._label = ""
        self._extra: dict[str, Any] = {}

    @contextmanager
    def track(self, label: str = "") -> Iterator[MetricsCollector]:
        self._label = label
        self._t0 = time.perf_counter()
        self._t_first = None
        self._out_tokens = 0
        self.success = None
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.device)  # 确保目标设备上下文存在
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                # 设备上下文未就绪或重置失败时降级：仅不重置峰值
                # （峰值可能包含引擎初始化，可接受，不应导致采集崩溃）
                pass
        try:
            yield self
        finally:
            pass  # 数值计算延迟到 metrics()

    def mark_first_token(self) -> None:
        """首个 token 产出时调用（仅记第一次）。"""
        if self._t_first is None and self._t0 is not None:
            self._t_first = time.perf_counter()

    def add_output_tokens(self, n: int) -> None:
        self._out_tokens += max(0, int(n))

    def set_extra(self, **kwargs: Any) -> None:
        self._extra.update(kwargs)

    def metrics(self) -> RunMetrics:
        now = time.perf_counter()
        e2e: float | None = (now - self._t0) if self._t0 is not None else None
        ttft: float | None = None
        if self._t_first is not None and self._t0 is not None:
            ttft = (self._t_first - self._t0) * 1000.0
        throughput: float | None = None
        if e2e and e2e > 0 and self._out_tokens > 0:
            throughput = self._out_tokens / e2e
        alloc = reserved = None
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                alloc = _gib(torch.cuda.max_memory_allocated(self.device))
                reserved = _gib(torch.cuda.max_memory_reserved(self.device))
            except RuntimeError:
                pass
        return RunMetrics(
            label=self._label,
            peak_gpu_mem_alloc_gib=alloc,
            peak_gpu_mem_reserved_gib=reserved,
            ttft_ms=ttft,
            e2e_latency_s=e2e,
            throughput_tok_per_s=throughput,
            task_success=self.success,
            extra=dict(self._extra),
        )


def save_results(results: list[RunMetrics], path: str | Path) -> None:
    """把多次运行结果序列化为 JSON 落盘（供测试报告/画图复用）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_results(path: str | Path) -> list[RunMetrics]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RunMetrics.from_dict(d) for d in data]
