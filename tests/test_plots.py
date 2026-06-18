"""``mimir.plots`` 的单元测试（生成 PNG，不依赖 GPU）。"""

from __future__ import annotations

from pathlib import Path

from mimir.metrics import RunMetrics
from mimir.plots import plot_ablation_curve, plot_kv_mem_comparison, plot_latency_comparison


def _mk(wl: str, label: str, mem: float, ttft: float, e2e: float, succ: bool = True) -> RunMetrics:
    return RunMetrics(
        label=label,
        peak_gpu_mem_alloc_gib=mem,
        ttft_ms=ttft,
        e2e_latency_s=e2e,
        throughput_tok_per_s=100.0,
        task_success=succ,
        extra={"workload": wl, "peak_kv_used_gib": mem},
    )


def test_kv_mem_comparison_png(tmp_path: Path) -> None:
    rs = [
        _mk("multi_turn", "baseline", 14.0, 120, 3.4),
        _mk("multi_turn", "optimized", 8.0, 60, 2.1),
        _mk("tool_call", "baseline", 18.0, 200, 5.0),
        _mk("tool_call", "optimized", 9.0, 90, 2.8),
    ]
    out = tmp_path / "kv.png"
    p = plot_kv_mem_comparison(rs, out)
    assert Path(p).exists() and Path(p).stat().st_size > 1000


def test_latency_comparison_png(tmp_path: Path) -> None:
    rs = [_mk("multi_turn", "baseline", 14, 120, 3.4), _mk("multi_turn", "optimized", 8, 60, 2.1)]
    p = plot_latency_comparison(rs, tmp_path / "lat.png")
    assert Path(p).exists() and Path(p).stat().st_size > 1000


def test_ablation_curve_png(tmp_path: Path) -> None:
    abl = [
        ("baseline", 18.0, 5.0, 1.0),
        ("+prefix", 12.0, 3.8, 1.0),
        ("+compress", 9.0, 3.2, 0.98),
        ("+cow", 7.0, 2.9, 0.97),
    ]
    p = plot_ablation_curve(abl, tmp_path / "abl.png")
    assert Path(p).exists() and Path(p).stat().st_size > 1000
