"""``mimir.metrics`` 的单元测试（不依赖 GPU，可在 CPU 上跑通逻辑）。"""

from __future__ import annotations

import json
from pathlib import Path

from mimir.metrics import MetricsCollector, RunMetrics, load_results, save_results


def test_metrics_collector_records_timing_and_throughput() -> None:
    """TTFT / E2E / 吞吐应被正确记录。"""
    col = MetricsCollector(device=0)
    with col.track("baseline"):
        # 模拟 prefill 后产出首 token
        col.mark_first_token()
        col.add_output_tokens(150)
        col.success = True
    m = col.metrics()
    assert m.label == "baseline"
    assert m.task_success is True
    assert m.ttft_ms is not None and m.ttft_ms >= 0.0
    assert m.e2e_latency_s is not None and m.e2e_latency_s >= 0.0
    assert m.throughput_tok_per_s is not None and m.throughput_tok_per_s > 0


def test_mark_first_token_records_only_first() -> None:
    col = MetricsCollector()
    with col.track("x"):
        col.mark_first_token()
        first = col._t_first  # noqa: SLF001
        col.mark_first_token()  # 不应覆盖
        assert col._t_first == first  # noqa: SLF001


def test_metrics_serialization_roundtrip(tmp_path: Path) -> None:
    """结果应能序列化为 JSON 并无损读回。"""
    rs = [
        RunMetrics(
            label="baseline",
            peak_gpu_mem_alloc_gib=14.2,
            ttft_ms=120.0,
            e2e_latency_s=3.4,
            throughput_tok_per_s=44.0,
            task_success=True,
        ),
        RunMetrics(
            label="optimized",
            peak_gpu_mem_alloc_gib=8.1,
            ttft_ms=60.0,
            e2e_latency_s=2.1,
            throughput_tok_per_s=71.0,
            task_success=True,
        ),
    ]
    out = tmp_path / "r.json"
    save_results(rs, out)
    loaded = load_results(out)
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in rs]
    assert json.loads(out.read_text())[0]["label"] == "baseline"


def test_metrics_without_gpu_is_graceful() -> None:
    """即便无 GPU，采集器也应产出结构（mem 为 None 不报错）。"""
    col = MetricsCollector()
    with col.track("nogpu"):
        col.add_output_tokens(10)
    m = col.metrics()
    # mem 字段为 None 或 float；其余字段必须存在
    assert m.label == "nogpu"
    assert m.throughput_tok_per_s is not None
