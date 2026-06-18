"""``mimir.manager`` 集成测试：MemoryManager 编排各特性管线。"""

from __future__ import annotations

from benchmarks.workloads import gen_multi_turn, gen_tool_call

from mimir.context.compressor import Fidelity
from mimir.manager import MemoryManager


def test_unknown_feature_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="未知的特性开关"):
        MemoryManager(features=["bogus"])


def test_apply_with_no_features_passthrough() -> None:
    case = gen_multi_turn(num_turns=4)
    mm = MemoryManager(features=[])
    r = mm.apply(case)
    assert r.case is case  # 无变换
    # 各步都记录（enabled=False）
    names = {s.name for s in r.steps}
    assert {"prefix_cache", "context_compress", "tool_offload", "tiered"} <= names
    assert r.steps[0].metric.get("static_prefix_chars", 0) > 0


def test_context_compress_reduces_chars() -> None:
    case = gen_tool_call(num_calls=6)
    mm = MemoryManager(features=["context_compress", "prefix_cache"], fidelity=Fidelity.BALANCED)
    r = mm.apply(case)
    step = next(s for s in r.steps if s.name == "context_compress")
    assert step.enabled is True
    assert step.metric["compressed_chars"] < step.metric["original_chars"]


def test_tool_offload_replaces_large_results() -> None:
    case = gen_tool_call(num_calls=4)
    mm = MemoryManager(features=["tool_offload", "tiered"])
    r = mm.apply(case)
    step = next(s for s in r.steps if s.name == "tool_offload")
    assert step.enabled is True
    assert step.metric.get("offloaded_count", 0) > 0
    # 变换后工具结果被替换为更短的引用
    assert len(r.case.tool_results[0].content) < len(case.tool_results[0].content)


def test_full_pipeline_all_features() -> None:
    case = gen_tool_call(num_calls=6)
    mm = MemoryManager(
        features=["prefix_cache", "context_compress", "tool_offload", "tiered", "lifecycle"],
        fidelity=Fidelity.BALANCED,
    )
    r = mm.apply(case, task_id="t1")
    # 各步都启用
    assert all(
        s.enabled
        for s in r.steps
        if s.name in {"context_compress", "tool_offload", "tiered", "lifecycle"}
    )
    # finish_task 触发（lifecycle 启用）
    assert mm.finish_task("t1") >= 0


def test_semantic_compress_step_when_enabled() -> None:
    def fake(prompt: str) -> str:
        return "SUMMARY"

    case = gen_multi_turn(num_turns=6)
    mm = MemoryManager(features=["semantic_compress"], summarize_fn=fake)
    r = mm.apply(case)
    step = next(s for s in r.steps if s.name == "semantic_compress")
    assert step.enabled is True
    assert step.metric["llm_calls"] > 0
