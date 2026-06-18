"""``mimir.tools.offload`` 单元测试（纯逻辑，不依赖 GPU）。"""

from __future__ import annotations

from pathlib import Path

from benchmarks.workloads import ToolResult

from mimir.tools.offload import (
    DEFAULT_INLINE_THRESHOLD,
    ToolDataStore,
    offload_workload_tool_results,
)


def test_small_result_inlined() -> None:
    store = ToolDataStore()
    out = store.put("calc", "42")
    assert out == "42"  # 小结果直接进上下文
    assert store.inline_count == 1
    assert store.offloaded_count == 0


def test_large_result_offloaded_with_ref() -> None:
    store = ToolDataStore()
    big = '{"x": "' + "a" * 2000 + '"}'
    out = store.put("search", big)
    # 应外置：上下文里是引用+摘要，远小于原文
    assert len(out) < len(big)
    assert "TOOL_RESULT" in out
    assert "ref=" in out
    assert store.offloaded_count == 1
    assert store.offloaded_chars == len(big)


def test_materialize_roundtrip() -> None:
    store = ToolDataStore()
    big = "[data] " * 500
    out = store.put("sql", big)
    # 从上下文文本里抠出 ref_id 再 materialize
    rid = out.split("ref=")[1].split()[0].rstrip("]")
    assert store.materialize(rid) == big
    assert store.materialize("nonexistent") is None


def test_disk_persistence(tmp_path: Path) -> None:
    store = ToolDataStore(disk_dir=tmp_path)
    big = '{"name":"x","data":"' + "z" * 1000 + '"}'
    out = store.put("search", big)
    rid = out.split("ref=")[1].split()[0].rstrip("]")
    assert (tmp_path / f"{rid}.json").exists()
    # 清掉内存后仍能从盘 materialize
    store._store.clear()  # noqa: SLF001
    assert store.materialize(rid) == big


def test_offload_workload_tool_results() -> None:
    store = ToolDataStore()
    results = [
        ToolResult(name="search", content="short", tokens_approx=1),
        ToolResult(name="sql", content='{"big":"' + "y" * 3000 + '"}', tokens_approx=750),
    ]
    texts, stats = offload_workload_tool_results(store, results)
    assert len(texts) == 2
    assert stats.offloaded_count == 1  # 只有大那个被外置
    assert stats.reduction_pct > 50
    # 第一个（小）保持原文
    assert texts[0] == "short"


def test_threshold_control() -> None:
    store = ToolDataStore()
    mid = "x" * (DEFAULT_INLINE_THRESHOLD + 1)
    out = store.put("t", mid, inline_threshold=DEFAULT_INLINE_THRESHOLD)
    assert "TOOL_RESULT" in out  # 超阈值→外置


def test_offload_with_tiered_backend_promotes_on_access() -> None:
    """外置数据进入分层后，materialize 会从冷层 promote 回来。"""
    from mimir.tiered.store import Tier, TieredStore

    tiered = TieredStore(gpu_cap=1, host_cap=1, disk_dir=None)
    store = ToolDataStore(tiered=tiered)
    # 放 3 个大结果：第 1 个最终落到 DISK
    for i in range(3):
        store.put("search", "x" * 2000 + str(i))
    stats = store.stats()
    assert stats["offloaded_count"] == 3
    # materialize 仍能取回（从任意层）
    # 取最后一个 ref（在 GPU）
    snap = tiered.snapshot()
    assert len(snap["disk"]) >= 1
    any_disk_key = snap["disk"][0]
    val = store.materialize(any_disk_key)
    assert val is not None
    assert tiered._tier_of(any_disk_key) is Tier.GPU  # promote 回热层
    assert tiered.stats.promotions >= 1
