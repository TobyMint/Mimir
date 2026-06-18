"""``mimir.multitask`` 单元测试（纯逻辑）。"""

from __future__ import annotations

from mimir.multitask import MultiTaskCoordinator, simulate_multi_task


def test_shared_prefix_pinned_once() -> None:
    coord = MultiTaskCoordinator(gpu_block_capacity=100, shared_prefix_blocks=10)
    coord.start_task("A")
    coord.add_blocks("A", own=5)
    coord.start_task("B")
    coord.add_blocks("B", own=5)
    # 前缀只被 pin 一次（10 块），不随任务数翻倍
    snap = coord._evictor.snapshot()  # noqa: SLF001
    assert snap["by_lifecycle"].get("pinned", 0) == 10


def test_finish_task_reclaims_own_blocks() -> None:
    coord = MultiTaskCoordinator(gpu_block_capacity=100, shared_prefix_blocks=5)
    coord.start_task("A")
    coord.add_blocks("A", own=20)
    reclaimed = coord.finish_task("A")
    assert reclaimed == 20
    # 前缀 pin 仍在
    assert coord._evictor.snapshot()["by_lifecycle"].get("pinned", 0) == 5  # noqa: SLF001


def test_sharing_savings_growth_with_tasks() -> None:
    """任务越多，共享前缀节省越多。"""
    coord = MultiTaskCoordinator(gpu_block_capacity=1000, shared_prefix_blocks=10)
    for t in range(4):
        coord.start_task(f"t{t}")
        coord.add_blocks(f"t{t}", own=8)
        coord.finish_task(f"t{t}")
    cs = coord.stats()
    # 4 任务共享 10 块前缀：节省 = 10 * (4-1) = 30
    assert cs.sharing_savings_blocks == 10 * 3


def test_concurrent_peak_tracked() -> None:
    coord = MultiTaskCoordinator(gpu_block_capacity=100, shared_prefix_blocks=2)
    coord.start_task("A")
    coord.start_task("B")
    coord.start_task("C")
    assert coord.peak_concurrent == 3
    coord.finish_task("A")
    coord.start_task("D")  # 并发回到 3（B,C,D）
    assert coord.peak_concurrent == 3


def test_simulate_multi_task_reports_coordination_benefit() -> None:
    r = simulate_multi_task(
        num_tasks=5, shared_prefix_blocks=15, own_blocks_per_task=30, gpu_capacity=500
    )
    # 协调后峰值远低于朴素（无共享无回收）
    assert r["coordinated"]["lifecycle_reclaims"] == 5 * 30  # 每任务结束回收 30
    assert r["coordination_benefit"]["reclaim_vs_naive_pct"] > 0
    assert r["coordination_benefit"]["sharing_savings_vs_naive_pct"] > 0


def test_unknown_task_blocks_raises() -> None:
    import pytest

    coord = MultiTaskCoordinator(gpu_block_capacity=50, shared_prefix_blocks=0)
    with pytest.raises(KeyError):
        coord.add_blocks("ghost", own=1)
