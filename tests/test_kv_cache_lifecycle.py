"""``mimir.kv_cache.lifecycle`` 单元测试（纯逻辑）。"""

from __future__ import annotations

from mimir.kv_cache.lifecycle import (
    BlockLifecycle,
    LifecycleEvictor,
    PureLRUEvictor,
    simulate_agent_trace,
)


def test_access_hit_miss() -> None:
    e = LifecycleEvictor(capacity=4)
    e.add("a", "task_0")
    assert e.access("a") is True  # 命中
    assert e.access("nope") is False  # 未命中
    assert e.stats.hits == 1
    assert e.stats.misses == 1


def test_finish_task_reclaims_blocks() -> None:
    e = LifecycleEvictor(capacity=10)
    for b in range(5):
        e.add(f"t0_b{b}", "task_0")
    reclaimed = e.finish_task("task_0")
    assert reclaimed == 5
    assert e.stats.lifecycle_reclaims == 5
    assert len(e._blocks) == 0  # noqa: SLF001


def test_pinned_not_reclaimed() -> None:
    e = LifecycleEvictor(capacity=10)
    e.add("sys", "task_0", pinned=True)  # 钉住
    e.add("a", "task_0")
    e.add("b", "task_0")
    reclaimed = e.finish_task("task_0")
    assert reclaimed == 2  # 仅 a,b 回收；sys 保留
    assert "sys" in e._blocks  # noqa: SLF001


def test_capacity_eviction_prefers_evictable() -> None:
    e = LifecycleEvictor(capacity=2)
    e.add("a", "task_0")
    e.add("b", "task_0")
    e.mark_task_finished("task_0")  # a,b -> EVICTABLE
    e.add("c", "task_1")  # 容量压力，应淘汰 EVICTABLE 而非 c
    # a 或 b 应被淘汰（EVICTABLE），c 保留
    assert "c" in e._blocks  # noqa: SLF001
    assert e.stats.evictions >= 1


def test_lifecycle_beats_lru_on_agent_trace() -> None:
    """同一 trace：lifecycle 因任务结束主动回收，命中率应 >= LRU（且回收更多）。"""
    lc = LifecycleEvictor(capacity=8)
    lru = PureLRUEvictor(capacity=8)
    s_lc = simulate_agent_trace(lc, num_tasks=5, blocks_per_task=3, reuse_within_task=2, capacity=8)
    s_lru = simulate_agent_trace(
        lru, num_tasks=5, blocks_per_task=3, reuse_within_task=2, capacity=8
    )
    # lifecycle 主动回收了任务块
    assert s_lc.lifecycle_reclaims > 0
    assert s_lru.lifecycle_reclaims == 0
    # lifecycle 命中率不应低于 LRU（主动回收腾出空间给活跃任务）
    assert s_lc.hit_rate >= s_lru.hit_rate


def test_mark_task_finished_sets_evictable() -> None:
    e = LifecycleEvictor(capacity=10)
    e.add("a", "task_0")
    n = e.mark_task_finished("task_0")
    assert n == 1
    assert e._blocks["a"].lifecycle is BlockLifecycle.EVICTABLE  # noqa: SLF001


def test_snapshot() -> None:
    e = LifecycleEvictor(capacity=5)
    e.add("a", "task_0", pinned=True)
    e.add("b", "task_0")
    snap = e.snapshot()
    assert snap["total"] == 2
    assert snap["by_lifecycle"]["pinned"] == 1
    assert snap["by_lifecycle"]["active"] == 1
