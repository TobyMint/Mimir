"""``mimir.tiered.store`` 单元测试（纯逻辑）。"""

from __future__ import annotations

from pathlib import Path

from mimir.tiered.store import MigrationPolicy, Tier, TieredStore


def test_put_into_gpu_until_cap_then_demote() -> None:
    store = TieredStore(gpu_cap=2, host_cap=4)
    store.put("a", 1)
    store.put("b", 2)
    assert store._tier_of("a") is Tier.GPU
    assert store._tier_of("b") is Tier.GPU
    # 第三项：a 被 demote 到 HOST
    store.put("c", 3)
    assert store._tier_of("a") is Tier.HOST
    assert store._tier_of("c") is Tier.GPU


def test_host_overflow_demotes_to_disk(tmp_path: Path) -> None:
    store = TieredStore(gpu_cap=1, host_cap=2, disk_dir=tmp_path)
    for i in range(6):
        store.put(f"k{i}", i)
    snap = store.snapshot()
    assert len(snap["gpu"]) == 1
    assert len(snap["host"]) == 2
    assert len(snap["disk"]) == 3  # 6 - 1 - 2
    # 最早进去的应已落盘
    assert "k0" in snap["disk"]


def test_get_promotes_to_gpu() -> None:
    store = TieredStore(gpu_cap=1, host_cap=4)
    store.put("a", 1)
    store.put("b", 2)  # a -> host
    assert store._tier_of("a") is Tier.HOST
    v = store.get("a")  # 访问 a -> promote 回 gpu
    assert v == 1
    assert store._tier_of("a") is Tier.GPU
    assert store.stats.promotions >= 1


def test_get_from_disk_reads_and_promotes(tmp_path: Path) -> None:
    store = TieredStore(gpu_cap=1, host_cap=1, disk_dir=tmp_path)
    store.put("a", "A")
    store.put("b", "B")  # a -> host
    store.put("c", "C")  # b -> host, a -> disk
    assert store._tier_of("a") is Tier.DISK
    v = store.get("a")  # disk -> ... -> gpu
    assert v == "A"
    assert store.stats.disk_reads >= 1


def test_missing_key_returns_none() -> None:
    store = TieredStore()
    assert store.get("nope") is None


def test_lru_order_in_gpu() -> None:
    """GPU 满时淘汰最久未访问的。"""
    store = TieredStore(gpu_cap=2, host_cap=4)
    store.put("a", 1)
    store.put("b", 2)
    store.get("a")  # a 变最新
    store.put("c", 3)  # 应淘汰 b（最旧）
    assert store._tier_of("a") is Tier.GPU
    assert store._tier_of("b") is Tier.HOST
    assert store._tier_of("c") is Tier.GPU


def test_migration_policy_make_store() -> None:
    pol = MigrationPolicy(gpu_cap=3, host_cap=10)
    store = pol.make_store()
    assert store.gpu_cap == 3
    assert store.host_cap == 10


def test_stats_dict() -> None:
    store = TieredStore(gpu_cap=1, host_cap=1)
    store.put("a", 1)
    store.put("b", 2)
    store.put("c", 3)  # a -> host -> disk (b)
    d = store.stats_dict()
    assert d["total"] == 3
    assert d["demotions"] >= 2
