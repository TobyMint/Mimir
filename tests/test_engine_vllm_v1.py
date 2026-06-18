"""``mimir.engine_vllm_v1`` 单元测试（纯逻辑，用 mock 验证遍历）。"""

from __future__ import annotations

from types import SimpleNamespace

from mimir.engine_vllm_v1 import (
    VLLMEngineV1,
    _resolve_v1_block_pool,
    _resolve_v1_scheduler,
    v1_kv_usage,
)


def _mock_llm(num_gpu_blocks: int = 100, free: int = 30):
    """构造一个最小 mock 模拟 v1 InprocClient 遍历路径。"""
    bp = SimpleNamespace(
        num_gpu_blocks=num_gpu_blocks,
        get_num_free_blocks=lambda: free,
        get_usage=lambda: (num_gpu_blocks - free) / num_gpu_blocks,
    )
    kvm = SimpleNamespace(block_pool=bp)
    sched = SimpleNamespace(
        kv_cache_manager=kvm,
        running=[1, 2],
        waiting=[],
        mimir_lifecycle_reclaims=5,
        mimir_cow_reuses=3,
        mimir_pin_hits=1,
    )
    inner = SimpleNamespace(scheduler=sched)
    ec = SimpleNamespace(engine_core=inner)  # InprocClient
    eng = SimpleNamespace(engine_core=ec)
    llm = SimpleNamespace(llm_engine=eng)
    return llm, sched, bp


def test_resolve_v1_block_pool_traversal() -> None:
    llm, _sched, bp = _mock_llm()
    assert _resolve_v1_block_pool(llm) is bp


def test_resolve_v1_scheduler_traversal() -> None:
    llm, sched, _bp = _mock_llm()
    assert _resolve_v1_scheduler(llm) is sched


def test_v1_kv_usage_reads_blocks() -> None:
    llm, _sched, _bp = _mock_llm(num_gpu_blocks=200, free=50)
    kv = v1_kv_usage(llm)
    assert kv["total_blocks"] == 200
    assert kv["used_blocks"] == 150
    assert kv["utilization"] == 0.75


def test_v1_kv_usage_graceful_on_none() -> None:
    # engine_core 不是 InprocClient（无 .engine_core）-> bp None -> 全 None
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace()))
    kv = v1_kv_usage(llm)
    assert kv["used_blocks"] is None
    assert kv["total_blocks"] is None


def test_mimir_stats_reads_scheduler_counters() -> None:
    """get_mimir_stats 应读到 in-tree patch 暴露的计数器。"""

    class FakeSched:
        def __init__(self):
            self.kv_cache_manager = SimpleNamespace(
                block_pool=SimpleNamespace(
                    num_gpu_blocks=100,
                    get_num_free_blocks=lambda: 30,
                    get_usage=lambda: 0.7,
                )
            )
            self.running = [1, 2]
            self.waiting = []
            self.mimir_lifecycle_reclaims = 5
            self.mimir_cow_reuses = 3
            self.mimir_pin_hits = 1

        def get_mimir_stats(self):
            bp = self.kv_cache_manager.block_pool
            total = bp.num_gpu_blocks
            used = total - bp.get_num_free_blocks()
            return {
                "used_blocks": used,
                "total_blocks": total,
                "mimir_lifecycle_reclaims": self.mimir_lifecycle_reclaims,
                "mimir_cow_reuses": self.mimir_cow_reuses,
                "mimir_pin_hits": self.mimir_pin_hits,
            }

    sched = FakeSched()
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=sched))
        )
    )
    e = VLLMEngineV1.__new__(VLLMEngineV1)
    e._llm = llm  # type: ignore[attr-defined]
    st = e.mimir_stats()
    assert st["total_blocks"] == 100
    assert st["used_blocks"] == 70
    assert st["mimir_lifecycle_reclaims"] == 5
    assert st["mimir_cow_reuses"] == 3
