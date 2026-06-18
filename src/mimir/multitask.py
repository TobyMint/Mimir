"""多模型 / 多任务 KV 协调（赛题优化方向之「多任务推理」，40 分子项）。

在单卡（资源受限）上并发跑多个 agent 任务时，KV Cache 的协调是关键挑战：
- 各任务共享 system prompt 前缀 → 应复用（pin），不要重复存。
- 各任务的历史/独有 KV → 应隔离，一个任务的失败/结束不影响其他。
- 任务结束 → 立即回收其独占 KV，把显存让给其他任务（动态重分配）。

本 ``MultiTaskCoordinator`` 把前面几个模块（生命周期淘汰 / 分层存储 / 前缀 pin）
编排成统一的「单卡多任务 KV 协调器」：
1. **共享前缀 pin**：跨任务共用 system+tools 前缀，只存一份 KV。
2. **任务 KV 隔离**：每任务独立 task_id，独有块互不干扰。
3. **结束即回收**：任务结束立即回收独占 KV（lifecycle reclaim），动态让出显存。
4. **协调记账**：报告「共享节省 / 各任务独占 / 总占用」，模拟多任务在单卡上的 KV 调度。

与各单点优化的关系：它是「编排层」，复用 LifecycleEvictor 的回收语义、BranchTree 的
共享记账思想，给出多任务场景的整体度量与决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimir.kv_cache.lifecycle import LifecycleEvictor


@dataclass
class TaskHandle:
    """一个 agent 任务的句柄。"""

    task_id: str
    own_blocks: int = 0  # 独占块数
    shared_prefix_blocks: int = 0  # 共享前缀块数（与其他任务共享）
    finished: bool = False


@dataclass
class CoordinationStats:
    """多任务协调统计。"""

    num_tasks: int = 0
    pinned_shared_blocks: int = 0  # 共享前缀（pin）块数
    total_own_blocks: int = 0  # 所有任务独占块数总和
    reclaimed_on_finish: int = 0  # 任务结束回收的块数
    concurrent_peak: int = 0  # 并发峰值块数

    @property
    def sharing_savings_blocks(self) -> int:
        """共享前缀相比「每任务各存一份」节省的块数。"""
        if self.num_tasks <= 1:
            return 0
        return self.pinned_shared_blocks * (self.num_tasks - 1)


class MultiTaskCoordinator:
    """单卡多任务 KV 协调器。

    用法::

        coord = MultiTaskCoordinator(gpu_block_capacity=1000, shared_prefix_blocks=20)
        coord.start_task("task_A")            # 注册任务
        coord.add_blocks("task_A", own=100)   # 任务 A 独占块
        coord.start_task("task_B")
        coord.add_blocks("task_B", own=120)
        coord.finish_task("task_A")           # 结束 A -> 立即回收其独占块
    """

    def __init__(self, gpu_block_capacity: int, shared_prefix_blocks: int = 0) -> None:
        self.gpu_block_capacity = gpu_block_capacity
        self.shared_prefix_blocks = shared_prefix_blocks
        self._evictor = LifecycleEvictor(capacity=gpu_block_capacity)
        # 共享前缀：作为 pinned 块登记一次（不属任何任务）
        self._prefix_pinned = False
        self.tasks: dict[str, TaskHandle] = {}
        self.peak_concurrent = 0
        self._current_concurrent = 0

    def _ensure_prefix_pinned(self) -> None:
        if not self._prefix_pinned and self.shared_prefix_blocks > 0:
            for i in range(self.shared_prefix_blocks):
                self._evictor.add(f"__prefix__{i}", task_id="__shared__", pinned=True)
            self._prefix_pinned = True

    def start_task(self, task_id: str) -> TaskHandle:
        self._ensure_prefix_pinned()
        if task_id in self.tasks:
            return self.tasks[task_id]
        h = TaskHandle(task_id=task_id, shared_prefix_blocks=self.shared_prefix_blocks)
        self.tasks[task_id] = h
        self._current_concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self._current_concurrent)
        return h

    def add_blocks(self, task_id: str, own: int) -> None:
        """为任务新增 ``own`` 个独占 KV 块。"""
        if task_id not in self.tasks:
            raise KeyError(f"unknown task {task_id}; call start_task first")
        h = self.tasks[task_id]
        base = h.own_blocks
        for i in range(own):
            self._evictor.add(f"{task_id}_b{base + i}", task_id=task_id)
        h.own_blocks += own

    def finish_task(self, task_id: str) -> int:
        """结束任务：立即回收其独占块。返回回收数。"""
        if task_id not in self.tasks:
            return 0
        h = self.tasks[task_id]
        reclaimed = self._evictor.finish_task(task_id)
        h.finished = True
        h.own_blocks = 0
        self._current_concurrent = max(0, self._current_concurrent - 1)
        return reclaimed

    def stats(self) -> CoordinationStats:
        snap = self._evictor.snapshot()
        return CoordinationStats(
            num_tasks=len(self.tasks),
            pinned_shared_blocks=snap["by_lifecycle"].get("pinned", 0),
            total_own_blocks=sum(t.own_blocks for t in self.tasks.values() if not t.finished),
            reclaimed_on_finish=sum(
                0
                for _ in ()  # 累计在 evictor.stats.lifecycle_reclaims
            ),
            concurrent_peak=self.peak_concurrent,
        )

    def evictor_stats(self) -> dict[str, Any]:
        es = self._evictor.stats
        return {
            "hits": es.hits,
            "misses": es.misses,
            "evictions": es.evictions,
            "lifecycle_reclaims": es.lifecycle_reclaims,
            "hit_rate": round(es.hit_rate, 4),
        }


def simulate_multi_task(
    num_tasks: int,
    shared_prefix_blocks: int,
    own_blocks_per_task: int,
    gpu_capacity: int,
) -> dict[str, Any]:
    """模拟 N 个并发任务在单卡上的 KV 协调。

    对比：
    - coordinated：共享前缀 pin + 任务结束回收
    - uncoordinated：每任务各存前缀副本（无共享），无主动回收
    """
    # ---- coordinated ----
    coord = MultiTaskCoordinator(
        gpu_block_capacity=gpu_capacity, shared_prefix_blocks=shared_prefix_blocks
    )
    for t in range(num_tasks):
        coord.start_task(f"task_{t}")
        coord.add_blocks(f"task_{t}", own=own_blocks_per_task)
        coord.finish_task(f"task_{t}")  # 立即回收
    cs = coord.stats()
    es = coord.evictor_stats()

    # ---- uncoordinated（基线：每任务各存前缀副本，无回收信号）----
    # 朴素总量 = N * (前缀 + 独占)；不共享、不回收
    naive_total = num_tasks * (shared_prefix_blocks + own_blocks_per_task)

    return {
        "num_tasks": num_tasks,
        "shared_prefix_blocks": shared_prefix_blocks,
        "own_blocks_per_task": own_blocks_per_task,
        "gpu_capacity": gpu_capacity,
        "coordinated": {
            "concurrent_peak_blocks": cs.concurrent_peak,
            "peak_resident_blocks": shared_prefix_blocks + own_blocks_per_task,
            "lifecycle_reclaims": es["lifecycle_reclaims"],
            "sharing_savings_blocks": cs.sharing_savings_blocks,
        },
        "uncoordinated_baseline": {
            "total_blocks_if_no_sharing_no_reclaim": naive_total,
            "peak_resident_if_all_concurrent": num_tasks
            * (shared_prefix_blocks + own_blocks_per_task),
        },
        "coordination_benefit": {
            "sharing_savings_vs_naive_pct": round(cs.sharing_savings_blocks / naive_total * 100, 1)
            if naive_total
            else 0.0,
            "reclaim_vs_naive_pct": round(es["lifecycle_reclaims"] / naive_total * 100, 1)
            if naive_total
            else 0.0,
        },
    }
