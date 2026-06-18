"""KV Cache 生命周期管理 / 淘汰（赛题优化方向之一）。

设计面向长生命周期推理的缓存管理策略，实现 KV 的复用、淘汰与分层存储，以控制显存增长，
并支持不同生命周期任务下的动态资源回收与重分配。详见 ``docs/技术方案.md`` §3.1。

与 vLLM APC 的区别
------------------
vLLM APC 的淘汰是**纯 LRU**：只在显存不够时被动淘汰最久未用的块，不感知：
- 「这个块属于哪个 agent 任务」
- 「这个任务是否已结束」（结束即可立即回收，无需等容量压力）

Mimir ``LifecycleEvictor`` 提供任务语义的淘汰：
1. 每个缓存块标记所属 ``task_id`` 与 ``lifecycle``（ACTIVE/EVICTABLE/PINNED）。
2. 任务结束时（``finish_task``），其非 PINNED 块**立即**标记为可回收并释放。
3. 容量压力下，优先淘汰 EVICTABLE（已结束任务的残留），再淘汰最旧 ACTIVE。
4. 提供 LRU 基线对比：同一访问 trace 下，lifecycle 命中率 vs LRU 命中率。

本模块是策略与记账（不直接操作 vLLM 块）。可作为「回收信号」驱动引擎层释放。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BlockLifecycle(str, Enum):
    ACTIVE = "active"  # 属于进行中的任务
    EVICTABLE = "evictable"  # 可立即回收（任务已结束）
    PINNED = "pinned"  # 常驻（如 system prompt 前缀）


@dataclass
class KVBlockMeta:
    block_id: str
    task_id: str
    size_tokens: int = 1
    lifecycle: BlockLifecycle = BlockLifecycle.ACTIVE
    last_access_turn: int = 0


@dataclass
class EvictionStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    lifecycle_reclaims: int = 0  # 任务结束主动回收的块数

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class LifecycleEvictor:
    """任务语义感知的 KV 淘汰管理器。

    - ``capacity``：最多保留的块数（模拟显存块上限）。
    - ``access(block_id)``：访问一个块（命中/未命中），更新 LRU 顺序。
    - ``finish_task(task_id)``：任务结束，其非 PINNED 块立即回收（lifecycle reclaim）。
    - ``pin(block_id)``：钉住（system 前缀等）。
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # OrderedDict[block_id, KVBlockMeta]，按访问序
        self._blocks: OrderedDict[str, KVBlockMeta] = OrderedDict()
        self.stats = EvictionStats()
        self._turn = 0

    def add(
        self, block_id: str, task_id: str, *, size_tokens: int = 1, pinned: bool = False
    ) -> None:
        """新增一个块（来自 task_id）。"""
        meta = KVBlockMeta(
            block_id=block_id,
            task_id=task_id,
            size_tokens=size_tokens,
            lifecycle=BlockLifecycle.PINNED if pinned else BlockLifecycle.ACTIVE,
            last_access_turn=self._turn,
        )
        self._blocks[block_id] = meta
        self._evict_if_needed()

    def access(self, block_id: str) -> bool:
        """访问一个块，返回是否命中。命中则更新 LRU 顺序。"""
        self._turn += 1
        if block_id in self._blocks:
            self._blocks.move_to_end(block_id)
            self._blocks[block_id].last_access_turn = self._turn
            self.stats.hits += 1
            return True
        self.stats.misses += 1
        return False

    def finish_task(self, task_id: str) -> int:
        """任务结束：回收其所有非 PINNED 块。返回回收块数。"""
        reclaimed = 0
        to_remove = [
            bid
            for bid, m in self._blocks.items()
            if m.task_id == task_id and m.lifecycle is not BlockLifecycle.PINNED
        ]
        for bid in to_remove:
            del self._blocks[bid]
            reclaimed += 1
        self.stats.lifecycle_reclaims += reclaimed
        return reclaimed

    def pin(self, block_id: str) -> bool:
        if block_id in self._blocks:
            self._blocks[block_id].lifecycle = BlockLifecycle.PINNED
            return True
        return False

    def _evict_if_needed(self) -> None:
        """超容时：先淘汰 EVICTABLE（任务残留），再淘汰最旧 ACTIVE。PINNED 不动。"""
        while (
            sum(1 for m in self._blocks.values() if m.lifecycle is not BlockLifecycle.PINNED)
            > self.capacity
        ):
            # 优先淘汰 EVICTABLE
            victim = None
            for bid, m in self._blocks.items():
                if m.lifecycle is BlockLifecycle.EVICTABLE:
                    victim = bid
                    break
            if victim is None:
                # 淘汰最旧 ACTIVE（OrderedDict 首项）
                for bid, m in self._blocks.items():
                    if m.lifecycle is BlockLifecycle.ACTIVE:
                        victim = bid
                        break
            if victim is None:
                break  # 只剩 PINNED，无法再淘汰
            del self._blocks[victim]
            self.stats.evictions += 1

    def mark_task_finished(self, task_id: str) -> int:
        """把任务块标记为 EVICTABLE（但不立即删，等容量压力或显式 finish）。

        与 ``finish_task`` 区别：标记为可回收 vs 立即回收。
        """
        n = 0
        for m in self._blocks.values():
            if m.task_id == task_id and m.lifecycle is BlockLifecycle.ACTIVE:
                m.lifecycle = BlockLifecycle.EVICTABLE
                n += 1
        return n

    def snapshot(self) -> dict[str, Any]:
        from collections import Counter

        lc = Counter(m.lifecycle.value for m in self._blocks.values())
        return {
            "total": len(self._blocks),
            "by_lifecycle": dict(lc),
            "capacity": self.capacity,
        }


class PureLRUEvictor:
    """纯 LRU 基线（对照 vLLM APC 行为）：无任务语义，仅按访问序淘汰。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._blocks: OrderedDict[str, None] = OrderedDict()
        self.stats = EvictionStats()

    def add(self, block_id: str, task_id: str | None = None, **_: Any) -> None:
        self._blocks[block_id] = None
        self._evict()

    def access(self, block_id: str) -> bool:
        if block_id in self._blocks:
            self._blocks.move_to_end(block_id)
            self.stats.hits += 1
            return True
        self.stats.misses += 1
        return False

    def finish_task(self, task_id: str | None = None) -> int:
        return 0  # LRU 不感知任务，不主动回收

    def mark_task_finished(self, task_id: str | None = None) -> int:
        return 0

    def pin(self, block_id: str) -> bool:
        return False

    def _evict(self) -> None:
        while len(self._blocks) > self.capacity:
            self._blocks.popitem(last=False)
            self.stats.evictions += 1

    def snapshot(self) -> dict[str, Any]:
        return {"total": len(self._blocks), "capacity": self.capacity}


def simulate_agent_trace(
    evictor: LifecycleEvictor | PureLRUEvictor,
    num_tasks: int,
    blocks_per_task: int,
    reuse_within_task: int,
    capacity: int,
) -> EvictionStats:
    """跑一段 agent 访问 trace，返回统计。

    每个 task 产生 ``blocks_per_task`` 个块，访问模式：本 task 内重复访问 ``reuse_within_task`` 次。
    task 结束时调用 finish_task（lifecycle 会回收，LRU 不会）。
    """
    for t in range(num_tasks):
        for b in range(blocks_per_task):
            bid = f"t{t}_b{b}"
            evictor.add(bid, task_id=f"task_{t}")
        # 本 task 内复用
        for _ in range(reuse_within_task):
            evictor.access(f"t{t}_b{b // 2}")
        evictor.finish_task(f"task_{t}")  # 任务结束
    return evictor.stats
