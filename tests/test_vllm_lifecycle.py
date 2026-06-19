"""Phase C: v1 BlockPool mimir lifecycle 方法的逻辑测试。

用最小 mock 模拟 BlockPool 的块/缓存结构，验证 mimir_finish_task 的主动回收语义。
（不依赖真实 vLLM 引擎 / GPU。）
"""

from __future__ import annotations


class _FakeBlock:
    def __init__(self, bid):
        self.block_id = bid
        self.block_hash = None
        self.ref_cnt = 0
        self.is_null = False

    def reset_hash(self):
        self.block_hash = None


class _FakeBlockPool:
    """复刻 BlockPool 里 mimir patch 依赖的最小结构。"""

    def __init__(self, num_blocks=10):
        self.blocks = [_FakeBlock(i) for i in range(num_blocks)]
        self.num_gpu_blocks = num_blocks
        self.cached_block_hash_to_block = {}
        self.enable_kv_cache_events = False
        # Mimir patch 字段
        self.mimir_block_task = {}
        self.mimir_block_lifecycle = {}
        self.mimir_lifecycle_reclaims = 0
        self.mimir_used_blocks = 0
        self.mimir_cow_reuses = 0
        self.mimir_pin_hits = 0
        # free 队列（用 list 模拟）
        self._free = list(self.blocks)
        self.kv_event_queue = []

    # 复刻 mimir patch 的 mimir_finish_task（与 in-tree 代码一致）
    def mimir_finish_task(self, task_id):
        if task_id is None:
            return 0
        reclaimed = 0
        task_block_ids = [bid for bid, tid in self.mimir_block_task.items() if tid == task_id]
        for bid in task_block_ids:
            lc = self.mimir_block_lifecycle.get(bid, "active")
            if lc == "pinned":
                self.mimir_block_lifecycle[bid] = "evictable"
                self.mimir_pin_hits += 1
                continue
            block = self.blocks[bid]
            if block.ref_cnt != 0:
                self.mimir_block_lifecycle[bid] = "evictable"
                continue
            bh = block.block_hash
            if bh is not None:
                block.reset_hash()
                by_id = self.cached_block_hash_to_block.get(bh)
                if by_id is not None:
                    by_id.pop(bid, None)
                    if not by_id:
                        del self.cached_block_hash_to_block[bh]
            self._free.append(block)
            self.mimir_block_task.pop(bid, None)
            self.mimir_block_lifecycle.pop(bid, None)
            self.mimir_used_blocks = max(0, self.mimir_used_blocks - 1)
            reclaimed += 1
        self.mimir_lifecycle_reclaims += reclaimed
        return reclaimed

    def mimir_pin_blocks(self, block_ids):
        n = 0
        for bid in block_ids:
            if bid in self.mimir_block_lifecycle:
                self.mimir_block_lifecycle[bid] = "pinned"
                n += 1
        return n

    def mimir_get_task_block_ids(self, task_id):
        return [bid for bid, tid in self.mimir_block_task.items() if tid == task_id]

    def mimir_reclaim_evictable(self):
        reclaimed = 0
        evictable_ids = [bid for bid, lc in self.mimir_block_lifecycle.items() if lc == "evictable"]
        for bid in evictable_ids:
            block = self.blocks[bid]
            if block.ref_cnt != 0:
                continue
            bh = block.block_hash
            if bh is not None:
                block.reset_hash()
                by_id = self.cached_block_hash_to_block.get(bh)
                if by_id is not None:
                    by_id.pop(bid, None)
                    if not by_id:
                        del self.cached_block_hash_to_block[bh]
            self._free.append(block)
            self.mimir_block_task.pop(bid, None)
            self.mimir_block_lifecycle.pop(bid, None)
            self.mimir_used_blocks = max(0, self.mimir_used_blocks - 1)
            reclaimed += 1
        self.mimir_lifecycle_reclaims += reclaimed
        return reclaimed


def _occupy(bp, task_id, block_ids, ref_cnts=None, hashes=None):
    """模拟 cache_full_blocks 标记一段块归 task_id（可设 ref_cnt / hash）。"""
    for i, bid in enumerate(block_ids):
        bp.mimir_block_task[bid] = task_id
        bp.mimir_block_lifecycle[bid] = "active"
        blk = bp.blocks[bid]
        blk.ref_cnt = 0 if ref_cnts is None else ref_cnts[i]
        if hashes:
            blk.block_hash = hashes[i]
            bp.cached_block_hash_to_block.setdefault(hashes[i], {})[bid] = blk
        # 从 free 中移除
        if blk in bp._free:
            bp._free.remove(blk)
            bp.mimir_used_blocks += 1


def test_finish_task_reclaims_free_zero_refcnt_blocks() -> None:
    bp = _FakeBlockPool(num_blocks=10)
    _occupy(bp, "t1", [1, 2, 3], hashes=[b"h1", b"h2", b"h3"])
    assert len(bp._free) == 10 - 3
    n = bp.mimir_finish_task("t1")
    assert n == 3
    assert bp.mimir_lifecycle_reclaims == 3
    # 块回到 free，hash 清空，task 归属清除
    assert len(bp._free) == 10
    assert all(blk.block_hash is None for blk in bp.blocks[1:4])
    assert bp.mimir_block_task == {}


def test_finish_task_keeps_referenced_blocks_as_evictable() -> None:
    bp = _FakeBlockPool(num_blocks=10)
    _occupy(bp, "t1", [1, 2], ref_cnts=[0, 1], hashes=[b"h1", b"h2"])
    n = bp.mimir_finish_task("t1")
    assert n == 1  # 只回收 ref_cnt==0 的块 1
    # 块 2 仍被引用 -> 标记 evictable 但不回收
    assert bp.mimir_block_lifecycle.get(2) == "evictable"
    assert 2 in bp.mimir_block_task


def test_finish_task_skips_pinned_blocks() -> None:
    bp = _FakeBlockPool(num_blocks=10)
    _occupy(bp, "t1", [1, 2], hashes=[b"h1", b"h2"])
    bp.mimir_pin_blocks([1])  # pin 块 1
    n = bp.mimir_finish_task("t1")
    assert n == 1  # 只回收未 pin 的块 2
    # 块 1 pin -> 标记 evictable（保留可被后续压力淘汰）
    assert bp.mimir_block_lifecycle.get(1) == "evictable"
    assert 1 in bp.mimir_block_task  # 块 1 仍记录归属
    assert bp.mimir_pin_hits == 1  # pin 阻止了一次回收


def test_finish_task_none_returns_zero() -> None:
    bp = _FakeBlockPool()
    assert bp.mimir_finish_task(None) == 0


def test_finish_task_unknown_task_returns_zero() -> None:
    bp = _FakeBlockPool()
    _occupy(bp, "t1", [1], hashes=[b"h1"])
    assert bp.mimir_finish_task("ghost") == 0
    assert bp.mimir_lifecycle_reclaims == 0


def test_get_task_block_ids() -> None:
    bp = _FakeBlockPool()
    _occupy(bp, "t1", [1, 2])
    _occupy(bp, "t2", [3])
    assert set(bp.mimir_get_task_block_ids("t1")) == {1, 2}
    assert bp.mimir_get_task_block_ids("t2") == [3]


def test_reclaim_evictable_sweeps_marked_blocks() -> None:
    """Phase J：mimir_reclaim_evictable 主动回收所有 EVICTABLE 块。"""
    bp = _FakeBlockPool(num_blocks=10)
    _occupy(bp, "t1", [1, 2], ref_cnts=[1, 0], hashes=[b"h1", b"h2"])  # 块1 被引用, 块2 空闲
    # finish_task 把块1（被引用）标记 evictable，块2 直接回收
    bp.mimir_finish_task("t1")
    assert bp.mimir_block_lifecycle.get(1) == "evictable"
    # 模拟块1 引用释放（ref_cnt -> 0）后调 reclaim_evictable
    bp.blocks[1].ref_cnt = 0
    n = bp.mimir_reclaim_evictable()
    assert n == 1  # 回收块1
    assert bp.mimir_block_task == {}
    assert bp.mimir_lifecycle_reclaims >= 1


def test_reclaim_evictable_skips_still_referenced() -> None:
    bp = _FakeBlockPool(num_blocks=10)
    _occupy(bp, "t1", [1, 2], ref_cnts=[1, 1], hashes=[b"h1", b"h2"])
    bp.mimir_finish_task("t1")  # 两块都 evictable（被引用未回收）
    n = bp.mimir_reclaim_evictable()
    assert n == 0  # 都仍被引用，跳过
    assert len(bp.mimir_block_task) == 2
