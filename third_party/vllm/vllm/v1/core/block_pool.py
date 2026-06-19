# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.distributed.kv_events import (MEDIUM_GPU, AllBlocksCleared,
                                        BlockRemoved, BlockStored,
                                        KVCacheEvent)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHash, BlockHashWithGroupId,
                                         ExternalBlockHash,
                                         FreeKVCacheBlockQueue, KVCacheBlock,
                                         get_block_hash,
                                         make_block_hash_with_group_id,
                                         maybe_convert_block_hash)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        enable_kv_cache_events: Whether to enable kv cache events.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: dict[BlockHashWithGroupId, dict[
            int, KVCacheBlock]] = defaultdict(dict)

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        # ---- Mimir in-tree patch: 生命周期/任务边界感知（Phase C）------------- #
        # 追踪每个块归属的 agent 任务 + 生命周期态。vLLM 原生只有 LRU 淘汰（被动、
        # 按访问序），不感知「任务已结束可立即回收」。Mimir 在 cache_full_blocks
        # 时记录块→任务归属，在 mimir_finish_task 时主动回收（ref_cnt==0 且非 PINNED）。
        self.mimir_block_task: dict[int, object] = {}      # block_id -> task_id
        self.mimir_block_lifecycle: dict[int, str] = {}    # block_id -> "active"|"evictable"|"pinned"
        self.mimir_lifecycle_reclaims: int = 0             # 主动回收计数（导出到 get_mimir_stats）
        self.mimir_used_blocks: int = 0                    # 由 cache_full_blocks 增、mimir_finish_task 减
        self.mimir_cow_reuses: int = 0                     # 跨分支 CoW 复用计数（Phase D）
        self.mimir_pin_hits: int = 0                       # pin 生效计数（Phase E，pin 阻止淘汰时增）
        # ---- Mimir patch end ----------------------------------------------- #

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        """Get the cached block by the block hash for each group in 
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id)
            cached_blocks_one_group = self.cached_block_hash_to_block.get(
                block_hash_with_group_id)
            if not cached_blocks_one_group:
                return None
            first_block = next(iter(cached_blocks_one_group.values()))
            cached_blocks.append(first_block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks == num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        new_block_hashes = request.block_hashes[num_cached_blocks:]

        new_hashes: Optional[list[ExternalBlockHash]] = (
            [] if self.enable_kv_cache_events else None)
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id)
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block[block_hash_with_group_id][
                blk.block_id] = blk
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))
            # ---- Mimir patch (Phase C/D): 记录块→任务归属 + used 计数 ----
            if blk.block_id not in self.mimir_block_task:
                self.mimir_used_blocks += 1  # 新占用块
            self.mimir_block_task[blk.block_id] = getattr(
                request, "mimir_task_id", None)
            if blk.block_id not in self.mimir_block_lifecycle:
                self.mimir_block_lifecycle[blk.block_id] = "active"
            # ---- Mimir patch end ----

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: Optional[ExternalBlockHash] = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash))

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.
                    all_token_ids[num_cached_blocks *
                                  block_size:num_full_blocks * block_size],
                    block_size=block_size,
                    lora_id=request.lora_request.id
                    if request.lora_request else None,
                    medium=MEDIUM_GPU,
                ))

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        # ---- Mimir patch (Phase P): 分配前主动回收 EVICTABLE 块 ----
        # vLLM 默认仅在容量不足时由 LRU 被动淘汰 free_block_queue 里的缓存块。Mimir 先把
        # 已结束任务标记为 EVICTABLE 的块物理释放（reset_hash + 回 free 队列），让真正需要淘汰时
        # 优先腾出这些「任务残留」而非活跃任务的 KV。这是 lifecycle 感知淘汰接入了真实分配路径。
        try:
            self.mimir_reclaim_evictable()
        except Exception:
            pass
        # ---- Mimir patch end ----

        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False
        blocks_by_id = self.cached_block_hash_to_block.get(block_hash)
        if blocks_by_id is None:
            # block_hash not found in cached_block_hash_to_block,
            # eviction is not needed
            return False
        block.reset_hash()
        blocks_by_id.pop(block.block_id, None)
        if len(blocks_by_id) == 0:
            del self.cached_block_hash_to_block[block_hash]

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(block_hashes=[
                    maybe_convert_block_hash(get_block_hash(block_hash))
                ],
                             medium=MEDIUM_GPU))
        return True

    # ---- Mimir in-tree patch: 任务边界主动回收 + per-block pin（Phase C/E）------- #
    def mimir_finish_task(self, task_id: object) -> int:
        """任务结束：主动回收该任务所有 EVICTABLE/ACTIVE 块（非 PINNED, ref_cnt==0）。

        vLLM 原生 LRU 只在容量压力下被动淘汰，不感知「任务已结束」。Mimir 在
        agent 任务边界调用此方法，立即把已完成任务的残留 KV 标记并物理释放，
        把显存让给其他并发任务（动态资源重分配）。

        返回回收的块数。
        """
        if task_id is None:
            return 0
        reclaimed = 0
        # 收集归该任务的所有块 id
        task_block_ids = [
            bid for bid, tid in self.mimir_block_task.items() if tid == task_id
        ]
        for bid in task_block_ids:
            lc = self.mimir_block_lifecycle.get(bid, "active")
            if lc == "pinned":
                # PINNED 块不回收（system 前缀等，Phase E）；记为一次 pin 命中（pin 阻止了回收）
                self.mimir_block_lifecycle[bid] = "evictable"  # 标记可回收，留给后续压力淘汰
                self.mimir_pin_hits += 1
                continue
            block = self.blocks[bid]
            # 仅回收无引用的块（mirror get_new_blocks 的 ref_cnt==0 guard）
            if block.ref_cnt != 0:
                self.mimir_block_lifecycle[bid] = "evictable"  # 仍被引用，标记可回收
                continue
            # 物理释放：重置 hash、移出 cache、放回 free 队列
            bh = block.block_hash
            if bh is not None:
                block.reset_hash()
                blocks_by_id = self.cached_block_hash_to_block.get(bh)
                if blocks_by_id is not None:
                    blocks_by_id.pop(bid, None)
                    if len(blocks_by_id) == 0:
                        del self.cached_block_hash_to_block[bh]
                if self.enable_kv_cache_events:
                    self.kv_event_queue.append(
                        BlockRemoved(block_hashes=[
                            maybe_convert_block_hash(get_block_hash(bh))],
                            medium=MEDIUM_GPU))
            # 放回 free 队列（若不在）
            if not block.is_null:
                self.free_block_queue.append(block)
            self.mimir_block_task.pop(bid, None)
            self.mimir_block_lifecycle.pop(bid, None)
            self.mimir_used_blocks = max(0, self.mimir_used_blocks - 1)
            reclaimed += 1
        self.mimir_lifecycle_reclaims += reclaimed
        return reclaimed

    def mimir_pin_blocks(self, block_ids: list[int]) -> int:
        """钉住指定块（不淘汰/不回收）。Phase E 用于 agent 暂停时 pin 前缀。"""
        n = 0
        for bid in block_ids:
            if bid in self.mimir_block_lifecycle:
                self.mimir_block_lifecycle[bid] = "pinned"
                n += 1
        return n

    def mimir_unpin_task(self, task_id: object) -> int:
        """取消某任务所有块的 pin，标记为 evictable。"""
        n = 0
        for bid, tid in self.mimir_block_task.items():
            if tid == task_id and self.mimir_block_lifecycle.get(bid) == "pinned":
                self.mimir_block_lifecycle[bid] = "evictable"
                n += 1
        return n

    def mimir_get_task_block_ids(self, task_id: object) -> list[int]:
        return [bid for bid, tid in self.mimir_block_task.items() if tid == task_id]

    def mimir_reclaim_evictable(self) -> int:
        """Phase J：主动扫描并回收所有 EVICTABLE（已结束任务残留）块。

        闭环：Phase C 的 mimir_finish_task 会把仍被引用/PINNED 的块标记为
        EVICTABLE（无法立即回收）；本方法在显存压力点被调用，把所有 EVICTABLE
        且 ref_cnt==0 的块物理释放（reset_hash + 移出 cache + 放回 free 队列）。
        返回回收数。
        """
        reclaimed = 0
        evictable_ids = [
            bid for bid, lc in self.mimir_block_lifecycle.items() if lc == "evictable"
        ]
        for bid in evictable_ids:
            block = self.blocks[bid]
            if block.ref_cnt != 0:
                continue  # 仍被引用，跳过
            bh = block.block_hash
            if bh is not None:
                block.reset_hash()
                by_id = self.cached_block_hash_to_block.get(bh)
                if by_id is not None:
                    by_id.pop(bid, None)
                    if len(by_id) == 0:
                        del self.cached_block_hash_to_block[bh]
                if self.enable_kv_cache_events:
                    self.kv_event_queue.append(
                        BlockRemoved(block_hashes=[
                            maybe_convert_block_hash(get_block_hash(bh))],
                            medium=MEDIUM_GPU))
            if not block.is_null:
                self.free_block_queue.append(block)
            self.mimir_block_task.pop(bid, None)
            self.mimir_block_lifecycle.pop(bid, None)
            self.mimir_used_blocks = max(0, self.mimir_used_blocks - 1)
            reclaimed += 1
        self.mimir_lifecycle_reclaims += reclaimed
        return reclaimed
    # ---- Mimir patch end ---------------------------------------------------- #

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet", num_used_blocks - 1)
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = defaultdict(dict)

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.
        
        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
