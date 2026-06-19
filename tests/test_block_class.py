"""Block-class 创新（in-tree patch）的确定性单元测试。

不依赖 GPU/真实引擎：直接构造 BlockPool，手工给块打 mimir_block_class 标签，
验证：
1. mimir_class_aware_evict 按「reasoning > user > tool_result > system」优先级淘汰；
2. 高价值类别 (system/tool_result) 在 reasoning 未耗尽前不被淘汰；
3. mimir_class_stats 正确导出按类别淘汰数。
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")  # 跳过无 vllm 环境
from vllm.v1.core.block_pool import BlockPool  # noqa: E402


def _make_pool(n: int = 20) -> BlockPool:
    return BlockPool(num_gpu_blocks=n, enable_caching=True, enable_kv_cache_events=False)


def _tag_blocks(bp: BlockPool, mapping: dict[int, str]) -> None:
    """手工给指定 block_id 打类别标签 + 生命周期 + 伪造 cached hash，使其可被淘汰。"""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    for bid, cls in mapping.items():
        bp.mimir_block_class[bid] = cls
        bp.mimir_block_lifecycle[bid] = "evictable"
        bp.mimir_block_task[bid] = "task_x"
        blk = bp.blocks[bid]
        # 伪造 block_hash（bytes）使其「在 cache 中、可被 reset_hash 淘汰」
        if blk.block_hash is None:
            bh = make_block_hash_with_group_id(bid.to_bytes(8, "big"), 0)
            blk.block_hash = bh
            bp.cached_block_hash_to_block[bh][bid] = blk
        bp.mimir_used_blocks += 1


def test_class_aware_evict_prefers_reasoning():
    bp = _make_pool(20)
    _tag_blocks(bp, {1: "reasoning", 2: "tool_result", 3: "system", 4: "reasoning"})
    # 需淘汰 1 块 -> 应淘汰 reasoning（block 1 或 4）
    n = bp.mimir_class_aware_evict(1)
    assert n == 1
    ev = bp.mimir_class_stats()["class_evicts"]
    assert ev["reasoning"] == 1
    assert ev["tool_result"] == 0 and ev["system"] == 0


def test_class_aware_evict_priority_order():
    bp = _make_pool(20)
    _tag_blocks(bp, {
        1: "reasoning", 2: "reasoning",
        3: "user",
        4: "tool_result", 5: "tool_result",
        6: "system",
    })
    # 需淘汰 4 块：应按 reasoning(2) -> user(1) -> tool_result(1) 顺序，system 保留
    n = bp.mimir_class_aware_evict(4)
    assert n == 4, f"expected 4 evicted, got {n}"
    ev = bp.mimir_class_stats()["class_evicts"]
    assert ev["reasoning"] == 2
    assert ev["user"] == 1
    assert ev["tool_result"] == 1
    assert ev["system"] == 0, "system 应最后才淘汰，此处必须存活"


def test_class_aware_evict_skips_pinned():
    bp = _make_pool(20)
    _tag_blocks(bp, {1: "reasoning", 2: "tool_result"})
    # 把 reasoning 块 pin 住
    bp.mimir_block_lifecycle[1] = "pinned"
    n = bp.mimir_class_aware_evict(1)
    assert n == 1
    # reasoning 被 pin 跳过，淘汰落到下一优先级（tool_result 在 reasoning 之后，
    # 但 reasoning 候选已耗尽 -> user 无 -> tool_result）
    ev = bp.mimir_class_stats()["class_evicts"]
    # reasoning 被 pin 0 淘汰，tool_result 1
    assert ev["reasoning"] == 0
    assert ev["tool_result"] == 1


def test_class_stats_counts_reflect_tags():
    bp = _make_pool(20)
    _tag_blocks(bp, {1: "system", 2: "reasoning", 3: "tool_result", 4: "reasoning"})
    cs = bp.mimir_class_stats()["block_class_counts"]
    assert cs == {"system": 1, "reasoning": 2, "tool_result": 1}, cs


def test_evict_zero_need_is_noop():
    bp = _make_pool(20)
    _tag_blocks(bp, {1: "reasoning"})
    assert bp.mimir_class_aware_evict(0) == 0
    assert bp.mimir_class_stats()["class_evicts"]["reasoning"] == 0
