"""核心数据结构与编排逻辑的单元测试（纯逻辑，无需模型权重或 GPU）。"""

from __future__ import annotations

import pytest

import mimir
from mimir import DeviceTier, KVBlock, Lifecycle, MemoryManager


def test_package_exposes_version() -> None:
    """包应导出版本号，且符合语义化版本的基本形态。"""
    assert isinstance(mimir.__version__, str)
    assert mimir.__version__ and mimir.__version__[0].isdigit()


def test_memory_manager_exports() -> None:
    """顶层应导出核心公共 API。"""
    assert hasattr(mimir, "MemoryManager")
    assert hasattr(mimir, "KVBlock")
    assert hasattr(mimir, "DeviceTier")
    assert hasattr(mimir, "Lifecycle")


def test_kv_block_default_state() -> None:
    block = KVBlock(block_id=0, seq_id="s1", token_range=(0, 16))
    assert block.ref_count == 1
    assert block.lifecycle is Lifecycle.ACTIVE
    assert block.device_tier is DeviceTier.GPU


def test_kv_block_ref_counting() -> None:
    """引用计数应正确支持共享 / CoW 语义。"""
    block = KVBlock(block_id=1, seq_id="s1", token_range=(0, 16))
    block.acquire()
    assert block.ref_count == 2
    assert block.release() is False  # 仍有引用
    assert block.release() is True  # 释放到 0
    assert block.ref_count == 0


def test_memory_manager_feature_validation_accepts_known() -> None:
    mm = MemoryManager(backend="vllm", features=["prefix_cache", "branch_cow"])
    assert "prefix_cache" in mm.enabled
    assert mm.has("branch_cow")
    assert not mm.has("tiered")


def test_memory_manager_feature_validation_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="未知的特性开关"):
        MemoryManager(features=["does_not_exist"])
