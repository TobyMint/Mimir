"""各优化方向子模块应可被导入（冒烟测试）。"""

from __future__ import annotations

import importlib

import pytest

#: 赛题六个优化方向对应的子模块
OPTIMIZATION_SUBMODULES = [
    "mimir.kv_cache",
    "mimir.branch",
    "mimir.context",
    "mimir.tools",
    "mimir.tiered",
    "mimir.hardware",
]


@pytest.mark.parametrize("name", OPTIMIZATION_SUBMODULES)
def test_submodule_importable(name: str) -> None:
    """每个优化方向子模块都能被正常导入，且带有模块文档字符串。"""
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} 缺少模块文档字符串"
