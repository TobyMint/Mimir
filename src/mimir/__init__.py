"""Mimir — 面向智能体推理过程的内存管理系统。

Mimir 在 vLLM / llama.cpp 等推理框架之上，提供面向智能体长生命周期推理的内存管理优化，
包括 KV Cache 生命周期管理、分支推理内存共享（CoW）、上下文压缩、工具数据外置、
分层存储与异构硬件抽象。

典型用法::

    from mimir import MemoryManager

    mm = MemoryManager(
        backend="vllm",
        features=["prefix_cache", "branch_cow", "tool_offload"],
    )
"""

from __future__ import annotations

try:  # 从已安装的包元数据读取版本（开发模式下回退到硬编码）
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("mimir")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except ImportError:  # pragma: no cover - Python < 3.8
    __version__ = "0.1.0"

from mimir.core import (
    DeviceTier,
    KVBlock,
    Lifecycle,
    MemoryManager,
)

__all__ = [
    "DeviceTier",
    "KVBlock",
    "Lifecycle",
    "MemoryManager",
    "__version__",
]
