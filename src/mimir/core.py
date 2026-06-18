"""Mimir 核心编排层。

定义统一的块抽象（KVBlock）、生命周期与设备分层枚举，以及面向 Agent 工作流的
``MemoryManager``。各优化方向作为独立子模块，由 ``MemoryManager`` 经特性开关编排启用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Lifecycle(str, Enum):
    """KV 块的生命周期状态。"""

    ACTIVE = "active"  # 当前推理正在使用，不可淘汰
    EVICTABLE = "evictable"  # 可被淘汰/迁移
    PINNED = "pinned"  # 被显式钉住（如常驻 system prompt）


class DeviceTier(str, Enum):
    """分层存储层级（GPU 显存 → 主存 → 外部存储）。"""

    GPU = "gpu"  # 热数据
    HOST = "host"  # 温数据
    DISK = "disk"  # 冷数据


@dataclass
class KVBlock:
    """统一的 KV Cache 逻辑块抽象。

    借鉴 PagedAttention 的 block 思想，将 KV 划分为带元数据的逻辑块，以支持
    复用（前缀匹配）、共享/写时复制（分支）、淘汰（生命周期管理）与迁移（分层存储）。
    """

    block_id: int
    seq_id: str
    token_range: tuple[int, int]
    ref_count: int = 1
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    device_tier: DeviceTier = DeviceTier.GPU
    hash: bytes | None = None  # 内容指纹，用于前缀复用匹配
    metadata: dict[str, Any] = field(default_factory=dict)

    def acquire(self) -> KVBlock:
        """增加引用计数（共享 / CoW 场景）。"""
        self.ref_count += 1
        return self

    def release(self) -> bool:
        """减少引用计数，返回是否已无引用（可回收）。"""
        if self.ref_count > 0:
            self.ref_count -= 1
        return self.ref_count == 0


class MemoryManager:
    """面向智能体工作流的统一内存管理入口。

    通过 ``features`` 开关按需启用各优化方向，便于消融实验与渐进式集成::

        mm = MemoryManager(backend="vllm", features=["prefix_cache", "branch_cow"])

    注意：本类当前为脚手架，具体各特性的实现随子模块推进填充。
    """

    #: 全部受支持的特性开关
    SUPPORTED_FEATURES = frozenset(
        {
            "prefix_cache",  # KV 前缀复用
            "lifecycle",  # 生命周期感知淘汰
            "branch_cow",  # 分支写时复制共享
            "context_compress",
            "tool_offload",  # 工具中间数据外置
            "tiered",  # 分层存储与冷热迁移
        }
    )

    def __init__(
        self,
        backend: str = "vllm",
        features: list[str] | None = None,
        device: str = "cuda",
    ) -> None:
        self.backend = backend
        self.device = device
        self.features: set[str] = set(features or [])
        unknown = self.features - self.SUPPORTED_FEATURES
        if unknown:
            raise ValueError(f"未知的特性开关: {sorted(unknown)}")
        # 子模块管理器（随实现推进实例化）
        self._kv_cache: Any = None
        self._branch: Any = None
        self._context: Any = None
        self._tools: Any = None
        self._tiered: Any = None

    @property
    def enabled(self) -> set[str]:
        """当前启用的特性集合。"""
        return set(self.features)

    def has(self, feature: str) -> bool:
        return feature in self.features

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试
        return (
            f"MemoryManager(backend={self.backend!r}, "
            f"device={self.device!r}, features={sorted(self.features)})"
        )
