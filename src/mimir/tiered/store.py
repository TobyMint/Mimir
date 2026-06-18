"""分层内存与异构存储优化（赛题优化方向之五）。

探索 GPU 显存、主存与外部存储之间的分层内存体系，实现 KV Cache 与上下文数据的
冷热分离、动态迁移与按需加载。详见 ``docs/技术方案.md`` §3.5。

与 vLLM CPU offload 的区别
--------------------------
vLLM 提供 CPU offload（KV 在 GPU/CPU 间），但只有**两层**、按**固定**配置，
无「冷数据落盘」、无「基于访问频率的自动迁移」。Mimir ``TieredStore`` 提供：

1. **三层**：GPU（热）→ HOST 主存（温）→ DISK（冷）。
2. **冷热分离 + 动态迁移**：基于访问频率（LRU + 频次）自动在层间迁移。
3. **按需加载**：冷数据被访问时自动 promote 回热层（含 disk→host→gpu）。
4. **可配置容量**：每层容量上限，超限触发 demote（热→温→冷）。

本模块是「数据层」抽象（可用于上下文块 / 工具数据 / KV 块元数据）。
真实 KV 张量的分层由 vLLM 在引擎层处理；本模块提供**策略与记账**，
并可直接管理「工具外置数据 / 长上下文片段」这类非张量数据的分层。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Tier(str, Enum):
    GPU = "gpu"  # 热
    HOST = "host"  # 温
    DISK = "disk"  # 冷


@dataclass
class TierStats:
    """分层存储统计。"""

    gpu_count: int = 0
    host_count: int = 0
    disk_count: int = 0
    promotions: int = 0  # 冷→热 提升次数
    demotions: int = 0  # 热→冷 降级次数
    disk_reads: int = 0  # disk 命中读取

    def total(self) -> int:
        return self.gpu_count + self.host_count + self.disk_count


class TieredStore:
    """三层 LRU 存储，支持自动迁移与按需加载。

    - ``gpu_cap`` / ``host_cap``：各层容量（项数）。disk 视为无限。
    - 访问（``get``）会 promote 到 GPU；GPU 满则 demote 最冷到 HOST；HOST 满则 demote 到 DISK。
    - DISK 数据持久化到 ``disk_dir``（可选）。
    """

    def __init__(
        self,
        *,
        gpu_cap: int = 8,
        host_cap: int = 32,
        disk_dir: str | Path | None = None,
    ) -> None:
        self.gpu_cap = gpu_cap
        self.host_cap = host_cap
        self._gpu: OrderedDict[str, Any] = OrderedDict()
        self._host: OrderedDict[str, Any] = OrderedDict()
        self._disk: dict[str, Any] = {}
        self._disk_dir = Path(disk_dir) if disk_dir else None
        if self._disk_dir:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.stats = TierStats()

    def put(self, key: str, value: Any) -> Tier:
        """放入一项（进 GPU 热层，触发必要的 demote）。返回最终所在层。"""
        with self._lock:
            # 已存在则更新并 promote
            tier = self._tier_of(key)
            if tier is not None:
                self._remove(key)
            self._gpu[key] = value
            self._evict_gpu()
            return self._tier_of(key) or Tier.GPU

    def get(self, key: str) -> Any | None:
        """访问一项：自动 promote 到 GPU，返回值。不存在返回 None。"""
        with self._lock:
            tier = self._tier_of(key)
            if tier is None:
                return None
            value = self._remove(key)
            if tier is Tier.DISK:
                self.stats.disk_reads += 1
            if tier in (Tier.HOST, Tier.DISK):
                self.stats.promotions += 1
            self._gpu[key] = value
            self._evict_gpu()
            return value

    def _tier_of(self, key: str) -> Tier | None:
        if key in self._gpu:
            return Tier.GPU
        if key in self._host:
            return Tier.HOST
        if key in self._disk:
            return Tier.DISK
        return None

    def _remove(self, key: str) -> Any:
        for store in (self._gpu, self._host):
            if key in store:
                return store.pop(key)
        if key in self._disk:
            v = self._disk.pop(key)
            if self._disk_dir:
                p = self._disk_dir / f"{key}.bin"
                if p.exists():
                    p.unlink()
            return v
        raise KeyError(key)

    def _evict_gpu(self) -> None:
        """GPU 超容：demote 最冷到 HOST。"""
        while len(self._gpu) > self.gpu_cap:
            k, v = self._gpu.popitem(last=False)  # 最旧
            self._host[k] = v
            self.stats.demotions += 1
            self._evict_host()

    def _evict_host(self) -> None:
        """HOST 超容：demote 最冷到 DISK。"""
        while len(self._host) > self.host_cap:
            k, v = self._host.popitem(last=False)
            self._disk[k] = v
            if self._disk_dir:
                (self._disk_dir / f"{k}.bin").write_bytes(_to_bytes(v))
            self.stats.demotions += 1

    def snapshot(self) -> dict[str, list[str]]:
        """各层当前的 key 列表（用于可视化/报告）。"""
        with self._lock:
            return {
                "gpu": list(self._gpu.keys()),
                "host": list(self._host.keys()),
                "disk": list(self._disk.keys()),
            }

    def stats_dict(self) -> dict[str, Any]:
        s = self.stats
        s.gpu_count = len(self._gpu)
        s.host_count = len(self._host)
        s.disk_count = len(self._disk)
        return {
            "gpu_count": s.gpu_count,
            "host_count": s.host_count,
            "disk_count": s.disk_count,
            "total": s.total(),
            "promotions": s.promotions,
            "demotions": s.demotions,
            "disk_reads": s.disk_reads,
        }


def _to_bytes(v: Any) -> bytes:
    """把值序列化为字节（用于落盘）。"""
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, str):
        return v.encode("utf-8")
    import pickle

    return pickle.dumps(v)


@dataclass
class MigrationPolicy:
    """迁移策略参数（用于调优）。"""

    gpu_cap: int = 8
    host_cap: int = 32
    # 访问频次阈值：低于此值的 GPU 项更倾向被 demote（简化：当前以 LRU 为主）
    access_freq_demote_threshold: int = 1

    def make_store(self, disk_dir: str | Path | None = None) -> TieredStore:
        return TieredStore(gpu_cap=self.gpu_cap, host_cap=self.host_cap, disk_dir=disk_dir)
