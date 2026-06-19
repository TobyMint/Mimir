"""设备抽象层（赛题优化方向之六：异构 AI 加速硬件支持）。

统一 ``DeviceBackend`` 抽象，屏蔽 CUDA / ROCm / CPU 差异，并预留国产 NPU
（昇腾 DTK/CANN、寒武纪等）探测钩子。提供：
1. 运行时后端探测（torch.cuda / torch.version.hip / 环境变量）+ 降级链。
2. 资源报告（显存 / 利用率 / 设备名）跨后端统一口径。
3. 降级决策：无可用加速硬件时降级到 CPU / offload，保证可跑。

本模块纯 Python、不依赖 GPU，便于无卡环境单测与 CI。
与 ``mimir.gpu``（nvidia-smi 查询，CUDA 假定）的区别：本模块做后端抽象与降级，
``gpu.py`` 做单卡选择。两者可组合：先 detect_backend()，再按后端选卡/降级。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BackendKind(str, Enum):
    CUDA = "cuda"  # NVIDIA CUDA
    ROCM = "rocm"  # AMD ROCm
    CPU = "cpu"  # 无加速硬件，CPU 降级
    ASCEND = "ascend"  # 昇腾 NPU（DTK/CANN）
    CAMBRICON = "cambricon"  # 寒武纪
    UNKNOWN = "unknown"


@dataclass
class DeviceBackend:
    """一个被探测到的后端。"""

    kind: BackendKind
    available: bool  # 是否可用（torch 能否初始化）
    device_count: int = 0
    name: str = ""
    capability: tuple[int, int] | None = None  # (major, minor)，CUDA 算力
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def supports_fp8(self) -> bool:
        """fp8 KV 需要 Hopper(sm90)+ / Blackwell(sm100)，或等价 NPU。"""
        if self.kind is BackendKind.CUDA and self.capability is not None:
            return self.capability[0] >= 9  # sm90+
        return False

    def kv_cache_dtype_options(self) -> list[str]:
        """该后端支持的 KV cache dtype 选项（auto=bf16 总有）。"""
        opts = ["auto"]
        if self.supports_fp8():
            opts += ["fp8", "fp8_e4m3"]
        return opts


def _detect_cuda() -> DeviceBackend:
    try:
        import torch

        if not torch.cuda.is_available():
            return DeviceBackend(
                BackendKind.CUDA, available=False, notes="torch.cuda not available"
            )
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if n else ""
        cap = None
        if n:
            c = torch.cuda.get_device_capability(0)
            cap = (int(c[0]), int(c[1]))
        return DeviceBackend(
            BackendKind.CUDA, available=True, device_count=n, name=name, capability=cap
        )
    except Exception as e:  # noqa: BLE001
        return DeviceBackend(BackendKind.CUDA, available=False, notes=f"detect error: {e}")


def _detect_rocm() -> DeviceBackend:
    try:
        import torch

        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if not is_rocm:
            return DeviceBackend(BackendKind.ROCM, available=False, notes="not a ROCm torch build")
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0  # ROCm 用 cuda 接口
        name = torch.cuda.get_device_name(0) if n else ""
        return DeviceBackend(
            BackendKind.ROCM,
            available=bool(n),
            device_count=n,
            name=name,
            notes=f"ROCm {torch.version.hip}",
        )
    except Exception as e:  # noqa: BLE001
        return DeviceBackend(BackendKind.ROCM, available=False, notes=f"detect error: {e}")


def _detect_ascend() -> DeviceBackend:
    """昇腾 NPU（DTK/CANN）探测：torch_npu 或环境变量。"""
    try:
        import torch_npu  # type: ignore  # noqa: F401

        n = torch_npu.npu.device_count() if hasattr(torch_npu, "npu") else 0  # type: ignore[attr-defined]
        return DeviceBackend(
            BackendKind.ASCEND, available=bool(n), device_count=n, notes="torch_npu detected"
        )
    except Exception:
        pass
    # 环境变量回退
    import os

    if os.environ.get("ASCEND_HOME_PATH") or os.environ.get("CANN_HOME"):
        return DeviceBackend(
            BackendKind.ASCEND,
            available=False,
            notes="CANN/DTK env present but torch_npu not importable",
        )
    return DeviceBackend(BackendKind.ASCEND, available=False, notes="no ascend env")


def _detect_cambricon() -> DeviceBackend:
    try:
        import torch_mlu  # type: ignore  # noqa: F401

        return DeviceBackend(BackendKind.CAMBRICON, available=True, notes="torch_mlu detected")
    except Exception:
        return DeviceBackend(BackendKind.CAMBRICON, available=False, notes="no cambricon")


def detect_all_backends() -> list[DeviceBackend]:
    """探测所有后端，返回列表（按 CUDA/ROCm/Ascend/Cambricon 顺序）。"""
    return [
        _detect_cuda(),
        _detect_rocm(),
        _detect_ascend(),
        _detect_cambricon(),
    ]


def pick_backend() -> DeviceBackend:
    """选一个可用后端（降级链：CUDA -> ROCm -> Ascend -> Cambricon -> CPU）。"""
    for b in detect_all_backends():
        if b.available:
            return b
    return DeviceBackend(
        BackendKind.CPU, available=True, device_count=0, notes="no accelerator; CPU fallback"
    )


def hardware_report() -> dict[str, Any]:
    """硬件资源报告（跨后端统一口径）。"""
    backs = detect_all_backends()
    chosen = pick_backend()
    return {
        "backends": [
            {
                "kind": b.kind.value,
                "available": b.available,
                "device_count": b.device_count,
                "name": b.name,
                "capability": list(b.capability) if b.capability else None,
                "supports_fp8": b.supports_fp8(),
                "kv_dtype_options": b.kv_cache_dtype_options(),
                "notes": b.notes,
            }
            for b in backs
        ],
        "chosen_backend": chosen.kind.value,
        "fp8_kv_supported": chosen.supports_fp8(),
    }


def recommend_engine_config(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """根据探测后端推荐引擎配置（dtype/kv_cache_dtype/enforce_eager 等）。"""
    cfg = dict(base or {})
    b = pick_backend()
    if b.kind is BackendKind.CPU or not b.available:
        cfg.setdefault("dtype", "float32")
        cfg.setdefault("kv_cache_dtype", "auto")
        cfg.setdefault("enforce_eager", True)
        cfg["_mimir_note"] = "CPU fallback: float32, eager, no fp8"
    elif b.kind is BackendKind.CUDA:
        cfg.setdefault("dtype", "bfloat16")
        # fp8 仅在 sm90+ 推荐（3090 会触发 Phase F 优雅降级到 bf16）
        if b.supports_fp8():
            cfg.setdefault("kv_cache_dtype", "fp8")
        else:
            cfg.setdefault("kv_cache_dtype", "auto")
            cfg["_mimir_note"] = "CUDA but <sm90: fp8 will gracefully fall back to bf16 (Phase F)"
    else:
        cfg.setdefault("dtype", "bfloat16")
        cfg.setdefault("kv_cache_dtype", "auto")
    cfg["_backend"] = b.kind.value
    return cfg
