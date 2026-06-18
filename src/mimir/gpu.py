"""GPU 选择工具：选最空闲的单卡，设 ``CUDA_VISIBLE_DEVICES``。

依据：``nvidia-smi --query-gpu`` 的 ``memory.free``（最大空闲显存）与
``utilization.gpu``（最低利用率）综合排序。多卡互联差，按 ADR-002 只选单卡。

被占满时返回 None，调用方据此回退到非 GPU 轻量正确性验证（ADR-003）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class GPUInfo:
    index: int
    name: str
    mem_total_mib: int
    mem_used_mib: int
    mem_free_mib: int
    gpu_util: int  # 百分比 0-100

    @property
    def mem_free_gib(self) -> float:
        return self.mem_free_mib / 1024.0


def _nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query_gpus() -> list[GPUInfo]:
    """查询所有 GPU。无 nvidia-smi 或失败时返回空列表。"""
    if not _nvidia_smi_available():
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=15
        ).stdout.strip()
    except Exception:
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    mem_total_mib=int(parts[2]),
                    mem_used_mib=int(parts[3]),
                    mem_free_mib=int(parts[4]),
                    gpu_util=int(parts[5]),
                )
            )
        except ValueError:
            continue
    return gpus


def pick_least_busy_gpu(
    *,
    min_free_gib: float = 2.0,
    prefer_index: int | None = None,
) -> GPUInfo | None:
    """选最空闲的单卡。

    - 优先返回 ``prefer_index`` 指定的卡（若空闲足够）。
    - 否则按 (空闲显存多, 利用率低) 排序选最优。
    - 任何卡空闲显存都低于 ``min_free_gib`` 时返回 None（建议回退非 GPU 工作）。
    """
    gpus = query_gpus()
    if not gpus:
        return None
    if prefer_index is not None:
        for g in gpus:
            if g.index == prefer_index and g.mem_free_gib >= min_free_gib:
                return g
    viable = [g for g in gpus if g.mem_free_gib >= min_free_gib]
    if not viable:
        return None
    # 排序：空闲显存越多越好，利用率越低越好
    viable.sort(key=lambda g: (-g.mem_free_gib, g.gpu_util))
    return viable[0]


def as_env(gpu: GPUInfo | None) -> dict[str, str]:
    """把选中的 GPU 转为可 ``os.environ`` 注入的环境变量。"""
    if gpu is None:
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(gpu.index)}


def select_and_describe(*, min_free_gib: float = 2.0, prefer_index: int | None = None) -> str:
    """选卡并返回人读描述（用于日志/PROGRESS）。不修改环境。"""
    g = pick_least_busy_gpu(min_free_gib=min_free_gib, prefer_index=prefer_index)
    if g is None:
        all_g = query_gpus()
        state = ", ".join(
            f"GPU{i}:{gm.mem_free_gib:.1f}GiB free/{gm.gpu_util}% util"
            for i, gm in enumerate(all_g)
        )
        return f"无可选 GPU（均 < {min_free_gib}GiB 空闲）。当前: {state}"
    env = as_env(g)
    return (
        f"选中 GPU {g.index} ({g.name})：空闲 {g.mem_free_gib:.1f}GiB / "
        f"{g.mem_total_mib / 1024:.1f}GiB，利用率 {g.gpu_util}%。"
        f"建议 export {list(env.items())[0][0]}={list(env.items())[0][1]}"
    )


def snapshot_env() -> dict[str, object]:
    """记录当前 GPU 状态快照（用于结果落盘，保证可复现性）。"""
    return {
        "gpus": [g.__dict__ for g in query_gpus()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def to_json(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2)
