"""vLLM v1 引擎适配器（InprocClient 单进程，可观测 block_pool）。

与 ``engine_vllm.py``（v0）并列。v1 默认把引擎核心跑在子进程（``SyncMPClient``），
父进程看不到 ``block_pool``/显存。我们用 ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` 切到
``InprocClient``，引擎核心在主进程内，父进程可直接遍历到 scheduler.kv_cache_manager.block_pool。

遍历路径：
    llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
                ^LLMEngine  ^InprocClient  ^EngineCore   ^Scheduler

启用前需 ``source scripts/activate_env.sh``（设 LD_LIBRARY_PATH 让 _C 找到 libtorch.so）。
"""

from __future__ import annotations

import os
from typing import Any

from mimir.engine_vllm import VLLMEngine, _dtype_bytes


def _resolve_v1_block_pool(llm: Any) -> Any | None:
    """从 v1 LLM 遍历到 BlockPool（InprocClient 下），失败返回 None。"""
    try:
        eng = llm.llm_engine  # LLMEngine (v1)
        ec = eng.engine_core  # InprocClient（单进程）或 SyncMPClient（子进程）
        inner = getattr(ec, "engine_core", ec)  # InprocClient.engine_core = EngineCore
        sched = inner.scheduler
        return sched.kv_cache_manager.block_pool
    except Exception:
        return None


def _resolve_v1_scheduler(llm: Any) -> Any | None:
    try:
        return llm.llm_engine.engine_core.engine_core.scheduler
    except Exception:
        return None


def v1_kv_usage(llm: Any) -> dict[str, Any]:
    """查询 v1 BlockPool 的 KV 块使用（优化敏感指标）。"""
    out: dict[str, Any] = {
        "used_blocks": None,
        "total_blocks": None,
        "utilization": None,
        "used_gib": None,
    }
    bp = _resolve_v1_block_pool(llm)
    if bp is None:
        return out
    try:
        total = bp.num_gpu_blocks
        free = bp.get_num_free_blocks()
        used = total - free
        out["total_blocks"] = total
        out["used_blocks"] = used
        out["utilization"] = (
            bp.get_usage() if hasattr(bp, "get_usage") else (used / total if total else None)
        )
        # best-effort 字节估算
        try:
            cc = llm.llm_engine.vllm_config.cache_config
            mc = llm.llm_engine.vllm_config.model_config
            block_size = cc.block_size
            num_layers = mc.get_num_layers()
            num_kv_heads = mc.get_total_num_kv_heads()
            head_size = mc.get_head_size()
            dpb = _dtype_bytes(getattr(mc, "dtype", "bfloat16"))
            bytes_per_token = num_layers * 2 * num_kv_heads * head_size * dpb
            out["used_gib"] = (used * block_size * bytes_per_token) / (1024**3)
        except Exception:
            pass
    except Exception:
        pass
    return out


class VLLMEngineV1(VLLMEngine):
    """vLLM v1 引擎（InprocClient 单进程），接口与 v0 ``VLLMEngine`` 一致。"""

    def _init_engine(self) -> None:
        import time

        from vllm import LLM

        # 强制 v1 单进程（必须在 import vllm 后、构造前设好；activate_env.sh 已设默认）
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        c = self.config
        kwargs: dict[str, Any] = {
            "model": c.model,
            "dtype": c.dtype,
            "gpu_memory_utilization": c.gpu_memory_utilization,
            "enable_prefix_caching": c.enable_prefix_caching,
            "max_model_len": c.max_model_len,
            "enforce_eager": c.enforce_eager,
            "tensor_parallel_size": c.tensor_parallel_size,
            "seed": c.seed,
        }
        if c.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = c.kv_cache_dtype
        kwargs.update(c.extra)
        t0 = time.perf_counter()
        self._llm = LLM(**kwargs)
        self._engine_init_s = time.perf_counter() - t0

    def kv_usage(self) -> dict[str, Any]:
        if self._llm is None:
            return {
                "used_blocks": None,
                "total_blocks": None,
                "utilization": None,
                "used_gib": None,
            }
        return v1_kv_usage(self._llm)

    def mimir_scheduler(self) -> Any | None:
        """返回 v1 Scheduler（供 Mimir 调用 block_pool 的 mimir_* 方法 / 读 stats）。"""
        return _resolve_v1_scheduler(self._llm)

    def mimir_block_pool(self) -> Any | None:
        return _resolve_v1_block_pool(self._llm)

    def mimir_stats(self) -> dict[str, Any]:
        """读取 scheduler 上由 in-tree patch 暴露的 Mimir 统计（Phase B）。"""
        sched = self.mimir_scheduler()
        if sched is None:
            return {}
        getter = getattr(sched, "get_mimir_stats", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return {}
        return {}
