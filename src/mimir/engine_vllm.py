"""vLLM 引擎适配器。

把 Mimir 的内存优化层接到 vLLM 之上。本模块只负责「与 vLLM 交互 + 采集 KV/显存指标」，
不感知具体工作流（工作流→请求的构建在 ``benchmarks/harness.py``）。

延迟导入 vllm，使本模块可在未安装 vllm 的环境被导入（便于单测与无卡环境）。

指标说明
--------
vLLM 会按 ``gpu_memory_utilization`` **预分配**一大块 KV cache 池，因此
``torch.cuda.max_memory_reserved`` ≈ 池大小（固定，与优化无关），无法反映真实 KV 占用。
故本适配器额外提供 ``kv_usage()``：直接查询 vLLM block manager 的「已用/空闲块」，
这是对 KV 优化（压缩 / 外置 / CoW / 淘汰）**最敏感**的指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


@dataclass
class EngineConfig:
    """vLLM 引擎配置。"""

    model: str
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    max_model_len: int = 8192
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


def _dtype_bytes(dtype: Any) -> int:
    name = str(dtype).lower()
    if "bfloat16" in name or "bf16" in name or "float16" in name or "fp16" in name:
        return 2
    if "float32" in name or "fp32" in name:
        return 4
    if "float8" in name or "fp8" in name:
        return 1
    return 2


def vllm_kv_usage(llm: Any) -> dict[str, Any]:
    """查询 vLLM 当前 KV cache 块使用（优化敏感指标）。

    返回 ``used_blocks / total_blocks / utilization / used_gib``，任一不可得则为 None。
    版本敏感，全部 try/except 保护（不同 vLLM 版本内部 API 有差异）。
    """
    out: dict[str, Any] = {
        "used_blocks": None,
        "total_blocks": None,
        "utilization": None,
        "used_gib": None,
    }
    try:
        engine = llm.llm_engine
        sched = getattr(engine, "scheduler", None)
        if isinstance(sched, (list, tuple)):  # 多调度器情形
            sched = sched[0]
        bm = sched.block_manager
        free = bm.get_num_free_gpu_blocks()
        total = bm.get_num_total_gpu_blocks()
        used = total - free
        out["used_blocks"] = used
        out["total_blocks"] = total
        out["utilization"] = (used / total) if total else None
        # best-effort 字节估算
        try:
            cc = engine.cache_config
            mc = engine.model_config
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


class VLLMEngine:
    """vLLM 离线引擎的薄封装。"""

    def __init__(self, config: EngineConfig, *, device: int = 0) -> None:
        self.config = config
        self.device = device
        self._llm: Any = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._init_engine()
        return self._llm

    def _init_engine(self) -> None:
        from vllm import LLM  # 延迟导入

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
        kwargs.update(c.extra)
        self._llm = LLM(**kwargs)

    def kv_usage(self) -> dict[str, Any]:
        if self._llm is None:
            return {
                "used_blocks": None,
                "total_blocks": None,
                "utilization": None,
                "used_gib": None,
            }
        return vllm_kv_usage(self._llm)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, int]:
        """单轮 chat，返回 ``(text, output_token_count)``。"""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=self.config.seed)
        outs = self.llm.chat([messages], sp, use_tqdm=False)
        out = outs[0].outputs[0]
        n_tok = len(getattr(out, "token_ids", []) or [])
        return out.text, max(1, n_tok)
