# ruff: noqa: E501, E701, E702
"""诚实指标：tool-call probe 在 KV 压力下的「存活率」——native(LRU) vs Mimir(class-aware)。

为什么这个指标诚实：LongBench 均分掩盖 agent 真实需求（工具参数召回）。这里构造 agent
场景——长上下文含一个「工具调用 probe」（特定 key）+ 大量 reasoning 干扰。压力下：
- native(LRU)：盲按访问序淘汰，probe 的 tool_result 块可能被 reasoning 挤掉 → 第二次用时失忆
- Mimir(class-aware)：优先淘汰 reasoning，保留 tool_result → probe 块存活 → 第二次命中

指标：第二次（含 probe 前缀）请求的 num_cached_tokens —— 越大说明 probe/tool_result 块越存活。
单引擎两路：同 fcfs 引擎（无自驱动回收，块保留可控）；class-aware 路在第二次前手动
mimir_class_aware_evict 优先淘汰 reasoning。两路用不同 probe key 避免前缀串扰。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.harness import _req_metrics

from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from mimir.gpu import as_env, pick_least_busy_gpu


def _probe_tool_result(key: str) -> str:
    return (
        f"[TOOL_RESULT search]\nThe stored credential is key={key}. "
        + " ".join(f"distinct_fact_{key}_{i} value_{i}." for i in range(60))
    )


def _reasoning_noise(tag: str) -> str:
    return " ".join(
        f"reasoning_{tag}_step_{i} consider bandwidth latency tradeoff tier_{i}." for i in range(160)
    )


def _run_path(eng, *, key: str, tag: str, class_aware_evict: bool, mtok: int) -> dict:
    """一路：第一次撑满 + （可选）类别淘汰 + 第二次复用 probe 前缀。"""
    bp = eng.mimir_block_pool()
    first = [
        {"role": "system", "content": "You are a research agent. Always cite sources. " * 12},
        {"role": "user", "content": _probe_tool_result(key)},
        {"role": "assistant", "content": _reasoning_noise(tag)},
        {"role": "user", "content": "Acknowledge the stored key."},
    ]
    eng.set_current_task(f"{tag}_t")
    ro1 = eng.chat_full(first, max_tokens=mtok, temperature=0.0)
    rm1 = _req_metrics(ro1)
    cls1 = bp.mimir_class_stats()
    evicted = 0
    if class_aware_evict:
        pre_reasoning = cls1["block_class_counts"].get("reasoning", 0)
        evicted = bp.mimir_class_aware_evict(pre_reasoning)
    second = [
        {"role": "system", "content": "You are a research agent. Always cite sources. " * 12},
        {"role": "user", "content": _probe_tool_result(key)},
        {"role": "user", "content": "What was the stored key? Call the tool again with it if needed."},
    ]
    ro2 = eng.chat_full(second, max_tokens=mtok, temperature=0.0)
    rm2 = _req_metrics(ro2)
    cls2 = bp.mimir_class_stats()
    return {
        "class_aware_evict": class_aware_evict,
        "key": key,
        "first_prompt_tokens": rm1.get("num_prompt_tokens", 0),
        "second_prompt_tokens": rm2.get("num_prompt_tokens", 0),
        "second_cached_tokens": rm2.get("num_cached_tokens", 0) or 0,
        "second_new_prefill": rm2.get("num_prompt_tokens", 0) - (rm2.get("num_cached_tokens", 0) or 0),
        "second_ttft_ms": rm2.get("ttft_ms"),
        "manual_evicted_reasoning": evicted,
        "class_counts_after_first": cls1.get("block_class_counts", {}),
        "class_evicts": cls2.get("class_evicts", {}),
    }


def main() -> int:
    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU"); return 2
    os.environ.update(as_env(g))
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    model = "/data/models/Qwen3-4B-Instruct-2507"
    # 单 fcfs 引擎跑两路（不同 probe key 避免前缀串扰）——干净对比 LRU vs class-aware
    eng = VLLMEngineV1(EngineConfig(
        model=model, dtype="bfloat16", gpu_memory_utilization=0.90,
        enable_prefix_caching=True, max_model_len=8192,
        extra={"scheduling_policy": "fcfs"}), device=0)
    _ = eng.llm
    mtok = 16

    lru = _run_path(eng, key="ALPHA_7391", tag="lru", class_aware_evict=False, mtok=mtok)
    print("\n=== native(LRU) ===", flush=True)
    print(f"  first prompt={lru['first_prompt_tokens']} second prompt={lru['second_prompt_tokens']}", flush=True)
    print(f"  second cached(probe 存活)={lru['second_cached_tokens']} "
          f"new_prefill={lru['second_new_prefill']} ttft={lru['second_ttft_ms']!s}ms", flush=True)

    cls = _run_path(eng, key="BETA_2048", tag="cls", class_aware_evict=True, mtok=mtok)
    print("\n=== Mimir(class-aware) ===", flush=True)
    print(f"  first prompt={cls['first_prompt_tokens']} second prompt={cls['second_prompt_tokens']}", flush=True)
    print(f"  主动淘汰 reasoning={cls['manual_evicted_reasoning']} 块", flush=True)
    print(f"  second cached(probe 存活)={cls['second_cached_tokens']} "
          f"new_prefill={cls['second_new_prefill']} ttft={cls['second_ttft_ms']!s}ms", flush=True)

    n, m = lru["second_cached_tokens"], cls["second_cached_tokens"]
    summary = {
        "model": Path(model).name,
        "metric": ("second_cached_tokens: tool-call probe 所在 tool_result 块在 KV 压力淘汰后"
                   "的存活 token 数（越大=probe 越没失忆）"),
        "native_lru_cached": n,
        "mimir_class_aware_cached": m,
        "mimir_advantage_tokens": m - n,
        "lru": lru, "class_aware": cls,
    }
    out = Path("benchmark_results"); out.mkdir(parents=True, exist_ok=True)
    jp = out / f"recall_metric_{Path(model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nprobe 二次命中 token: native(LRU)={n} -> Mimir(class-aware)={m} (优势 +{m-n} tokens)")
    print(f"JSON: {jp}")
    print("RECALL_METRIC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
