# ruff: noqa: E501, E701, E702
"""Block-class 演示：在真实 agent 压力下展示类别标签 + 类别感知淘汰生效。

不与 fcfs 比 new_prefill（fcfs 无类别淘汰，路径不同，直接比会误导），而是：
1. 在一条构造压力的长 prompt 上跑 Mimir 引擎，给 KV 块打语义类别标签；
2. 主动调 mimir_class_aware_evict 触发类别感知淘汰；
3. 导出 mimir_class_stats，证明标签分布合理（system/reasoning/tool_result 各有）+
   淘汰按 reasoning > user > tool_result > system 优先级发生。

类别感知的正确性由 tests/test_block_class.py（5 个确定性单测）保证。
本脚本补充「在真实 vLLM v1 引擎 + 真实 Qwen3-4B 上标签注入成功」的端到端证据。
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


def main() -> int:
    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU"); return 2
    os.environ.update(as_env(g))
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    model = "/data/models/Qwen3-4B-Instruct-2507"
    # 用 fcfs 策略：mimir 策略有 Phase L 自驱动回收（任务结束自动回收 KV），会清掉一次性
    # chat 后的块标签。fcfs 不自动回收，块标签在 chat 后保留，便于演示类别分布 + 主动淘汰。
    # block-class 打标签（cache_full_blocks 注入）与策略无关，对 fcfs/mimir 都生效。
    eng = VLLMEngineV1(EngineConfig(
        model=model, dtype="bfloat16", gpu_memory_utilization=0.90,
        enable_prefix_caching=True, max_model_len=8192,
        extra={"scheduling_policy": "fcfs"}), device=0)
    _ = eng.llm
    bp = eng.mimir_block_pool()

    system = "You are a meticulous research agent. Always cite sources. " * 30
    # 用「唯一」内容造 reasoning / tool_result，确保每个块 hash 不同、都被 cache_full_blocks
    # 新缓存（从而走 Mimir block-class 打标签路径）；重复内容会被 vLLM APC 去重复用、绕过打标签。
    reasoning = " ".join(f"reasoning_step_{i} considers memory_bandwidth_tier_{i}." for i in range(120))
    tool_result = "[TOOL_RESULT search]\n" + " ".join(
        f"fact_{i}: key_kv_cache_detail_{i}." for i in range(120)
    )

    msgs = [
        {"role": "system", "content": system},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": tool_result},
        {"role": "user", "content": "Summarize the key facts."},
    ]
    eng.set_current_task("blockclass_demo")
    ro = eng.chat_full(msgs, max_tokens=16, temperature=0.0)
    rm = _req_metrics(ro)

    tagged = bp.mimir_class_stats()
    print(f"\nprefill: prompt={rm.get('num_prompt_tokens')} cached={rm.get('num_cached_tokens')}", flush=True)
    print(f"block-class 分布: {tagged['block_class_counts']}", flush=True)

    # 主动触发类别感知淘汰（淘汰 reasoning 块），证明策略生效
    pre_reasoning = tagged["block_class_counts"].get("reasoning", 0)
    evicted = bp.mimir_class_aware_evict(max(1, pre_reasoning // 2))
    after = bp.mimir_class_stats()
    print(f"mimir_class_aware_evict({max(1, pre_reasoning//2)}): 实际淘汰 {evicted} 块", flush=True)
    print(f"淘汰后分布: {after['block_class_counts']}", flush=True)
    print(f"按类别淘汰数: {after['class_evicts']}", flush=True)

    summary = {
        "model": Path(model).name,
        "prefill_prompt_tokens": rm.get("num_prompt_tokens"),
        "prefill_cached_tokens": rm.get("num_cached_tokens"),
        "block_class_counts_before_evict": tagged["block_class_counts"],
        "evict_request": max(1, pre_reasoning // 2),
        "evicted": evicted,
        "block_class_counts_after_evict": after["block_class_counts"],
        "class_evicts": after["class_evicts"],
    }
    out = Path("benchmark_results")
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"phase_blockclass_{Path(model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {jp}")
    print("PHASE_BLOCKCLASS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
