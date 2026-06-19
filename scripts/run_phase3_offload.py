"""Phase 3 evaluation：baseline vs tool-data-offload Comparison。

把工具调用的大返回Offload（``ToolDataStore``），上下文只放引用+摘要，避免大块进入 KV。
在相同硬件/模型上Comparison baseline（全量进上下文）vs offload。

输出：
- benchmark_results/phase3_offload_<model>.json
- benchmark_results/phase3_offload_<model>_{mem,lat}.png

用法（mimir 环境）：
    python scripts/run_phase3_offload.py [--max-tokens 96]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.harness import run_workload
from benchmarks.workloads import all_workloads

from mimir.engine_vllm import EngineConfig, VLLMEngine
from mimir.gpu import as_env, pick_least_busy_gpu
from mimir.metrics import save_results
from mimir.plots import plot_kv_mem_comparison, plot_latency_comparison
from mimir.tools.offload import ToolDataStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU：无单卡空闲 >=6GiB。请协调 GPU 后重试。")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index} ({g.name}), free {g.mem_free_gib:.1f}GiB", flush=True)

    tag = Path(args.model).name
    cfg = EngineConfig(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_util,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        use_v1=False,
    )
    eng = VLLMEngine(cfg, device=0)
    _ = eng.llm
    print(f"engine_init_seconds={eng.engine_init_seconds:.1f}", flush=True)

    results = []
    offload_stats = {}
    cases = all_workloads()
    for name, case in cases.items():
        print(f"\n=== {name} ===", flush=True)
        # baseline
        m_base = run_workload(eng, case, max_tokens=args.max_tokens, label=f"baseline_{name}")
        mem = m_base.peak_gpu_mem_alloc_gib
        cached = m_base.extra.get("total_cached_tokens")
        print(
            f"  baseline: TTFT={m_base.ttft_ms:.1f}ms new_prefill="
            f"{m_base.extra.get('total_prefill_new_tokens')} mem={mem:.2f}GiB cached={cached}",
            flush=True,
        )
        # offload
        store = ToolDataStore()
        m_opt = run_workload(
            eng, case, max_tokens=args.max_tokens, label=f"offload_{name}", offload_store=store
        )
        offload_stats[name] = store.stats()
        mem2 = m_opt.peak_gpu_mem_alloc_gib
        cached2 = m_opt.extra.get("total_cached_tokens")
        print(
            f"  offload:  TTFT={m_opt.ttft_ms:.1f}ms new_prefill="
            f"{m_opt.extra.get('total_prefill_new_tokens')} mem={mem2:.2f}GiB cached={cached2}",
            flush=True,
        )
        st = store.stats()
        print(
            f"  offload_stats: offloaded={st['offloaded_count']} chars={st['offloaded_chars']} "
            f"inline={st['inline_count']}",
            flush=True,
        )
        if m_base.ttft_ms and m_opt.ttft_ms:
            d = (1 - m_opt.ttft_ms / m_base.ttft_ms) * 100
            print(
                f"  -> TTFT {m_base.ttft_ms:.1f} -> {m_opt.ttft_ms:.1f} ms ({d:+.1f}%)", flush=True
            )
        np_b = m_base.extra.get("total_prefill_new_tokens")
        np_o = m_opt.extra.get("total_prefill_new_tokens")
        if np_b and np_o:
            print(
                f"  -> new_prefill {np_b} -> {np_o} ({(1 - np_o / np_b) * 100:+.1f}%)", flush=True
            )
        results.extend([m_base, m_opt])

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"phase3_offload_{tag}.json"
    save_results(results, json_path)
    (out_dir / f"phase3_offload_{tag}_stats.json").write_text(
        json.dumps(offload_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_kv_mem_comparison(
        results,
        out_dir / f"phase3_offload_{tag}_mem.png",
        title="Phase 3 工具数据Offload：PeakMemory",
    )
    plot_latency_comparison(
        results, out_dir / f"phase3_offload_{tag}_lat.png", title="Phase 3 工具数据Offload：Latency"
    )
    print(f"\n保存: {json_path}")
    print(f"保存: {out_dir / f'phase3_offload_{tag}_mem.png'}")
    print("PHASE3_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
