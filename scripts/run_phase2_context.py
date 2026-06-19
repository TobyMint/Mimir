"""Phase 2 evaluation：baseline vs context-compression Comparison。

在相同硬件/模型上跑同一工作流的两种版本，采集真实指标Comparison：
- baseline：原始工作流（不Compress）
- context_compress：用 ``ContextCompressor`` Compress后跑（BALANCED）

输出：
- benchmark_results/phase2_context_<model>.json
- benchmark_results/phase2_context_<model>.png  （KV/TTFT Comparison图 + Compress率）

用法（mimir 环境）：
    python scripts/run_phase2_context.py [--fidelity balanced|aggressive|lossless]
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

from mimir.context.compressor import ContextCompressor, Fidelity
from mimir.engine_vllm import EngineConfig, VLLMEngine
from mimir.gpu import as_env, pick_least_busy_gpu
from mimir.metrics import save_results
from mimir.plots import plot_kv_mem_comparison, plot_latency_comparison


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--fidelity", default="balanced", choices=[f.value for f in Fidelity])
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU：无单卡空闲 >=6GiB。请协调 GPU 后重试。")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index} ({g.name}), free {g.mem_free_gib:.1f}GiB", flush=True)

    tag = Path(args.model).name
    fidelity = Fidelity(args.fidelity)
    cfg = EngineConfig(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_util,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        use_v1=True,  # v1 + Phase R patch gives real TTFT (v0 RequestOutput.metrics is None)
    )
    eng = VLLMEngine(cfg, device=0)
    _ = eng.llm
    print(f"engine_init_seconds={eng.engine_init_seconds:.1f}", flush=True)

    results = []
    comp_stats = {}
    cases = all_workloads()
    for name, case in cases.items():
        print(f"\n=== {name} ===", flush=True)
        # baseline
        m_base = run_workload(eng, case, max_tokens=args.max_tokens, label=f"baseline_{name}")
        mem = m_base.peak_gpu_mem_alloc_gib
        cached = m_base.extra.get("total_cached_tokens")
        print(
            f"  baseline: TTFT={m_base.ttft_ms!s} E2E={m_base.e2e_latency_s!s} "
            f"mem={mem!s} cached={cached}",
            flush=True,
        )
        # compressed
        comp = ContextCompressor(fidelity=fidelity, keep_recent_turns=2)
        case_c = comp.compress(case)
        comp_stats[name] = comp.stats.__dict__
        cs = comp.stats
        print(
            f"  compression: {cs.compressed_chars}/{cs.original_chars} chars "
            f"(-{cs.char_reduction_pct:.1f}%), {cs.tool_results_summarized} summarized",
            flush=True,
        )
        m_opt = run_workload(eng, case_c, max_tokens=args.max_tokens, label=f"compress_{name}")
        mem2 = m_opt.peak_gpu_mem_alloc_gib
        cached2 = m_opt.extra.get("total_cached_tokens")
        print(
            f"  compressed: TTFT={m_opt.ttft_ms!s} E2E={m_opt.e2e_latency_s!s} "
            f"mem={mem2!s} cached={cached2}",
            flush=True,
        )
        # Comparison
        if m_base.ttft_ms and m_opt.ttft_ms:
            print(
                f"  -> TTFT {m_base.ttft_ms:.1f} -> {m_opt.ttft_ms:.1f} ms "
                f"({(1 - m_opt.ttft_ms / m_base.ttft_ms) * 100:+.1f}%)",
                flush=True,
            )
        if m_base.e2e_latency_s and m_opt.e2e_latency_s:
            print(
                f"  -> E2E  {m_base.e2e_latency_s:.2f} -> {m_opt.e2e_latency_s:.2f} s "
                f"({(1 - m_opt.e2e_latency_s / m_base.e2e_latency_s) * 100:+.1f}%)",
                flush=True,
            )
        results.extend([m_base, m_opt])

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"phase2_context_{tag}.json"
    save_results(results, json_path)
    (out_dir / f"phase2_context_{tag}_compstats.json").write_text(
        json.dumps(comp_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_kv_mem_comparison(
        results,
        out_dir / f"phase2_context_{tag}_mem.png",
        title=f"Phase 2 context-compress: Peak KV Memory ({fidelity.value})",
    )
    plot_latency_comparison(
        results,
        out_dir / f"phase2_context_{tag}_lat.png",
        title=f"Phase 2 context-compress: Latency ({fidelity.value})",
    )
    print(f"\n保存: {json_path}")
    print(f"保存: {out_dir / f'phase2_context_{tag}_mem.png'}")
    print(f"保存: {out_dir / f'phase2_context_{tag}_lat.png'}")
    print("PHASE2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
