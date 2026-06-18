"""Phase 7a：KV Cache fp8 量化（新优化方向）。

vLLM 支持 ``kv_cache_dtype=\"fp8\"``：把 KV cache 从 bf16(2B) 量化为 fp8(1B)，
**显存占用减半** → 同样显存可容纳 2x 的上下文/并发。与 Mimir 其它优化正交。

对比：bf16 KV vs fp8 KV，度量：
- KV cache 总块数（fp8 应 ~2x bf16，因同样显存放更多块）
- 实际可用 KV 显存
- 任务成功率（fp8 量化对效果的影响，核心：基本不下降）
- TTFT / new_prefill（口径一致）

输出：benchmark_results/phase7_fp8kv_<model>.json + _cmp.png

用法（mimir 环境）：
    python scripts/run_phase7_fp8kv.py
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
from mimir.metrics import RunMetrics, save_results
from mimir.plots import plot_latency_comparison


def run_with_kv_dtype(
    model: str, gpu_util: float, kv_dtype: str | None, max_tokens: int, max_model_len: int
) -> tuple[VLLMEngine, list[RunMetrics]]:
    cfg = EngineConfig(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_util,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
        use_v1=False,
        kv_cache_dtype=kv_dtype,
    )
    eng = VLLMEngine(cfg, device=0)
    _ = eng.llm
    # KV 总块数（fp8 应更大）
    kv = eng.kv_usage()
    print(
        f"  [{kv_dtype or 'bf16'}] total_kv_blocks={kv.get('total_blocks')} "
        f"used={kv.get('used_blocks')}",
        flush=True,
    )
    results = []
    for name, case in all_workloads().items():
        m = run_workload(eng, case, max_tokens=max_tokens, label=f"{kv_dtype or 'bf16'}_{name}")
        results.append(m)
    return eng, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    tag = Path(args.model).name
    print("\n=== bf16 KV ===", flush=True)
    eng_bf, res_bf = run_with_kv_dtype(
        args.model, args.gpu_memory_util, None, args.max_tokens, args.max_model_len
    )
    kv_bf = eng_bf.kv_usage().get("total_blocks")
    del eng_bf

    print("\n=== fp8 KV ===", flush=True)
    eng_fp, res_fp = run_with_kv_dtype(
        args.model, args.gpu_memory_util, "fp8", args.max_tokens, args.max_model_len
    )
    kv_fp = eng_fp.kv_usage().get("total_blocks")
    del eng_fp

    print(
        f"\nKV 总块数: bf16={kv_bf}  fp8={kv_fp}  "
        f"(fp8/bf16 = {(kv_fp / kv_bf) if kv_bf else 0:.2f}x)",
        flush=True,
    )

    # 汇总对比
    summary = {
        "model": tag,
        "bf16_kv_total_blocks": kv_bf,
        "fp8_kv_total_blocks": kv_fp,
        "fp8_capacity_gain": round((kv_fp / kv_bf) if kv_bf else 0, 2),
        "per_workload": [],
    }
    for rb, rf in zip(res_bf, res_fp, strict=False):
        wl = rb.extra["workload"]
        row = {
            "workload": wl,
            "bf16": {
                "ttft_ms": rb.ttft_ms,
                "new_prefill": rb.extra.get("total_prefill_new_tokens"),
                "success": rb.task_success,
            },
            "fp8": {
                "ttft_ms": rf.ttft_ms,
                "new_prefill": rf.extra.get("total_prefill_new_tokens"),
                "success": rf.task_success,
            },
        }
        summary["per_workload"].append(row)
        bf_ttft = f"{rb.ttft_ms:.1f}" if rb.ttft_ms is not None else "ERR"
        fp_ttft = f"{rf.ttft_ms:.1f}" if rf.ttft_ms is not None else "ERR"
        print(
            f"  [{wl}] bf16 TTFT={bf_ttft}ms ok={rb.task_success} | "
            f"fp8 TTFT={fp_ttft}ms ok={rf.task_success}",
            flush=True,
        )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"phase7_fp8kv_{tag}.json"
    save_results(res_bf + res_fp, json_path)
    (out_dir / f"phase7_fp8kv_{tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_latency_comparison(
        res_bf + res_fp,
        out_dir / f"phase7_fp8kv_{tag}_lat.png",
        title="Phase 7a KV 量化：bf16 vs fp8 延迟",
    )
    print(f"\n保存: {json_path}")
    print(f"保存: {out_dir / f'phase7_fp8kv_{tag}_summary.json'}")
    print("PHASE7A_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
