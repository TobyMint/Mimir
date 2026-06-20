# ruff: noqa: E501, E701, E702
"""DeepSeek-trace A/B benchmark: native vLLM vs Mimir on REAL agent trajectories.

Replays DeepSeek-V4-Pro-generated agent traces through Qwen3-4B under both:
  - native:  fcfs policy, tool results in-context, no reclaim
  - Mimir:   mimir policy + tool_offload + per-step auto-reclaim

Same trace, same seed, same GPU => clean A/B. The traces reflect a frontier
model's real agent behavior (tool calls, growing context), replacing the
weak-model-driven trajectories used by run_agent_benchmark.py.

Output: benchmark_results/trace_bench_<model>.json + _curves.png (English labels)

Usage: python scripts/run_trace_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.gpu import pick_least_busy_gpu

CHILD = r"""
import os, json, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
model, gpu, util, mlen, mtok, policy, offload, trace_dir = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6], sys.argv[7] == "1", sys.argv[8])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from benchmarks.trace_replay import replay_trace
eng = VLLMEngineV1(EngineConfig(
    model=model, dtype="bfloat16", gpu_memory_utilization=util,
    enable_prefix_caching=True, max_model_len=mlen,
    extra={"scheduling_policy": policy}), device=0)
_ = eng.llm
results = []
for p in sorted(Path(trace_dir).glob("*.json")):
    trace = json.loads(p.read_text(encoding="utf-8"))
    r = replay_trace(eng, trace, policy=policy, tool_offload=offload, max_tokens=mtok)
    results.append(r.to_dict())
print("RESULT_JSON:" + json.dumps(results))
"""


def run_side(model, g, util, mlen, mtok, policy, offload, trace_dir):
    r = subprocess.run(
        ["python", "-c", CHILD, model, str(g.index), str(util), str(mlen),
         str(mtok), policy, "1" if offload else "0", trace_dir],
        capture_output=True, text=True, env=dict(os.environ), timeout=900,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    print(f"[{policy}] no RESULT_JSON (rc={r.returncode}):", r.stderr[-600:].replace("\r", ""), flush=True)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--trace-dir", default="benchmark_results/traces")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)
    trace_dir = Path(args.trace_dir)
    traces = sorted(trace_dir.glob("*.json"))
    print(f"Traces: {[p.stem for p in traces]}", flush=True)

    summary = {"model": Path(args.model).name, "trace_source": "deepseek-v4-pro",
               "native": {}, "mimir": {}}
    for label, policy, offload in [("native", "fcfs", False), ("mimir", "mimir", True)]:
        print(f"\n=== {label} ({policy}, offload={offload}) ===", flush=True)
        res = run_side(args.model, g, args.gpu_memory_util, args.max_model_len,
                       args.max_tokens, policy, offload, str(trace_dir))
        for r in res:
            n_real = [s for s in r["steps"] if s.get("used_blocks", 0) != -1]
            print(f"  {r['label']}: {r['num_steps']} steps, peak_used={r['peak_used_blocks']}, "
                  f"tool_offloaded={r['total_tool_data_bytes']}B", flush=True)
        summary[label] = {"results": res}

    # Comparison
    native_res = summary["native"].get("results", [])
    mimir_res = summary["mimir"].get("results", [])
    comparison = []
    for n, m in zip(native_res, mimir_res, strict=False):
        task = n["label"].rsplit("_", 1)[0] if "_" in n["label"] else n["label"]

        def _ended_in_crash(run):
            steps = run.get("steps", [])
            return bool(steps) and steps[-1].get("used_blocks") == -1
        n_crashed = _ended_in_crash(n)

        n_peak_raw = n.get("peak_used_blocks", 0)
        n_peak = None if n_peak_raw is None or n_peak_raw < 0 else n_peak_raw
        m_peak = m.get("peak_used_blocks", 0)

        # Matched-step TTFT (native crashes early in heavy traces)
        n_real = [s for s in n.get("steps", []) if s.get("used_blocks", 0) != -1]
        n_n = len(n_real)
        m_match = [s for s in m.get("steps", []) if s.get("used_blocks", 0) != -1][:n_n]
        nt = [s["ttft_ms"] for s in n_real if s.get("ttft_ms") is not None]
        mt = [s["ttft_ms"] for s in m_match if s.get("ttft_ms") is not None]
        if nt and mt:
            n_avg = round(sum(nt) / len(nt), 1)
            m_avg = round(sum(mt) / len(mt), 1)
            change = round((m_avg / n_avg - 1) * 100, 1) if n_avg else None
        else:
            n_avg = m_avg = change = None

        # Matched-step new_prefill (last value)
        n_np = [s["new_prefill_tokens"] for s in n_real if s.get("new_prefill_tokens") is not None]
        m_np = [s["new_prefill_tokens"] for s in m_match if s.get("new_prefill_tokens") is not None]
        n_lastnp = n_np[-1] if n_np else None
        m_lastnp = m_np[-1] if m_np else None

        reduction = round((1 - m_peak / n_peak) * 100, 1) if (n_peak and m_peak and m_peak >= 0) else None
        comparison.append({
            "task": task, "native_peak_used": n_peak, "mimir_peak_used": m_peak,
            "reduction_pct": reduction, "native_steps": n.get("num_steps", 0),
            "mimir_steps": m.get("num_steps", 0), "native_crashed": n_crashed,
            "native_matched_ttft_ms": n_avg, "mimir_matched_ttft_ms": m_avg,
            "matched_ttft_change_pct": change, "matched_steps": n_n,
            "native_last_new_prefill": n_lastnp, "mimir_last_new_prefill": m_lastnp,
            "mimir_tool_offloaded_bytes": m.get("total_tool_data_bytes", 0),
        })
    summary["comparison"] = comparison

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"trace_bench_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_tasks = len(comparison)
        if n_tasks:
            fig, axes = plt.subplots(2, n_tasks, figsize=(6 * n_tasks, 8.5), squeeze=False)
            for i, (n, m) in enumerate(zip(native_res, mimir_res, strict=False)):
                name = n["label"].rsplit("_", 1)[0] if "_" in n["label"] else n["label"]
                n_real = [s for s in n["steps"] if s.get("used_blocks", 0) != -1]
                m_real = [s for s in m["steps"] if s.get("used_blocks", 0) != -1]
                ax = axes[0][i]
                if n_real:
                    ax.plot([s["step"] for s in n_real], [s["used_blocks"] for s in n_real],
                            "rx-", label="native vLLM (fcfs)")
                ax.plot([s["step"] for s in m_real], [s["used_blocks"] for s in m_real],
                        "g^-", label="Mimir (mimir + offload)")
                ax.set_xlabel("trace step"); ax.set_ylabel("used KV blocks")
                ax.set_title(f"{name} — memory"); ax.legend(fontsize=8)
                ax2 = axes[1][i]
                ntt = [(s["step"], s["ttft_ms"]) for s in n_real if s.get("ttft_ms") is not None]
                mtt = [(s["step"], s["ttft_ms"]) for s in m_real if s.get("ttft_ms") is not None]
                npp = [(s["step"], s["new_prefill_tokens"]) for s in n_real if s.get("new_prefill_tokens") is not None]
                mpp = [(s["step"], s["new_prefill_tokens"]) for s in m_real if s.get("new_prefill_tokens") is not None]
                if ntt: ax2.plot([x for x, _ in ntt], [y for _, y in ntt], "rx--", label="native TTFT(ms)")
                if mtt: ax2.plot([x for x, _ in mtt], [y for _, y in mtt], "g^-", label="Mimir TTFT(ms)")
                if npp: ax2.plot([x for x, _ in npp], [y for _, y in npp], "r:", alpha=0.5, label="native new_prefill")
                if mpp: ax2.plot([x for x, _ in mpp], [y for _, y in mpp], "g:", alpha=0.5, label="Mimir new_prefill")
                ax2.set_xlabel("trace step"); ax2.set_ylabel("TTFT(ms) / new_prefill(tok)")
                ax2.set_title(f"{name} — latency"); ax2.legend(fontsize=7)
            fig.suptitle(f"DeepSeek-Trace Benchmark: native vs Mimir ({Path(args.model).name})",
                         fontsize=11)
            fig.tight_layout()
            png = out / f"trace_bench_{Path(args.model).name}_curves.png"
            fig.savefig(png, dpi=140); plt.close(fig)
            print(f"\nPNG: {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)

    print(f"\nJSON: {jp}")
    for c in comparison:
        crash = " [NATIVE CRASHED]" if c.get("native_crashed") else ""
        ttft = (f", matched TTFT({c['matched_steps']}): {c['native_matched_ttft_ms']}vs"
                f"{c['mimir_matched_ttft_ms']}ms ({c['matched_ttft_change_pct']}%)") if c.get("matched_steps") else ""
        print(f"  {c['task']}: native peak={c['native_peak_used']} -> Mimir peak={c['mimir_peak_used']} "
              f"({c.get('reduction_pct')}%){crash}{ttft}", flush=True)
    print("TRACE_BENCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
