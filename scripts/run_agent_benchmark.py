# ruff: noqa: E501
"""Unified agent-loop benchmark: native vLLM vs Mimir, real agent tasks.

Runs 3 agent tasks (research / compare / estimate) as real agent loops:
  LLM generates -> parse tool_call -> execute mock tool -> result back -> LLM again

A/B: same task, same seed, same GPU:
  - native:  fcfs policy, tool results full in KV, no reclaim
  - Mimir:   mimir policy + tool_offload + per-step auto-reclaim

Output: benchmark_results/agent_benchmark_<model>.json + _curves.png (English labels)

Usage: python scripts/run_agent_benchmark.py [--repeats 3] [--tool-kb 5]
"""

from __future__ import annotations

# ruff: noqa: E501
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.agent_loop import TASKS, run_agent_loop
from mimir.gpu import pick_least_busy_gpu

CHILD = r"""
import os, json, sys, traceback
sys.path.insert(0, os.getcwd())
try:
    from mimir.engine_vllm import EngineConfig
    from mimir.engine_vllm_v1 import VLLMEngineV1
    from benchmarks.agent_loop import TASKS, run_agent_loop
    model, gpu, util, mlen, mtok, policy, offload, tool_kb = (
        sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
        int(sys.argv[5]), sys.argv[6], sys.argv[7] == "1", int(sys.argv[8]))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    eng = VLLMEngineV1(EngineConfig(
        model=model, dtype="bfloat16", gpu_memory_utilization=util,
        enable_prefix_caching=True, max_model_len=mlen,
        extra={"scheduling_policy": policy}), device=0)
    _ = eng.llm
    results = []
    for task in TASKS:
        r = run_agent_loop(eng, task, policy=policy, tool_offload=offload,
                           tool_result_kb=tool_kb, max_tokens=mtok)
        results.append(r.to_dict())
    print("RESULT_JSON:" + json.dumps(results))
except Exception:
    traceback.print_exc()
"""


def run_side(model, g, util, mlen, mtok, policy, offload, tool_kb):
    r = subprocess.run(
        ["python", "-c", CHILD, model, str(g.index), str(util), str(mlen),
         str(mtok), policy, "1" if offload else "0", str(tool_kb)],
        capture_output=True, text=True, env=dict(os.environ), timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    print(f"[{policy}] ERROR:", r.stderr[-300:].replace("\r", ""), flush=True)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--tool-kb", type=int, default=5, help="mock tool result size in KB")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)
    print(f"Tasks: {[t['name'] for t in TASKS]}", flush=True)
    print(f"Tool result size: {args.tool_kb}KB", flush=True)

    summary = {"model": Path(args.model).name, "tool_kb": args.tool_kb, "native": {}, "mimir": {}}

    for label, policy, offload in [("native", "fcfs", False), ("mimir", "mimir", True)]:
        print(f"\n=== {label} ({policy}, offload={offload}) ===", flush=True)
        results = run_side(args.model, g, args.gpu_memory_util, args.max_model_len,
                           args.max_tokens, policy, offload, args.tool_kb)
        for r in results:
            print(f"  {r['label']}: {r['num_steps']} steps, peak_used={r['peak_used_blocks']}, "
                  f"tool_data={r['total_tool_data_bytes']}B, final={r['final_answer'][:60]}", flush=True)
        summary[label] = {"results": results}

    # Build comparison table
    native_res = summary["native"].get("results", [])
    mimir_res = summary["mimir"].get("results", [])
    comparison = []
    for n, m in zip(native_res, mimir_res, strict=False):
        task_name = n["label"].rsplit("_", 1)[0] if "_" in n["label"] else n["label"]
        n_peak = n.get("peak_used_blocks", 0)
        m_peak = m.get("peak_used_blocks", 0)
        comparison.append({
            "task": task_name,
            "native_peak_used": n_peak,
            "mimir_peak_used": m_peak,
            "reduction_pct": round((1 - m_peak / n_peak) * 100, 1) if n_peak else 0,
            "native_steps": n.get("num_steps", 0),
            "mimir_steps": m.get("num_steps", 0),
            "native_final": n.get("final_answer", "")[:80],
            "mimir_final": m.get("final_answer", "")[:80],
        })
    summary["comparison"] = comparison

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"agent_benchmark_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot: step-by-step used_blocks for each task, native vs Mimir (English labels)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_tasks = len(comparison)
        fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 4.5), squeeze=False)
        for i, (n, m) in enumerate(zip(native_res, mimir_res, strict=False)):
            ax = axes[0][i]
            n_steps = [s["step"] for s in n["steps"]]
            n_used = [s["used_blocks"] for s in n["steps"]]
            m_steps = [s["step"] for s in m["steps"]]
            m_used = [s["used_blocks"] for s in m["steps"]]
            task_name = n["label"].rsplit("_", 1)[0] if "_" in n["label"] else n["label"]
            ax.plot(n_steps, n_used, "rx-", label="native vLLM (fcfs)")
            ax.plot(m_steps, m_used, "g^-", label="Mimir (mimir + offload)")
            ax.set_xlabel("agent step")
            ax.set_ylabel("used KV blocks")
            ax.set_title(task_name)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Agent-Loop Benchmark: native vs Mimir "
            f"({Path(args.model).name}, tool={args.tool_kb}KB)",
            fontsize=11,
        )
        fig.tight_layout()
        png = out_dir / f"agent_benchmark_{Path(args.model).name}_curves.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"\nPNG: {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)

    print(f"\nJSON: {jp}")
    for c in comparison:
        print(f"  {c['task']}: native peak={c['native_peak_used']} -> Mimir peak={c['mimir_peak_used']} "
              f"({c['reduction_pct']}%)")
    print("AGENT_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
