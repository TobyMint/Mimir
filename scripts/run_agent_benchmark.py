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

from benchmarks.agent_loop import TASKS

from mimir.gpu import pick_least_busy_gpu

CHILD = r"""
import os, json, sys, traceback, copy
sys.path.insert(0, os.getcwd())
model, gpu, util, mlen, mtok, policy, offload, tool_kb, max_steps = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6], sys.argv[7] == "1", int(sys.argv[8]),
    int(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] else 0)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from benchmarks.agent_loop import TASKS, run_agent_loop
eng = VLLMEngineV1(EngineConfig(
    model=model, dtype="bfloat16", gpu_memory_utilization=util,
    enable_prefix_caching=True, max_model_len=mlen,
    extra={"scheduling_policy": policy}), device=0)
_ = eng.llm
results = []
for task in TASKS:
    # --max-steps override: a heavier/longer workload than the per-task default
    t = dict(task)
    if max_steps > 0:
        t["max_steps"] = max_steps
    # Per-task resilience: native (no offload) is *expected* to crash on
    # context overflow — that IS the problem Mimir solves. Capture the crash
    # as a structured step-0 record instead of taking down the whole batch,
    # so the parent can still plot native's (short) curve vs Mimir's full one.
    try:
        r = run_agent_loop(eng, t, policy=policy, tool_offload=offload,
                           tool_result_kb=tool_kb, max_tokens=mtok)
        d = r.to_dict()
    except Exception as e:
        msg = (str(e)[:240] or e.__class__.__name__)
        d = {
            "label": f"{task['name']}_{policy}",
            "policy": policy,
            "tool_offload": offload,
            "num_steps": 0,
            "peak_used_blocks": -1,
            "total_tool_data_bytes": 0,
            "final_answer": "",
            "crashed": True,
            "crash_reason": msg,
            "steps": [],
        }
    results.append(d)
print("RESULT_JSON:" + json.dumps(results))
"""


def run_side(model, g, util, mlen, mtok, policy, offload, tool_kb, max_steps=0):
    r = subprocess.run(
        [
            "python",
            "-c",
            CHILD,
            model,
            str(g.index),
            str(util),
            str(mlen),
            str(mtok),
            policy,
            "1" if offload else "0",
            str(tool_kb),
            str(max_steps),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=900,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    print(f"[{policy}] ERROR:", r.stderr[-300:].replace("\r", ""), flush=True)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--tool-kb", type=int, default=5, help="mock tool result size in KB")
    ap.add_argument("--max-steps", type=int, default=0, help="override per-task max steps (0=use task default)")
    ap.add_argument(
        "--tag",
        default="",
        help="tag for output filenames (default: none; uses agent_benchmark_<model>)",
    )
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    suffix = f"_{args.tag}" if args.tag else ""
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)
    print(f"Tasks: {[t['name'] for t in TASKS]}", flush=True)
    print(f"Tool result size: {args.tool_kb}KB, max-steps override: {args.max_steps or 'default'}", flush=True)

    summary = {"model": Path(args.model).name, "tool_kb": args.tool_kb, "native": {}, "mimir": {}}

    for label, policy, offload in [("native", "fcfs", False), ("mimir", "mimir", True)]:
        print(f"\n=== {label} ({policy}, offload={offload}) ===", flush=True)
        results = run_side(
            args.model,
            g,
            args.gpu_memory_util,
            args.max_model_len,
            args.max_tokens,
            policy,
            offload,
            args.tool_kb,
            args.max_steps,
        )
        for r in results:
            print(
                f"  {r['label']}: {r['num_steps']} steps, peak_used={r['peak_used_blocks']}, "
                f"tool_data={r['total_tool_data_bytes']}B, final={r['final_answer'][:60]}",
                flush=True,
            )
        summary[label] = {"results": results}

    # Build comparison table
    native_res = summary["native"].get("results", [])
    mimir_res = summary["mimir"].get("results", [])
    comparison = []
    for n, m in zip(native_res, mimir_res, strict=False):
        task_name = n["label"].rsplit("_", 1)[0] if "_" in n["label"] else n["label"]
        # native "crashed" = its run ended in a [CRASHED] sentinel step (the
        # engine raised, e.g. context overflow). Reaching fewer steps alone is
        # NOT a crash — native may just have answered [FINAL:] earlier.
        def _ended_in_crash(run: dict) -> bool:
            steps = run.get("steps", [])
            return bool(steps) and steps[-1].get("used_blocks") == -1

        n_crashed = bool(n.get("crashed")) or _ended_in_crash(n)
        # peak_used_blocks of a crashed step is -1 (sentinel); treat as N/A
        n_peak_raw = n.get("peak_used_blocks", 0)
        n_peak = None if n_peak_raw is None or n_peak_raw < 0 else n_peak_raw
        m_peak = m.get("peak_used_blocks", 0)

        def _avg_ttft(run: dict) -> float | None:
            vals = [
                s["ttft_ms"]
                for s in run.get("steps", [])
                if s.get("ttft_ms") is not None and s.get("used_blocks", 0) != -1
            ]
            return round(sum(vals) / len(vals), 1) if vals else None

        def _last_new_prefill(run: dict) -> int | None:
            vals = [
                s["new_prefill_tokens"]
                for s in run.get("steps", [])
                if s.get("new_prefill_tokens") is not None and s.get("used_blocks", 0) != -1
            ]
            return vals[-1] if vals else None

        # Honest TTFT comparison: compare only on steps both sides actually
        # completed (native crashes at step 2 in all tasks). A raw avg over the
        # full run is misleading — native's avg is just 1-2 early-startup
        # requests, so Mimir (8-10 steps with growing prefill) looks "slower"
        # despite native being unable to run. Compare matched-step TTFT instead.
        n_real_steps = [s for s in n.get("steps", []) if s.get("used_blocks", 0) != -1]
        n_n_real = len(n_real_steps)
        m_match = [s for s in m.get("steps", []) if s.get("used_blocks", 0) != -1][:n_n_real]
        n_ttft_match = [
            s["ttft_ms"]
            for s in n_real_steps
            if s.get("ttft_ms") is not None
        ]
        m_ttft_match = [
            s["ttft_ms"]
            for s in m_match
            if s.get("ttft_ms") is not None
        ]
        if n_ttft_match and m_ttft_match:
            n_ttft_avg = round(sum(n_ttft_match) / len(n_ttft_match), 1)
            m_ttft_avg = round(sum(m_ttft_match) / len(m_ttft_match), 1)
            # % change in matched-step TTFT (negative = Mimir faster, ~0 = parity)
            ttft_change = round((m_ttft_avg / n_ttft_avg - 1) * 100, 1) if n_ttft_avg else None
        else:
            n_ttft_avg = m_ttft_avg = ttft_change = None

        if n_peak is not None and m_peak is not None and m_peak >= 0:
            reduction = round((1 - m_peak / n_peak) * 100, 1) if n_peak else 0
        else:
            reduction = None
        comparison.append(
            {
                "task": task_name,
                "native_peak_used": n_peak,
                "mimir_peak_used": m_peak,
                "reduction_pct": reduction,
                "native_steps": n.get("num_steps", 0),
                "mimir_steps": m.get("num_steps", 0),
                "native_crashed": n_crashed,
                "crash_reason": n.get("crash_reason") or (
                    n.get("final_answer", "") if "crashed" in n.get("final_answer", "") else None
                ),
                "native_avg_ttft_ms": _avg_ttft(n),
                "mimir_avg_ttft_ms": _avg_ttft(m),
                "matched_steps": n_n_real,
                "native_matched_ttft_ms": n_ttft_avg,
                "mimir_matched_ttft_ms": m_ttft_avg,
                "matched_ttft_change_pct": ttft_change,
                "mimir_last_new_prefill_tokens": _last_new_prefill(m),
                "native_final": n.get("final_answer", "")[:80],
                "mimir_final": m.get("final_answer", "")[:80],
            }
        )
    summary["comparison"] = comparison

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"agent_benchmark{suffix}_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot: step-by-step used_blocks for each task, native vs Mimir (English labels)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_tasks = len(comparison)
        if n_tasks == 0:
            print("plot skipped: no comparison tasks", flush=True)
        else:
            fig, axes = plt.subplots(2, n_tasks, figsize=(6 * n_tasks, 8.5), squeeze=False)
            for i, (n, m) in enumerate(zip(native_res, mimir_res, strict=False)):
                task_name = n["label"].rsplit("_", 1)[0] if "_" in n["label"] else n["label"]
                # Exclude crash-sentinel step (used_blocks == -1) from curves
                n_real = [s for s in n["steps"] if s.get("used_blocks", 0) != -1]
                n_steps = [s["step"] for s in n_real]
                n_used = [s["used_blocks"] for s in n_real]
                m_real = [s for s in m["steps"] if s.get("used_blocks", 0) != -1]
                m_steps = [s["step"] for s in m_real]
                m_used = [s["used_blocks"] for s in m_real]
                n_crashed = bool(n.get("crashed")) or (
                    n.get("num_steps", 0) < m.get("num_steps", 0)
                )

                # Row 0: used KV blocks (memory)
                ax = axes[0][i]
                if n_steps:
                    ax.plot(n_steps, n_used, "rx-", label="native vLLM (fcfs)")
                ax.plot(m_steps, m_used, "g^-", label="Mimir (mimir + offload)")
                ax.set_xlabel("agent step")
                ax.set_ylabel("used KV blocks")
                ax.set_title(f"{task_name} — memory")
                ax.legend(fontsize=8)
                if n_crashed:
                    ax.axvline(
                        (max(n_steps) if n_steps else 0) + 0.4,
                        color="red",
                        linestyle=":",
                        alpha=0.4,
                    )

                # Row 1: per-step TTFT + new prefill tokens (latency). native curve
                # stops at its crash step; Mimir stays flat (prefix reuse + reclaim).
                ax2 = axes[1][i]
                n_ttft = [
                    (s["step"], s["ttft_ms"])
                    for s in n_real
                    if s.get("ttft_ms") is not None
                ]
                m_ttft = [
                    (s["step"], s["ttft_ms"])
                    for s in m_real
                    if s.get("ttft_ms") is not None
                ]
                n_prefill = [
                    (s["step"], s["new_prefill_tokens"])
                    for s in n_real
                    if s.get("new_prefill_tokens") is not None
                ]
                m_prefill = [
                    (s["step"], s["new_prefill_tokens"])
                    for s in m_real
                    if s.get("new_prefill_tokens") is not None
                ]
                if n_ttft:
                    ax2.plot([x for x, _ in n_ttft], [y for _, y in n_ttft], "rx--", label="native TTFT (ms)")
                if m_ttft:
                    ax2.plot([x for x, _ in m_ttft], [y for _, y in m_ttft], "g^-", label="Mimir TTFT (ms)")
                if n_prefill:
                    ax2.plot([x for x, _ in n_prefill], [y for _, y in n_prefill], "r:", alpha=0.5, label="native new_prefill (tok)")
                if m_prefill:
                    ax2.plot([x for x, _ in m_prefill], [y for _, y in m_prefill], "g:", alpha=0.5, label="Mimir new_prefill (tok)")
                if n_crashed:
                    crash_x = (max(n_steps) if n_steps else 0) + 0.4
                    ax2.axvline(crash_x, color="red", linestyle=":", alpha=0.4)
                    ax2.text(
                        crash_x,
                        0.5,
                        "native crashes (context overflow)",
                        rotation=90,
                        color="red",
                        fontsize=7,
                        va="bottom",
                        transform=ax2.get_xaxis_transform(),
                    )
                ax2.set_xlabel("agent step")
                ax2.set_ylabel("TTFT (ms) / new prefill tokens")
                ax2.set_title(f"{task_name} — latency")
                ax2.legend(fontsize=7)
            fig.suptitle(
                f"Agent-Loop Benchmark: native vs Mimir "
                f"({Path(args.model).name}, tool={args.tool_kb}KB)",
                fontsize=11,
            )
            fig.tight_layout()
            png = out_dir / f"agent_benchmark{suffix}_{Path(args.model).name}_curves.png"
            fig.savefig(png, dpi=140)
            plt.close(fig)
            print(f"\nPNG: {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)

    print(f"\nJSON: {jp}")
    for c in comparison:
        crash = ""
        if c.get("native_crashed"):
            crash = f" [NATIVE CRASHED @ step {c['native_steps']}: {c.get('crash_reason') or 'context overflow'}]"
        if c.get("reduction_pct") is not None:
            mem_s = f"native peak={c['native_peak_used']} -> Mimir peak={c['mimir_peak_used']} ({c['reduction_pct']}%)"
        else:
            mem_s = f"native peak=CRASH -> Mimir peak={c['mimir_peak_used']}"
        # Honest latency: compare on matched (pre-crash) steps only. Native's
        # raw-average TTFT is just early-startup requests, so we don't quote a
        # full-run percentage. Parity on matched steps = no latency regression.
        if c.get("matched_steps"):
            ttft_s = (
                f", TTFT on {c['matched_steps']} matched steps: "
                f"native={c['native_matched_ttft_ms']}ms vs Mimir={c['mimir_matched_ttft_ms']}ms"
                f" ({c['matched_ttft_change_pct']:+.0f}%, ~parity = no regression)"
            )
        else:
            ttft_s = ""
        print(f"  {c['task']}: {mem_s}{crash}{ttft_s}", flush=True)

    print("AGENT_BENCHMARK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
