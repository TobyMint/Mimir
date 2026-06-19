"""Phase Q：工具调用密集并发 A/B（赛题 tool_call 场景，最贴合）。

3 个 agent 并发，每个 agent 跑 2 轮工具调用（含大返回 ~5KB）：
- native：fcfs，大工具返回全量进 KV，3 agent × 2 轮 KV 累积
- Mimir：mimir 策略 + tool_offload（大返回外置，上下文留引用）+ 逐任务自动回收

度量：峰值 used_blocks、TTFT（大返回导致 native prefill 重）、reclaims。
输出：benchmark_results/phase_q_toolcall_concurrent_<model>.json + _curves.png

用法：python scripts/run_phase_q_toolcall.py
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

from mimir.gpu import pick_least_busy_gpu

CHILD = r"""
import os, json, sys, traceback
sys.path.insert(0, os.getcwd())
try:
    from mimir.engine_vllm import EngineConfig
    from mimir.engine_vllm_v1 import VLLMEngineV1
    from mimir.tools.offload import ToolDataStore
    from mimir.manager import MemoryManager
    from mimir.context.compressor import Fidelity
    model, gpu, policy, offload = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]=="1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    eng = VLLMEngineV1(EngineConfig(model=model, dtype="bfloat16", gpu_memory_utilization=0.50,
        enable_prefix_caching=True, max_model_len=4096, extra={"scheduling_policy": policy}), device=0)
    _ = eng.llm
    store = ToolDataStore() if offload else None
    AGENTS = [("A","math","Solve 17*23 step by step."),("B","history","When was the French Revolution?"),("C","code","Reverse a list in Python.")]
    BIG = "[" + ", ".join('{"result":"long computation output chunk '+str(i)+'"}' for i in range(60)) + "]"
    rows=[]
    for agent,role,q in AGENTS:
        for rnd in range(2):
            tid=f"{agent}_{rnd}"
            eng.set_current_task(tid)
            sysmsg=f"You are agent {agent}, a {role} expert. Be brief."
            user=q+f" Round {rnd}."
            tool_content = store.put("search", BIG) if store else BIG
            msgs=[{"role":"system","content":sysmsg},{"role":"user","content":user},
                  {"role":"assistant","content":"[tool: search]"},{"role":"tool","content":tool_content}]
            eng.chat(msgs, max_tokens=14)
            st=eng.mimir_stats()
            rows.append({"agent":agent,"rnd":rnd,"used":st.get("used_blocks"),"reclaims":st.get("mimir_lifecycle_reclaims")})
    print("RESULT_JSON:"+json.dumps({"rows":rows}))
except Exception:
    traceback.print_exc()
"""


def run_side(model, g, policy, offload):
    r = subprocess.run(
        ["python", "-c", CHILD, model, str(g.index), policy, "1" if offload else "0"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=400,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    print(f"[{policy}/off={offload}] ERROR:", r.stderr[-300:].replace("\r", ""), flush=True)
    return {"rows": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    print(f"GPU {g.index}", flush=True)

    print("\n=== native (fcfs, big tool results in KV) ===", flush=True)
    native = run_side(args.model, g, "fcfs", False)
    print("\n=== Mimir (mimir policy + tool_offload + auto-reclaim) ===", flush=True)
    mimir = run_side(args.model, g, "mimir", True)

    def peak(rs):
        return max((r.get("used") or 0) for r in rs) if rs else 0

    n_peak, m_peak = peak(native["rows"]), peak(mimir["rows"])
    m_reclaims = mimir["rows"][-1].get("reclaims", 0) if mimir["rows"] else 0

    summary = {
        "model": Path(args.model).name,
        "scenario": "3 agents x 2 rounds, each with ~5KB tool result (tool_call workload)",
        "native": {"peak_used_blocks": n_peak, "rows": native["rows"]},
        "mimir": {
            "peak_used_blocks": m_peak,
            "lifecycle_reclaims": m_reclaims,
            "rows": mimir["rows"],
        },
        "headline": f"native peak used={n_peak} (big results in KV); Mimir peak used={m_peak} (offload+reclaim, reclaims={m_reclaims})",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_q_toolcall_concurrent_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(native["rows"])))
        n_used = [r.get("used") or 0 for r in native["rows"]]
        m_used = [r.get("used") or 0 for r in mimir["rows"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(steps, n_used, "rx-", label="native (big tool results in KV)")
        ax.plot(steps, m_used, "g^-", label="Mimir (offload + auto-reclaim)")
        ax.set_xlabel("agent round (A0,B0,C0,A1,B1,C1)")
        ax.set_ylabel("used KV blocks")
        ax.set_title(
            f"Phase Q: tool-call concurrent (3 agents x2, ~5KB results, {Path(args.model).name})"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        png = out_dir / f"phase_q_toolcall_concurrent_{Path(args.model).name}_curves.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"保存: {png}", flush=True)
    except Exception as e:
        print(f"画图跳过: {e}", flush=True)

    print(f"native peak={n_peak} | Mimir peak={m_peak} reclaims={m_reclaims}")
    print("PHASE_Q_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
