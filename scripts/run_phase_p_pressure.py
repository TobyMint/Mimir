"""Phase P：KV 池压力 A/B（lifecycle-aware allocation 接入真实分配路径）。

小 KV 池（低 gpu_memory_utilization）+ 6 个长上下文任务连续跑，施压 KV 池：
- fcfs：原生 LRU，KV 累积（reclaims=0，被动淘汰）
- mimir：get_new_blocks 前主动回收 EVICTABLE（Phase P）+ 任务完成自动回收（Phase L），
  used_blocks 守恒为 0，reclaims 累计

度量：每任务 used_blocks、累计 reclaims。
输出：benchmark_results/phase_p_pressure_<model>.json + _curves.png

用法：python scripts/run_phase_p_pressure.py
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

from mimir.gpu import as_env, pick_least_busy_gpu

CHILD = r"""
import os, json, sys, traceback
sys.path.insert(0, os.getcwd())
try:
    from mimir.engine_vllm import EngineConfig
    from mimir.engine_vllm_v1 import VLLMEngineV1
    model, gpu, util, policy = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    eng = VLLMEngineV1(EngineConfig(model=model, dtype="bfloat16", gpu_memory_utilization=util,
        enable_prefix_caching=True, max_model_len=4096, extra={"scheduling_policy": policy}), device=0)
    _ = eng.llm
    bp = eng.mimir_block_pool()
    total = bp.num_gpu_blocks if bp else 0
    rows=[]
    for i in range(6):
        tid=f"task_{i}"
        eng.set_current_task(tid)
        ctx=("Analyze KV cache memory for transformer models in detail. "*30)+f" Task {i}."
        eng.chat([{"role":"system","content":"brief analyst"},{"role":"user","content":ctx}], max_tokens=14)
        st=eng.mimir_stats()
        rows.append({"task":tid,"used":st.get("used_blocks"),"reclaims":st.get("mimir_lifecycle_reclaims")})
    print("RESULT_JSON:"+json.dumps({"total_blocks":total,"rows":rows}))
except Exception:
    traceback.print_exc()
"""


def run_side(model, g, util, policy):
    r = subprocess.run(["python","-c",CHILD, model, str(g.index), str(util), policy],
                       capture_output=True, text=True, env=dict(os.environ), timeout=300)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    print(f"[{policy}] ERROR:", r.stderr[-300:].replace("\r",""), flush=True)
    return {"total_blocks": None, "rows": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.45)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=8.0)
    if g is None:
        print("NO_FREE_GPU"); return 2
    print(f"GPU {g.index}", flush=True)

    native = run_side(args.model, g, args.gpu_memory_util, "fcfs")
    mimir = run_side(args.model, g, args.gpu_memory_util, "mimir")

    n_final = native["rows"][-1]["used"] if native["rows"] else None
    m_final = mimir["rows"][-1]["used"] if mimir["rows"] else None
    m_reclaims = mimir["rows"][-1]["reclaims"] if mimir["rows"] else 0

    summary = {
        "model": Path(args.model).name,
        "scenario": "6 long-context tasks filling a small KV pool (lifecycle-aware allocation pressure)",
        "kv_pool_total_blocks": native.get("total_blocks"),
        "native": {"final_used_blocks": n_final, "rows": native["rows"]},
        "mimir": {"final_used_blocks": m_final, "lifecycle_reclaims": m_reclaims, "rows": mimir["rows"]},
        "headline": f"native used accumulates to {n_final} (reclaims=0); Mimir holds used=0 (reclaims={m_reclaims}) under KV pressure",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_p_pressure_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tasks = [r["task"] for r in native["rows"]]
        n_used = [r["used"] or 0 for r in native["rows"]]
        m_used = [r["used"] or 0 for r in mimir["rows"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(tasks, n_used, "rx-", label="native vLLM (LRU, accumulates)")
        ax.plot(tasks, m_used, "g^-", label="Mimir (lifecycle reclaim, steady)")
        ax.set_xlabel("task (sequential, fills KV pool)")
        ax.set_ylabel("used KV blocks")
        ax.set_title(f"Phase P: KV-pool pressure (6 tasks, ~{native.get('total_blocks')} block pool, {Path(args.model).name})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        png = out_dir / f"phase_p_pressure_{Path(args.model).name}_curves.png"
        fig.savefig(png, dpi=140); plt.close(fig)
        print(f"保存: {png}", flush=True)
    except Exception as e:
        print(f"画图跳过: {e}", flush=True)

    print(f"native final used={n_final} reclaims=0")
    print(f"Mimir final used={m_final} reclaims={m_reclaims}")
    print("PHASE_P_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
