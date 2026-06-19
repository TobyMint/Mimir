# ruff: noqa: E501
"""Phase O：Concurrent多 agent A/B（赛题「多Task推理」场景，引擎级验证）。

3 个 agent Task交替跑（A,B,C,A,B,C,...），单卡Concurrent：
- native：fcfs，不区分Task，所有 KV 累积
- Mimir：mimir 策略，每Task独立 task_id，请求完成自动Reclaim（Phase L）

度量：Peak used_blocks、累计 lifecycle_reclaims、是否稳态（Mimir 应保持平稳，native 累积）。
输出：benchmark_results/phase_o_concurrent_<model>.json + _curves.png

用法：python scripts/run_phase_o_concurrent.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.gpu import pick_least_busy_gpu

CHILD = r"""
import os, json, sys
sys.path.insert(0, os.getcwd())
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
model, gpu, util, mlen, mtok, policy = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
eng = VLLMEngineV1(EngineConfig(model=model, dtype="bfloat16", gpu_memory_utilization=util,
    enable_prefix_caching=True, max_model_len=mlen, extra={"scheduling_policy": policy}), device=0)
_ = eng.llm
AGENTS = {
    "A": ("You are agent A, a mathematician. Answer briefly.", "Solve: what is 17*23?"),
    "B": ("You are agent B, a historian. Answer briefly.", "When was the French Revolution?"),
    "C": ("You are agent C, a coder. Answer briefly.", "Write a Python one-liner to sum a list."),
}
# 交替 6 轮（A,B,C,A,B,C）
order = ["A","B","C","A","B","C"]
rows = []
for i, key in enumerate(order):
    sys_msg, q = AGENTS[key]
    tid = f"agent_{key}_{i}"  # 每轮唯一Task id（mimir 模式触发自动Reclaim）
    ctx = q + (" Consider KV cache memory implications. " * 4)
    eng.set_current_task(tid)
    eng.chat([{"role":"system","content":sys_msg},{"role":"user","content":ctx}], max_tokens=mtok)
    st = eng.mimir_stats()
    rows.append({"step":i+1,"agent":key,"task_id":tid,"used_blocks":st.get("used_blocks"),"lifecycle_reclaims":st.get("mimir_lifecycle_reclaims")})
print("RESULT_JSON:"+json.dumps(rows))
"""


def run_side(policy: str, g, args) -> list:
    import subprocess

    r = subprocess.run(
        [
            "python",
            "-c",
            CHILD,
            args.model,
            str(g.index),
            str(args.gpu_memory_util),
            str(args.max_model_len),
            str(args.max_tokens),
            policy,
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=400,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:") :])
    print("ERROR:", r.stderr[-300:], flush=True)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=14)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    print(f"GPU {g.index}", flush=True)

    print("\n=== native (fcfs, KV accumulates) ===", flush=True)
    native = run_side("fcfs", g, args)
    print("\n=== Mimir (per-task auto-reclaim) ===", flush=True)
    mimir = run_side("mimir", g, args)

    def peak(rows):
        return max((r.get("used_blocks") or 0) for r in rows) if rows else 0

    def final(rows):
        return (rows[-1].get("used_blocks") if rows else 0) or 0

    n_peak, m_peak = peak(native), peak(mimir)
    n_final, m_final = final(native), final(mimir)
    m_reclaims = mimir[-1].get("lifecycle_reclaims") if mimir else 0

    summary = {
        "model": Path(args.model).name,
        "scenario": "3 agents (A/B/C) interleaved 6 steps on one GPU",
        "native": {"peak_used_blocks": n_peak, "final_used_blocks": n_final, "rows": native},
        "mimir": {
            "peak_used_blocks": m_peak,
            "final_used_blocks": m_final,
            "lifecycle_reclaims": m_reclaims,
            "rows": mimir,
        },
        "headline": f"native peak={n_peak} vs Mimir peak={m_peak} (reclaims={m_reclaims})",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_o_concurrent_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in native]
        n_used = [r.get("used_blocks") or 0 for r in native]
        m_used = [r.get("used_blocks") or 0 for r in mimir]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(steps, n_used, "rx-", label="native vLLM (fcfs, accumulates)")
        ax.plot(steps, m_used, "g^-", label="Mimir (per-task auto-reclaim)")
        ax.set_xlabel("agent step (A,B,C,A,B,C)")
        ax.set_ylabel("used KV blocks")
        ax.set_title(
            f"Phase O: concurrent multi-agent (3 agents interleaved, {Path(args.model).name})"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        png = out_dir / f"phase_o_concurrent_{Path(args.model).name}_curves.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"保存: {png}", flush=True)
    except Exception as e:
        print(f"画图跳过: {e}", flush=True)

    print(f"native peak={n_peak} final={n_final}")
    print(f"Mimir peak={m_peak} final={m_final} reclaims={m_reclaims}")
    print("PHASE_O_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
