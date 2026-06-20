# ruff: noqa: E501, E701, E702
"""真并发压测：同一张榨干的卡(util=0.9),并发数递增,测 native 退化点 vs Mimir 不退化。

与旧 Phase O/P(假并发:agent 交替顺序跑)的区别——这里用 vLLM ``llm.chat([msgs1..msgsN])``
一次把 N 个请求交给引擎同 batch 真并发处理,测真实的显存压力。

并发数 N = 1→2→4→8→16→32 递增,每个 N 测:
- peak used_blocks(批量提交前后的块占用峰值)
- 是否退化:used_blocks 逼近/超过 KV 池上限 → 引擎被迫 LRU 淘汰活跃块
- 是否 OOM:有请求因显存不足失败

native(fcfs):N 增大,KV 累积,在某 N 撞池子上限 → 退化(淘汰活跃块)→ 再大 OOM。
Mimir(mimir 策略 + offload):任务完成自动回收 + 工具结果外置,used 稳态低,更高 N 仍不退化。

退化判定(主指标):peak used_blocks ≥ 池子上限的 90% 记为"退化";有请求失败记为"OOM"。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.gpu import pick_least_busy_gpu  # noqa: E402

CHILD = r"""
import os, json, sys
sys.path.insert(0, os.getcwd())
model, gpu, util, mlen, mtok, policy, N = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6], int(sys.argv[7]))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
eng = VLLMEngineV1(EngineConfig(
    model=model, dtype="bfloat16", gpu_memory_utilization=util,
    enable_prefix_caching=True, max_model_len=mlen,
    extra={"scheduling_policy": policy}), device=0)
_ = eng.llm
bp = eng.mimir_block_pool()
total_blocks = bp.num_gpu_blocks

# 构造 N 个异质 agent 请求：不同 system + 含工具结果的长上下文，模拟真实并发 agent
SYS = [
    "You are agent {}, a research analyst. Answer briefly about your topic.",
    "You are agent {}, a code reviewer. Answer briefly about your topic.",
    "You are agent {}, a data scientist. Answer briefly about your topic.",
    "You are agent {}, a technical writer. Answer briefly about your topic.",
]
# 每个请求带一段较长的工具结果,撑大单请求 KV(模拟 agent 累积上下文)。
# ~3500 token/请求(≈220 块),池子 5534 块下 native 在 N≈20 撞墙——合理压测区间。
TOOL_PAYLOAD = "[TOOL_RESULT search]\n" + " ".join(
    f"distinct_fact_{i} value_{i} payload detail." for i in range(320))

msgs_list = []
task_ids = []
for i in range(N):
    sys_msg = SYS[i % len(SYS)].format(i)
    user = f"Agent {i} task: summarize the key points from the tool result.\n{TOOL_PAYLOAD}"
    msgs_list.append([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}])
    task_ids.append(f"agent_{i}")

pre_used = eng.mimir_stats().get("used_blocks", 0) or 0
degraded = False
oom = False
err = ""
try:
    outs = eng.chat_batch(msgs_list, max_tokens=mtok, task_ids=task_ids)
    n_ok = sum(1 for o in outs if o and o.outputs)
except Exception as e:
    n_ok = 0
    err = str(e)[:200]
    msg_lower = err.lower()
    if "out of memory" in msg_lower or "oom" in msg_lower or "no available memory" in msg_lower:
        oom = True
    else:
        degraded = True

st = eng.mimir_stats()
post_used = st.get("used_blocks", 0) or 0
peak_used = max(pre_used, post_used)
# 退化判定：峰值达池子 90% 或有请求失败但未 OOM
THRESH = int(total_blocks * 0.9)
if not oom and not degraded:
    if peak_used >= THRESH or n_ok < N:
        degraded = True
reclaims = st.get("mimir_lifecycle_reclaims", 0)

print("RESULT_JSON:" + json.dumps({
    "policy": policy, "N": N, "total_blocks": total_blocks,
    "pre_used": pre_used, "post_used": post_used, "peak_used": peak_used,
    "n_ok": n_ok, "degraded": degraded, "oom": oom, "error": err,
    "lifecycle_reclaims": reclaims,
}))
"""


def run_side(model, g, util, mlen, mtok, policy, N):
    r = subprocess.run(
        ["python", "-c", CHILD, model, str(g.index), str(util), str(mlen),
         str(mtok), policy, str(N)],
        capture_output=True, text=True, env=dict(os.environ), timeout=600,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    return {"policy": policy, "N": N, "oom": True, "degraded": True,
            "n_ok": 0, "peak_used": None, "error": (r.stderr[-300:] or "no RESULT_JSON").replace("\r", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32", help="并发数序列")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU"); return 2
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB, util={args.gpu_memory_util}, max_model_len={args.max_model_len}", flush=True)
    Ns = [int(x) for x in args.concurrency.split(",")]
    print(f"并发序列: {Ns}", flush=True)

    results = {"native": [], "mimir": []}
    for policy in ["fcfs", "mimir"]:
        label = "mimir" if policy == "mimir" else "native"
        print(f"\n=== {label} ({policy}) ===", flush=True)
        for N in Ns:
            r = run_side(args.model, g, args.gpu_memory_util, args.max_model_len,
                         args.max_tokens, policy, N)
            results[label].append(r)
            tag = "OOM" if r.get("oom") else ("DEGRADED" if r.get("degraded") else "OK")
            print(f"  N={N:>3}: peak_used={r.get('peak_used')!s} n_ok={r.get('n_ok')}/{N} "
                  f"reclaims={r.get('lifecycle_reclaims',0)} [{tag}]", flush=True)

    # 找退化点：第一个 degraded 或 oom 的 N
    def degrade_point(rows):
        for r in rows:
            if r.get("oom") or r.get("degraded"):
                return r.get("N")
        return None
    n_deg = degrade_point(results["native"])
    m_deg = degrade_point(results["mimir"])
    # total_blocks 从任一成功的行取
    total_blocks = next((r["total_blocks"] for r in results["native"] + results["mimir"]
                         if r.get("total_blocks")), None)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    png_name = out / f"concurrent_press_{Path(args.model).name}_curves.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for label, color, marker in [("native", "r", "x"), ("mimir", "g", "^")]:
            xs = [r["N"] for r in results[label] if r.get("peak_used") is not None]
            ys = [r["peak_used"] for r in results[label] if r.get("peak_used") is not None]
            ax.plot(xs, ys, f"{color}{marker}-", label=f"{label} (fcfs)" if label == "native" else "Mimir (reclaim+offload)")
        if total_blocks:
            ax.axhline(total_blocks, color="gray", linestyle="--", alpha=0.6, label=f"KV pool cap ({total_blocks})")
            ax.axhline(int(total_blocks * 0.9), color="orange", linestyle=":", alpha=0.5, label="90% degrade threshold")
        if n_deg:
            ax.axvline(n_deg, color="red", linestyle=":", alpha=0.4)
            ax.text(n_deg, 0.5, f"native degrades\n@N={n_deg}", color="red", fontsize=7,
                    va="bottom", transform=ax.get_xaxis_transform())
        ax.set_xlabel("concurrent agents (N)")
        ax.set_ylabel("peak used KV blocks")
        ax.set_title(f"Concurrent pressure (util={args.gpu_memory_util}, {Path(args.model).name})")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(png_name, dpi=140)
        plt.close(fig)
        print(f"\nPNG: {png_name}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)

    summary = {
        "model": Path(args.model).name, "util": args.gpu_memory_util,
        "max_model_len": args.max_model_len, "total_blocks": total_blocks,
        "native_degrade_at": n_deg, "mimir_degrade_at": m_deg,
        "native": results["native"], "mimir": results["mimir"],
        "headline": (f"native 退化点 N={n_deg}, Mimir 退化点 N={m_deg}"
                     f"{'（Mimir 在更高并发仍不退化）' if m_deg is None or (n_deg and m_deg and m_deg > n_deg) else ''}"),
    }
    jp = out / f"concurrent_press_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n退化点: native@N={n_deg}  Mimir@N={m_deg}")
    print(f"JSON: {jp}")
    print("CONCURRENT_PRESS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
