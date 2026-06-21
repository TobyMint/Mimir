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
import os, json, sys, time
sys.path.insert(0, os.getcwd())
# args: model gpu util mlen mtok policy N rounds
model, gpu, util, mlen, mtok, policy, N, rounds = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6], int(sys.argv[7]), int(sys.argv[8]))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from benchmarks.harness import _req_metrics
eng = VLLMEngineV1(EngineConfig(
    model=model, dtype="bfloat16", gpu_memory_utilization=util,
    enable_prefix_caching=True, max_model_len=mlen,
    extra={"scheduling_policy": policy}), device=0)
_ = eng.llm
bp = eng.mimir_block_pool()
total_blocks = bp.num_gpu_blocks

SYS = [
    "You are agent {}, a research analyst. Answer briefly about your topic.",
    "You are agent {}, a code reviewer. Answer briefly about your topic.",
    "You are agent {}, a data scientist. Answer briefly about your topic.",
    "You are agent {}, a technical writer. Answer briefly about your topic.",
]
# 每轮追加的工具结果（模拟 agent 每轮调用工具拿新结果，上下文跨轮累积）
def tool_payload(k):
    return "[TOOL_RESULT search]\n" + " ".join(
        f"turn{k}_fact_{i} value_{i} payload detail." for i in range(120))

# 每个 agent 维护自己的累积上下文（跨轮增长，模拟真实 agent）
histories = []
for i in range(N):
    sys_msg = SYS[i % len(SYS)].format(i)
    histories.append([{"role": "system", "content": sys_msg}])

peak_used = 0
all_ttfts = []
oom = False
err = ""
n_ok_total = 0
t_start = time.perf_counter()
try:
    for k in range(rounds):
        # 本轮：N 个 agent 各自把"新 user 消息(含工具结果)"加进自己的上下文，一起批量提交
        msgs_list = []
        task_ids = []
        for i in range(N):
            user = f"Agent {i} turn {k}: summarize the tool result.\n{tool_payload(k)}"
            # 注意：histories[i] 是累积的（跨轮），这里复制一份加本轮 user，不污染原历史用于下一轮拼接
            msgs = list(histories[i]) + [{"role": "user", "content": user}]
            msgs_list.append(msgs)
            task_ids.append(f"agent_{i}_turn_{k}")
        # 峰值采样：批量提交后立即读（此刻 KV 占用最高）
        outs = eng.chat_batch(msgs_list, max_tokens=mtok, task_ids=task_ids)
        st = eng.mimir_stats()
        cur_used = st.get("used_blocks", 0) or 0
        peak_used = max(peak_used, cur_used)
        # 收 TTFT + 把本轮输出加回各 agent 历史（native 侧累积，mimir 侧回收后下一轮也重填）
        for i, o in enumerate(outs):
            if o and o.outputs:
                n_ok_total += 1
                txt = o.outputs[0].text
                histories[i].append({"role": "user", "content": f"turn {k}: summarize tool result.\n{tool_payload(k)}"})
                histories[i].append({"role": "assistant", "content": txt})
                ttft = _req_metrics(o).get("ttft_ms")
                if ttft is not None:
                    all_ttfts.append(ttft)
    wall = time.perf_counter() - t_start
except Exception as e:
    wall = time.perf_counter() - t_start
    err = str(e)[:200]
    msg_lower = err.lower()
    if "out of memory" in msg_lower or "oom" in msg_lower or "no available memory" in msg_lower:
        oom = True

reclaims = eng.mimir_stats().get("mimir_lifecycle_reclaims", 0)
expected = N * rounds
avg_ttft = round(sum(all_ttfts)/len(all_ttfts), 1) if all_ttfts else None
max_ttft = round(max(all_ttfts), 1) if all_ttfts else None
throughput = round(n_ok_total / wall, 2) if wall > 0 else None
svc_fail = (n_ok_total < expected) or oom
svc_degraded = (not svc_fail) and max_ttft is not None and max_ttft > 5000

print("RESULT_JSON:" + json.dumps({
    "policy": policy, "N": N, "rounds": rounds, "total_blocks": total_blocks,
    "peak_used": peak_used, "n_ok": n_ok_total, "expected": expected,
    "svc_fail": svc_fail, "svc_degraded": svc_degraded, "oom": oom,
    "error": err, "wall_s": round(wall, 2),
    "avg_ttft_ms": avg_ttft, "max_ttft_ms": max_ttft, "throughput_req_s": throughput,
    "lifecycle_reclaims": reclaims,
}))
"""


def run_side(model, g, util, mlen, mtok, policy, N, rounds):
    r = subprocess.run(
        ["python", "-c", CHILD, model, str(g.index), str(util), str(mlen),
         str(mtok), policy, str(N), str(rounds)],
        capture_output=True, text=True, env=dict(os.environ), timeout=600,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    return {"policy": policy, "N": N, "oom": True, "svc_fail": True, "svc_degraded": True,
            "n_ok": 0, "peak_used": None, "error": (r.stderr[-300:] or "no RESULT_JSON").replace("\r", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", default="2,4,8,16", help="并发数序列")
    ap.add_argument("--rounds", type=int, default=5, help="每个 agent 跑的轮数（跨轮累积，Mimir 轮间回收）")
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
                         args.max_tokens, policy, N, args.rounds)
            results[label].append(r)
            tag = "FAIL" if r.get("svc_fail") else ("SLOW" if r.get("svc_degraded") else "OK")
            print(f"  N={N:>3}: n_ok={r.get('n_ok')}/{r.get('expected',N)} avg_ttft={r.get('avg_ttft_ms')!s}ms "
                  f"wall={r.get('wall_s')}s peak={r.get('peak_used')!s} reclaims={r.get('lifecycle_reclaims',0)} [{tag}]", flush=True)

    # 服务维度的退化/失败点（用服务指标说话，不靠内部 used_blocks）
    def fail_point(rows):
        for r in rows:
            if r.get("svc_fail") or r.get("oom"):
                return r.get("N")
        return None
    def slow_point(rows):
        for r in rows:
            if r.get("svc_degraded"):
                return r.get("N")
        return None
    n_fail, m_fail = fail_point(results["native"]), fail_point(results["mimir"])
    n_slow, m_slow = slow_point(results["native"]), slow_point(results["mimir"])
    total_blocks = next((r["total_blocks"] for r in results["native"] + results["mimir"]
                         if r.get("total_blocks")), None)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    png_name = out / f"concurrent_press_{Path(args.model).name}_curves.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
        # 上：n_ok（服务完成数）—— 失败就看这条掉
        for label, color, marker in [("native", "r", "x"), ("mimir", "g", "^")]:
            xs = [r["N"] for r in results[label]]
            ys = [r.get("n_ok", 0) for r in results[label]]
            ax1.plot(xs, ys, f"{color}{marker}-", label="native (fcfs)" if label == "native" else "Mimir")
        ax1.set_ylabel("completed requests (n_ok)")
        ax1.set_title(f"Concurrent pressure — service metrics (util={args.gpu_memory_util}, {Path(args.model).name})")
        ax1.legend(fontsize=8, loc="upper left")
        # 下：avg TTFT（服务延迟）—— 退化看这条涨
        for label, color, marker in [("native", "r", "x"), ("mimir", "g", "^")]:
            xs = [r["N"] for r in results[label] if r.get("avg_ttft_ms") is not None]
            ys = [r["avg_ttft_ms"] for r in results[label] if r.get("avg_ttft_ms") is not None]
            ax2.plot(xs, ys, f"{color}{marker}-", label="native (fcfs)" if label == "native" else "Mimir")
        ax2.set_xlabel("concurrent agents (N)")
        ax2.set_ylabel("avg TTFT (ms)")
        ax2.set_yscale("log")
        ax2.legend(fontsize=8, loc="upper left")
        if n_fail:
            ax1.axvline(n_fail, color="red", linestyle=":", alpha=0.5)
            ax1.text(n_fail, 0.3, f"native fails\n@N={n_fail}", color="red", fontsize=7,
                     va="bottom", transform=ax1.get_xaxis_transform())
        fig.tight_layout()
        fig.savefig(png_name, dpi=140)
        plt.close(fig)
        print(f"\nPNG: {png_name}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)

    summary = {
        "model": Path(args.model).name, "util": args.gpu_memory_util,
        "max_model_len": args.max_model_len, "total_blocks": total_blocks,
        "native_fail_at": n_fail, "mimir_fail_at": m_fail,
        "native_slow_at": n_slow, "mimir_slow_at": m_slow,
        "native": results["native"], "mimir": results["mimir"],
        "headline": (f"native 服务失败点 N={n_fail}（退化点 N={n_slow}）；"
                     f"Mimir 失败点 N={m_fail}（退化点 N={m_slow}）"),
    }
    jp = out / f"concurrent_press_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n服务失败点: native@N={n_fail}  Mimir@N={m_fail}")
    print(f"服务退化点: native@N={n_slow}  Mimir@N={m_slow}")
    print(f"JSON: {jp}")
    print("CONCURRENT_PRESS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
