# ruff: noqa: E501
"""端到端 benchmark:native vs Continuum-pin vs SSC-reload,多 agent × 多轮。

设计(回应"不只看 step2,看整个 agent 程序跑完"):
  - N 个 agent 并发,每个跑 R 轮 ReAct(工具调用 → gap → 下一步)
  - tool_gap 有长有短(有的在 TTL 期内回来=pin 命中,有的超 TTL=KV 没了)
  - 三档:native(fcfs,无 pin 无 SSC) / pin(mimir+touch 保活) / SSC(fcfs+SharedStorageConnector)
  - 测:所有 agent 跑完的总时间、平均每步 TTFT、总重算 token、命中率
  - 异步 in-process(add_request+step 交错,保可观测)

负载:agent 每步 prompt 累积增长(真实 agent 上下文膨胀),步间注入干扰挤显存。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, sys, time, re, json, random
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
sys.path.insert(0, os.getcwd())
from mimir.lmcache_compat import _fix_otel_logger_provider
_fix_otel_logger_provider()

mode = sys.argv[2]  # native / pin / ssc
N_AGENTS = int(sys.argv[3])
ROUNDS = int(sys.argv[4])
CTX = int(sys.argv[5])
INTF_N = int(sys.argv[6])
INTF_CTX = int(sys.argv[7])
INTF_MAXTOK = int(sys.argv[8])
UTIL = float(sys.argv[9])
SEED = int(sys.argv[10])

from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1

extra = {"max_model_len": 8192}
if mode == "pin":
    extra["scheduling_policy"] = "mimir"
elif mode == "ssc":
    extra["scheduling_policy"] = "fcfs"
    store_path = "/dev/shm/ssc_e2e"
    os.makedirs(store_path, exist_ok=True)
    extra["kv_transfer_config"] = {
        "kv_connector": "SharedStorageConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"shared_storage_path": store_path},
    }
else:  # native
    extra["scheduling_policy"] = "fcfs"

eng = VLLMEngineV1(EngineConfig(
    model="/data/models/Qwen3-4B-Instruct-2507", dtype="bfloat16",
    gpu_memory_utilization=UTIL, enable_prefix_caching=True,
    max_model_len=8192, seed=SEED, extra=extra), device=0)
_ = eng.llm
le = eng.llm.llm_engine
tok = eng.llm.get_tokenizer()

from vllm import SamplingParams
from vllm.inputs import TokensPrompt

FINAL_RE = re.compile(r"\[FINAL:\s*(.*?)\]", re.DOTALL)
TOOL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\(([^)]*)\)\s*\]")
random.seed(SEED)

def _tok(msgs):
    return TokensPrompt(prompt_token_ids=tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True))

# N 个异质 agent(不同 system + 长 user 上下文)
topics = ["memory management", "kv cache", "scheduling", "compression",
          "offloading", "prefix reuse", "tool calling", "batching"]
agents = []
for i in range(N_AGENTS):
    topic = topics[i % len(topics)]
    filler = "lorem ipsum dolor sit amet, consectetur adipiscing elit. " * (CTX // 12 + 1)
    system = (f"You are agent-{i} researching {topic}. You MUST respond ONLY with "
              f"[TOOL: search(\"query\")]. After tool results, write [FINAL: answer].")
    agents.append({
        "id": i, "msgs": [{"role": "system", "content": system},
                          {"role": "user", "content": filler + f"\nQ: summarize {topic}."}],
        "done": False, "step": 0, "ttfts": [], "new_prefills": [], "cached": [],
        "job_id": f"agent_{i}",
    })

# 干扰请求池
INTF_DOCS = ["Doc-" + str(j) + ": " + " ".join(["quis nostrud exercitation. "] * (INTF_CTX // 6 + 1))
             for j in range(INTF_N * ROUNDS * N_AGENTS + 50)]
sp_intf = SamplingParams(temperature=0.0, max_tokens=INTF_MAXTOK, seed=SEED)
rid_c = [0]
def _rid(): rid_c[0] += 1; return f"intf_{rid_c[0]}"

def _step_some(n):
    for _ in range(n):
        if not le.has_unfinished_requests(): break
        for _o in le.step(): pass

def _drain_all(timeout=600):
    t0 = time.time()
    while le.has_unfinished_requests() and time.time() - t0 < timeout:
        for _o in le.step(): pass

t_start = time.time()

for rnd in range(ROUNDS):
    # 每轮:所有未完成 agent 各提交一步(交错 add_request),然后 step 循环跑完
    pending = [(i, a) for i, a in enumerate(agents) if not a["done"]]
    if not pending:
        break
    for i, a in pending:
        sp = SamplingParams(temperature=0.0, max_tokens=48, seed=SEED)
        if mode == "pin":
            sp.extra_args = {"job_id": a["job_id"]}
        rid = f"agent_{i}_r{rnd}"
        try:
            le.abort_request(rid)
        except Exception:
            pass
        le.add_request(rid, _tok(a["msgs"]), sp, arrival_time=time.time())
    # step 循环直到所有本轮 agent 请求完成
    pending_rids = {f"agent_{i}_r{rnd}" for i, a in pending}
    t0 = time.time()
    outs = {}
    while le.has_unfinished_requests() and time.time() - t0 < 300:
        for o in le.step():
            if o.finished and o.request_id in pending_rids:
                outs[o.request_id] = o
        if len(outs) >= len(pending_rids):
            break
    # 处理输出
    for i, a in pending:
        rid = f"agent_{i}_r{rnd}"
        o = outs.get(rid)
        if o is None:
            a["done"] = True
            continue
        m = o.metrics
        np_tok = len(o.prompt_token_ids)
        nc_tok = getattr(o, "num_cached_tokens", 0) or 0
        ttft = getattr(m, "first_token_time", None)
        arr = getattr(m, "arrival_time", None)
        ttft_ms = ((ttft - arr) * 1000) if ttft and arr and ttft > arr else None
        a["ttfts"].append(ttft_ms)
        a["new_prefills"].append(max(0, np_tok - nc_tok))
        a["cached"].append(nc_tok)
        a["step"] += 1
        text = o.outputs[0].text
        if FINAL_RE.search(text):
            a["done"] = True
            a["msgs"].append({"role": "assistant", "content": text})
        else:
            a["msgs"].append({"role": "assistant", "content": text})
            tm = TOOL_RE.search(text)
            a["msgs"].append({"role": "user", "content": "[TOOL_RESULT: data]" if tm else "Use tool or [FINAL: ans]."})
    # 步间干扰挤显存(交错制造争用)
    if rnd < ROUNDS - 1:
        for j in range(INTF_N):
            doc = INTF_DOCS[(rnd * INTF_N + j) % len(INTF_DOCS)]
            le.add_request(_rid(), _tok([{"role": "user", "content": doc}]), sp_intf, arrival_time=time.time())
        _step_some(INTF_N * 3)
        _drain_all()

t_total = time.time() - t_start

# 汇总
all_ttft = [t for a in agents for t in a["ttfts"] if t is not None]
all_np = [p for a in agents for p in a["new_prefills"]]
all_c = [c for a in agents for c in a["cached"]]
all_p = [np_tok for a in agents for np_tok in [max(0, p) for p in a["new_prefills"]]]
done_n = sum(1 for a in agents if a["done"])
total_prompt = sum(a["new_prefills"][s] + a["cached"][s] for a in agents for s in range(len(a["new_prefills"])) if s < len(a["cached"]))
summary = {
    "mode": mode, "n_agents": N_AGENTS, "rounds": ROUNDS, "util": UTIL,
    "total_time_s": round(t_total, 1),
    "mean_ttft_ms": round(sum(all_ttft) / len(all_ttft), 1) if all_ttft else None,
    "median_ttft_ms": round(sorted(all_ttft)[len(all_ttft)//2], 1) if all_ttft else None,
    "p90_ttft_ms": round(sorted(all_ttft)[int(len(all_ttft)*0.9)] if all_ttft else 0, 1),
    "total_new_prefill_tokens": sum(all_np),
    "total_cached_tokens": sum(all_c),
    "hit_ratio": round(sum(all_c) / (sum(all_c) + sum(all_np)), 3) if (sum(all_c) + sum(all_np)) > 0 else None,
    "n_steps_total": len(all_ttft),
    "n_agents_done": done_n,
    "agents_detail": [{"id": a["id"], "done": a["done"], "steps": a["step"],
                       "mean_ttft": round(sum(a["ttfts"])/len(a["ttfts"]), 1) if a["ttfts"] else None,
                       "total_new_prefill": sum(a["new_prefills"])} for a in agents],
}
print("E2E_RESULT " + json.dumps(summary))
"""


def run_side(gpu, mode, n_agents, rounds, ctx, intf_n, intf_ctx, intf_maxtok, util, seed):
    r = subprocess.run(
        ["python", "-c", CHILD, str(gpu), mode, str(n_agents), str(rounds),
         str(ctx), str(intf_n), str(intf_ctx), str(intf_maxtok), str(util), str(seed)],
        capture_output=True, text=True, env=dict(os.environ), timeout=3600)
    # CHILD print "E2E_RESULT {...}" (空格分隔,无冒号),检查 stdout 和 stderr
    for stream in (r.stdout, r.stderr):
        for line in stream.splitlines():
            if line.startswith("E2E_RESULT"):
                # 去掉前缀 "E2E_RESULT " 后解析 JSON
                return json.loads(line[len("E2E_RESULT"):].strip())
    sys.stderr.write(f"[{mode}] no E2E_RESULT (rc={r.returncode}):\n{r.stderr[-1500:]}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=2000)
    ap.add_argument("--intf-n", type=int, default=12)
    ap.add_argument("--intf-ctx", type=int, default=3000)
    ap.add_argument("--intf-maxtok", type=int, default=512)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="native,pin,ssc")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    print(f"GPU {args.gpu}, {args.agents} agents × {args.rounds} rounds, util={args.util}", flush=True)
    print(f"ctx={args.ctx}, intf={args.intf_n}×{args.intf_ctx}+{args.intf_maxtok}", flush=True)

    summary = {"load": vars(args), "results": {}}
    for mode in [m.strip() for m in args.modes.split(",")]:
        print(f"\n=== {mode} ===", flush=True)
        # 清 SSC store(ssc 模式)
        if mode == "ssc":
            subprocess.run(["rm", "-rf", "/dev/shm/ssc_e2e"], check=False)
            os.makedirs("/dev/shm/ssc_e2e", exist_ok=True)
        res = run_side(args.gpu, mode, args.agents, args.rounds, args.ctx,
                       args.intf_n, args.intf_ctx, args.intf_maxtok, args.util, args.seed)
        if res is None:
            print(f"  {mode}: FAILED", flush=True)
            continue
        print(f"  total={res['total_time_s']}s mean_ttft={res['mean_ttft_ms']}ms "
              f"hit={res['hit_ratio']} prefill={res['total_new_prefill_tokens']} "
              f"done={res['n_agents_done']}/{res['n_agents']}", flush=True)
        summary["results"][mode] = res

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    path = out / "e2e_three_tier.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
