# ruff: noqa: E501
"""并发多轮 agent A/B benchmark(批量同步版)。

**诚实定位(重要)**:三篇融合的"在线交错并发"收益(Continuum 论文用 vllm serve 异步
server 才能体现,见 §3.7)在 vLLM v1 **InprocClient 同步**设置下无法可信复现:
  - llm.chat 走引擎锁串行化,单进程内多线程并发 llm.chat 会触发
    'Forward context is not set' 崩溃,无法做在线交错。
  - 同步批量(本脚本:N agent 同 batch 进出)gap 期间全部暂停、无"持续涌入"——
    TTL pin 在冒烟(单步工具调用→pin→末步释放)与 run_target_interferer.py
    (max_pinned 验证)里证明机制正确,但端到端收益需要异步 server 才能体现。
详情见 docs/技术方案.md §3.7 诚实边界。

本脚本仍可用于:批量并发下的 TTFT / cache-hit / 重 prefill 基线测量(N agent 同
batch prefill/decode,显存争用下 vLLM preemption/recompute 量化),作为 native
vs mimir(TTL)的**机制触发验证**(看 max_pinned 是否 >0),但**非收益证据**。

A/B 四侧(消融梯度):
  native      : fcfs,无 LMCache,无 TTL(原版 vLLM)
  offload     : fcfs + LMCache CPU offload
  ttl         : mimir(Continuum TTL),无 LMCache
  full        : mimir + LMCache(三篇融合全开;CacheGen serde 经 LMCache 配置)

指标:per-step TTFT、new_prefill(prompt-cached)、cache hit ratio、成功率。
输出: benchmark_results/concurrent_fusion_<model>.json

用法: python scripts/run_concurrent_fusion.py [--agents 8] [--rounds 4]
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

# CHILD:子进程隔离每侧(保显存释放)。两侧同 model/util/seed/负载,只差 policy/lmcache。
CHILD = r"""
import os, json, sys, time
sys.path.insert(0, os.getcwd())
(model, gpu, util, mlen, mtok, policy, lmcache_on,
 agents, rounds, base_ctx, tool_gap, seed) = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
    sys.argv[6], sys.argv[7] == "1", int(sys.argv[8]), int(sys.argv[9]),
    int(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12]))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from mimir.lmcache_compat import ensure_lmcache  # 修 otel(幂等)

extra = {"scheduling_policy": policy, "max_model_len": mlen}
if lmcache_on:
    extra["lmcache"] = True
eng = VLLMEngineV1(EngineConfig(
    model=model, dtype="bfloat16", gpu_memory_utilization=util,
    enable_prefix_caching=True, max_model_len=mlen, seed=seed, extra=extra), device=0)
_ = eng.llm  # force init

# N 个异质 agent:不同 system + 不同初始长上下文(填充 base_ctx token,压显存)
import random
random.seed(seed)
TOOL_RE = None
def _make_agents(n, base_ctx):
    import re
    global TOOL_RE
    TOOL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\(([^)]*)\)\s*\]")
    FINAL_RE = re.compile(r"\[FINAL:\s*(.*?)\]", re.DOTALL)
    agents = []
    topics = ["memory management", "kv cache", "scheduling", "compression",
              "offloading", "prefix reuse", "tool calling", "batching",
              "throughput", "latency", "gpu memory", "inference engine"]
    for i in range(n):
        topic = topics[i % len(topics)]
        filler = (f"Background context about {topic}: " + "lorem ipsum dolor sit amet, consectetur adipiscing elit. " * (base_ctx // 12 + 1))
        system = (f"You are agent-{i} researching {topic}. Use the search tool as "
                  f"[TOOL: search('query')]. After each tool, continue. "
                  f"When you have enough, write [FINAL: your answer]. Keep responses short.")
        agents.append({"messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": filler + f"\nQuestion: summarize {topic}."}],
                       "done": False, "steps": 0, "ttfts": [], "new_prefills": [],
                       "cached_tokens": [], "prompt_tokens": [], "crashed": False})
    return agents

agents = _make_agents(agents, base_ctx)

from vllm import SamplingParams
sp = SamplingParams(temperature=0.0, max_tokens=mtok, seed=seed)

for rnd in range(rounds):
    # 收本轮未完成 agent 的请求,真批量提交
    pending = [(i, a) for i, a in enumerate(agents) if not a["done"]]
    if not pending:
        break
    msgs_list = [a["messages"] for _, a in pending]
    # 真·批量并发:一次 chat 提交所有未完成 agent(per-request SamplingParams 带 job_id)
    sp_list = []
    for i, _ in pending:
        s = SamplingParams(temperature=0.0, max_tokens=mtok, seed=seed)
        if policy == "mimir":
            s.extra_args = {"job_id": f"agent_{i}"}  # Continuum TTL:同 agent 跨步 KV 复用
        sp_list.append(s)
    try:
        outs_all = eng.llm.chat(msgs_list, sp_list, use_tqdm=False)
    except Exception as e:
        sys.stderr.write(f"round {rnd} batch crashed: {e!r}\n")
        for i, a in pending:
            a["crashed"] = True; a["done"] = True
        break
    outs = [(i, a, o) for (i, a), o in zip(pending, outs_all)]
    # 处理输出:解析 tool/final、记录指标、喂回工具结果、模拟 tool gap
    for i, a, o in outs:
        m = o.metrics
        # num_prompt_tokens: v1 不在 metrics 暴露,用 prompt_token_ids 长度
        np_tok = len(o.prompt_token_ids) if getattr(o, "prompt_token_ids", None) else None
        # num_cached_tokens: v1 报 0 不可靠(见 smoke),以 TTFT 首步 vs 后续差为复用信号
        nc_tok = getattr(o, "num_cached_tokens", 0) or 0
        ttft = getattr(m, "first_token_time", None) if m else None
        arrival = getattr(m, "arrival_time", None) if m else None
        ttft_ms = None
        if ttft is not None and arrival is not None and ttft > arrival:
            ttft_ms = (ttft - arrival) * 1000.0
        new_prefill = (max(0, np_tok - nc_tok) if np_tok is not None else None)
        a["ttfts"].append(ttft_ms)
        a["new_prefills"].append(new_prefill)
        a["cached_tokens"].append(nc_tok)
        a["prompt_tokens"].append(np_tok)
        a["steps"] += 1
        text = o.outputs[0].text
        fm = __import__("re").search(r"\[FINAL:\s*(.*?)\]", text, __import__("re").DOTALL)
        if fm:
            a["done"] = True
            a["messages"].append({"role": "assistant", "content": text})
        else:
            tm = TOOL_RE.search(text)
            a["messages"].append({"role": "assistant", "content": text})
            if tm:
                tool_name = tm.group(1)
                # mock 工具返回(小结果,不膨胀上下文——膨胀靠累积 user/system)
                result = f"[TOOL_RESULT: {tool_name} returned data about the query]"
                a["messages"].append({"role": "user", "content": result})
            else:
                a["messages"].append({"role": "user", "content": "Use a tool or give [FINAL: answer]."})
    # 模拟 tool gap(长尾):每个未完成 agent 暂停 tool_gap 秒,期间 KV 被 TTL pin(若开 TTL)
    time.sleep(tool_gap)

# 汇总
def _agg(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs)/len(xs)) if xs else None
summary = {
    "n_agents": len(agents),
    "rounds": rounds,
    "policy": policy,
    "lmcache": lmcache_on,
    "agents": [{
        "id": i, "done": a["done"], "crashed": a["crashed"], "steps": a["steps"],
        "mean_ttft_ms": _agg(a["ttfts"]),
        "total_new_prefill": sum(x for x in a["new_prefills"] if x is not None),
        "mean_new_prefill": _agg(a["new_prefills"]),
        "mean_cached": _agg(a["cached_tokens"]),
        "mean_prompt": _agg(a["prompt_tokens"]),
        "ttfts": a["ttfts"], "new_prefills": a["new_prefills"],
    } for i, a in enumerate(agents)],
}
done_n = sum(1 for a in agents if a["done"] and not a["crashed"])
summary["success_rate"] = done_n / len(agents)
all_ttft = [t for a in agents for t in a["ttfts"] if t is not None]
summary["overall_mean_ttft_ms"] = (sum(all_ttft)/len(all_ttft)) if all_ttft else None
all_prefill = [p for a in agents for p in a["new_prefills"] if p is not None]
summary["overall_total_new_prefill"] = sum(all_prefill)
all_cached = [c for a in agents for c in a["cached_tokens"] if c is not None]
all_prompt = [p for a in agents for p in a["prompt_tokens"] if p is not None]
summary["overall_hit_ratio"] = (sum(all_cached)/sum(all_prompt)) if all_prompt and sum(all_prompt)>0 else None
print("RESULT_JSON:" + json.dumps(summary))
"""


def run_side(model, gpu, util, mlen, mtok, policy, lmcache, n_agents, rounds,
             base_ctx, tool_gap, seed):
    r = subprocess.run(
        ["python", "-c", CHILD, model, str(gpu), str(util), str(mlen), str(mtok),
         policy, "1" if lmcache else "0", str(n_agents), str(rounds),
         str(base_ctx), str(tool_gap), str(seed)],
        capture_output=True, text=True, env=dict(os.environ), timeout=1200,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    sys.stderr.write(
        f"[{policy}/lmcache={lmcache}] no RESULT_JSON (rc={r.returncode}):\n"
        + r.stderr[-1200:].replace("\r", "") + "\n")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--base-ctx", type=int, default=1600,
                    help="每 agent 初始 user 上下文填充 token 数(压显存)")
    ap.add_argument("--tool-gap", type=float, default=1.0,
                    help="每轮工具调用 gap 秒(模拟长尾 tool time,触发 TTL pin)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="benchmark_results")
    ap.add_argument("--sides", default="native,offload,ttl,full",
                    help="要跑的侧,逗号分隔")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)
    print(f"负载: {args.agents} agents × {args.rounds} rounds, base_ctx={args.base_ctx}, "
          f"tool_gap={args.tool_gap}s, max_model_len={args.max_model_len}", flush=True)

    side_map = {
        "native":  ("fcfs",  False),
        "offload": ("fcfs",  True),
        "ttl":     ("mimir", False),
        "full":    ("mimir", True),
    }
    sides = [s for s in args.sides.split(",") if s.strip()]
    summary = {"model": Path(args.model).name, "load": {
        "agents": args.agents, "rounds": args.rounds, "base_ctx": args.base_ctx,
        "tool_gap": args.tool_gap, "max_model_len": args.max_model_len},
        "sides": {}}
    for s in sides:
        policy, lmcache = side_map[s]
        print(f"\n=== {s} (policy={policy}, lmcache={lmcache}) ===", flush=True)
        res = run_side(args.model, g.index, args.gpu_memory_util, args.max_model_len,
                       args.max_tokens, policy, lmcache, args.agents, args.rounds,
                       args.base_ctx, args.tool_gap, args.seed)
        if res is None:
            print(f"  {s}: FAILED", flush=True)
            continue
        print(f"  mean_ttft={res.get('overall_mean_ttft_ms')}, "
              f"hit_ratio={res.get('overall_hit_ratio')}, "
              f"success={res.get('success_rate')}, "
              f"total_new_prefill={res.get('overall_total_new_prefill')}", flush=True)
        summary["sides"][s] = res

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    tag = Path(args.model).name
    json_path = out / f"concurrent_fusion_{tag}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON → {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
