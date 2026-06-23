# ruff: noqa: E501
"""单 agent 多步 + 干扰批:Continuum TTL pin 的机制级 A/B(同步 InprocClient 可跑)。

**为什么有这个脚本(诚实定位)**:
三篇融合收益的"在线交错并发"基准(Continuum 论文用 vllm serve 异步 server)在我们
vLLM v1 **InprocClient 同步**设置下无法可信复现——llm.chat 走引擎锁串行化、无法
在单进程内"持续涌入请求"制造 agent 暂停时被挤掉 KV 的场景(多线程并发 llm.chat 会
触发 'Forward context is not set' 崩溃)。详见 docs/技术方案.md §3.7 诚实边界。

本脚本退而测**机制级**A/B:multi-step 单 agent + 步间干扰批。
  - agent step1 发 [TOOL: search(...)](强制 prompt)→ Continuum TTL pin 住其 KV
  - 步间提交干扰批(N 个异质长上下文请求,挤 KV 池,模拟"agent 暂停时其它请求抢显存")
  - agent step2:其前缀若被 TTL pin 保住 → 复用(cached>0);若被干扰顶掉 → 重算
A/B native(fcfs,无 TTL) vs mimir(+Continuum TTL),看 TTL 能否在干扰下保住跨步 KV。

**诚实实测结论(单卡 3090, Qwen3-4B, util=0.9, KV 池 88544 token)**:
  - TTL pin 机制正确工作:强制工具调用 prompt 下,max_pinned=1(mimir 侧),native=0。
  - 但干扰批难以顶掉 target 前缀:vLLM v1 APC 是 ref-counted LRU-tree,target 跨步
    共享前缀一直热(ref_cnt 不归零),短干扰批用完即释 → 干扰下 target step2 仍全命中。
    native 与 mimir 的 step2 TTFT / cached 几乎相同(均 ~20ms 全命中)——
    APC 已足够,TTL pin 无增量收益。
  - 根因:Continuum TTL 的真实收益需要**异步 server + 持续请求涌入**(agent 暂停数十秒、
    期间新请求持续填池把其 KV 挤掉),而我们 InprocClient 同步同步模式制造不出这个场景。
    这是**设置限制**,非机制缺陷——TTL pin 正确触发(冒烟 + 本脚本 max_pinned 验证)。

用法: python scripts/run_target_interferer.py <policy> <steps> <ctx> <interferer> <intf_ctx> <gap>
输出(stderr):TARGET_RESULT {...}  (含 per-step TTFT/cached/new_prefill/max_pinned)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimir.lmcache_compat import _fix_otel_logger_provider  # noqa: E402

_fix_otel_logger_provider()
from mimir.engine_vllm import EngineConfig  # noqa: E402
from mimir.engine_vllm_v1 import VLLMEngineV1  # noqa: E402

POLICY = sys.argv[1]
STEPS = int(sys.argv[2])
CTX = int(sys.argv[3])
INTF = int(sys.argv[4])
INTF_CTX = int(sys.argv[5])
GAP = float(sys.argv[6])

eng = VLLMEngineV1(
    EngineConfig(
        model="/data/models/Qwen3-4B-Instruct-2507",
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        max_model_len=8192,
        seed=42,
        extra={"scheduling_policy": POLICY, "max_model_len": 8192},
    ),
    device=0,
)
_ = eng.llm
sched = eng.mimir_scheduler()

from vllm import SamplingParams  # noqa: E402

FINAL_RE = re.compile(r"\[FINAL:\s*(.*?)\]", re.DOTALL)
TOOL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\(([^)]*)\)\s*\]")

agent_msgs = [
    {"role": "system", "content": "You MUST respond ONLY with [TOOL: search(\"query\")] to gather info. "
                                  "Do not explain. After tool results come back, write [FINAL: answer]."},
    {"role": "user", "content": "Background " + "lorem ipsum dolor sit amet. " * (CTX // 6 + 1) + "\nQ: summarize."},
]
sp_agent = SamplingParams(temperature=0.0, max_tokens=48, seed=42)
if POLICY == "mimir":
    sp_agent.extra_args = {"job_id": "agent_target"}
sp_intf = SamplingParams(temperature=0.0, max_tokens=4, seed=42)
intf_msgs = [
    [{"role": "user", "content": f"Doc-{j}: " + "quis nostrud exercitation ullamco. " * (INTF_CTX // 6 + 1) + f"\nReply ok{j}."}]
    for j in range(INTF)
]

records = []
max_pinned = 0
for step in range(STEPS):
    o = eng.llm.chat([agent_msgs], sp_agent, use_tqdm=False)[0]
    m = o.metrics
    np_tok = len(o.prompt_token_ids)
    nc_tok = getattr(o, "num_cached_tokens", 0) or 0
    ttft = getattr(m, "first_token_time", None)
    arr = getattr(m, "arrival_time", None)
    ttft_ms = ((ttft - arr) * 1000) if ttft and arr and ttft > arr else None
    records.append({"step": step, "ttft_ms": ttft_ms, "new_prefill": max(0, np_tok - nc_tok),
                    "prompt": np_tok, "cached": nc_tok})
    pinned_now = len(sched.pinned_requests)
    if pinned_now > max_pinned:
        max_pinned = pinned_now
    time.sleep(GAP)
    if step < STEPS - 1:
        _ = eng.llm.chat(intf_msgs, [sp_intf] * INTF, use_tqdm=False)
    text = o.outputs[0].text
    if FINAL_RE.search(text):
        agent_msgs.append({"role": "assistant", "content": text})
        break
    agent_msgs.append({"role": "assistant", "content": text})
    tm = TOOL_RE.search(text)
    agent_msgs.append({"role": "user", "content": "[TOOL_RESULT: data]" if tm else "Use tool or [FINAL: ans]."})

all_ttft = [r["ttft_ms"] for r in records if r["ttft_ms"] is not None]
all_np = [r["new_prefill"] for r in records]
all_c = [r["cached"] for r in records]
all_p = [r["prompt"] for r in records]
print("TARGET_RESULT " + json.dumps({
    "policy": POLICY, "steps": STEPS, "ctx": CTX, "interferer": INTF,
    "intf_ctx": INTF_CTX, "gap": GAP,
    "mean_ttft": round(sum(all_ttft) / len(all_ttft), 1) if all_ttft else None,
    "total_new_prefill": sum(all_np),
    "hit_ratio": round(sum(all_c) / sum(all_p), 3) if all_p else None,
    "max_pinned": max_pinned,
    "records": records,
}), file=sys.stderr, flush=True)
