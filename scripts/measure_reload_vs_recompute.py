# ruff: noqa: E501, I001
"""测量 pin+SSC 核心假设:reload(PCIe 传输 N token KV) vs 重算(prefill N token)。

pin+SSC 卖点:KV 丢时用 PCIe 传输(reload)代替重算(prefill)。若 reload << 重算,
pin+SSC 在 KV 丢场景快于 native。当前 total 984s 是 SSC store 每轮写盘慢,不是 reload 慢。
测不同 N + GPU 压力大时(干扰占 GPU),拆开看 reload vs 重算。
"""
import os
import sys
import time

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "3"
sys.path.insert(0, os.getcwd())
from mimir.lmcache_compat import _fix_otel_logger_provider  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

_fix_otel_logger_provider()

model = "/data/models/Qwen3-4B-Instruct-2507"
cfg = AutoConfig.from_pretrained(model)
nl = cfg.num_hidden_layers
nkh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
hd = cfg.hidden_size // cfg.num_attention_heads
kvpt = 2 * nl * nkh * hd * 2  # bf16 bytes/token (K+V)
print(f"KV/token={kvpt / 1024:.0f}KB (layers={nl} kv_heads={nkh} head_dim={hd})", flush=True)

llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.55, max_model_len=8192)
le = llm.llm_engine
sp1 = SamplingParams(temperature=0, max_tokens=1)


def prefill_ms(N):
    """重算:prefill N token 的 forward 时间(优先 metrics TTFT,失败用 wall-clock)。"""
    rid = "m_p"
    try:
        le.abort_request(rid)
    except Exception:
        pass
    t0 = time.time()
    le.add_request(rid, TokensPrompt(prompt_token_ids=[1] * N), sp1, arrival_time=t0)
    out = None
    while le.has_unfinished_requests():
        for o in le.step():
            if o.request_id == rid and o.finished:
                out = o
        if out:
            break
    if out:
        ttft = getattr(out.metrics, "first_token_time", None)
        arr = getattr(out.metrics, "arrival_time", None)
        if ttft and arr and ttft > arr:
            return (ttft - arr) * 1000
    return (time.time() - t0) * 1000


def pcie_ms(N):
    """reload 近似:N token KV 从 CPU(pinned)→GPU 的 PCIe 传输时间。"""
    nbytes = kvpt * N
    cpu = torch.empty(nbytes // 2, dtype=torch.bfloat16, pin_memory=True)
    _ = cpu.cuda()
    torch.cuda.synchronize()  # warmup
    ts = []
    for _ in range(5):
        t0 = time.time()
        g = cpu.cuda()  # noqa: F841
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    return min(ts)


def measure(label):
    print(f"\n=== {label} ===", flush=True)
    for N in [512, 1024, 2048, 4096]:
        pm = prefill_ms(N)
        cm = pcie_ms(N)
        ratio = f"{pm / cm:.1f}x" if cm > 0 else "?"
        print(f"N={N:>5}: prefill(重算)={pm:>7.1f}ms  pcie(reload)={cm:>6.2f}ms  "
              f"reload快{ratio}", flush=True)


measure("无干扰")

# GPU 压力:提交干扰请求占 GPU,step 跑几步让 GPU 忙,再测
print("\n--- 注入 GPU 压力(4×3000 token 干扰占 GPU)---", flush=True)
sp_i = SamplingParams(temperature=0, max_tokens=512)
for i in range(4):
    try:
        le.abort_request(f"intf_{i}")
    except Exception:
        pass
    le.add_request(f"intf_{i}", TokensPrompt(prompt_token_ids=[1] * 3000), sp_i,
                   arrival_time=time.time())
for _ in range(3):
    if not le.has_unfinished_requests():
        break
    for _o in le.step():  # noqa: F841
        pass
measure("GPU 压力(干扰占 GPU)")

while le.has_unfinished_requests():
    for _o in le.step():  # noqa: F841
        pass
