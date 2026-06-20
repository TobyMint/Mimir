# ruff: noqa: E501, E701, E702
"""Speculative decoding A/B：ngram（训练无关）开/关，度量 decode 加速。

赛题「资源受限场景推理加速」的直接贡献。ngram proposer（vLLM v1 原生，训练无关）：
用 prompt 里已出现的 n-gram 匹配预测下一批 token，主模型一次 verify 多个草稿 token。
3090 上无需 draft model（省显存）、无需训练，是单卡 agent decode 提速的干净路径。

对照：同一 agent 轨迹，关 spec decode vs 开 ngram spec decode（num_speculative_tokens=5）。
指标：每步 e2e + 输出 token 吞吐；accepted_tokens（命中的草稿数）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.harness import _req_metrics

from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from mimir.gpu import pick_least_busy_gpu


def _make_eng(model, util, mlen, *, spec: bool):
    extra = {"scheduling_policy": "mimir"}
    if spec:
        extra["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 2,
        }
    cfg = EngineConfig(
        model=model, dtype="bfloat16", gpu_memory_utilization=util,
        enable_prefix_caching=True, max_model_len=mlen, use_v1=True, extra=extra,
    )
    return VLLMEngineV1(cfg, device=0)


def _bench_decode(eng, msgs, *, max_tokens: int, repeats: int) -> dict:
    """跑 repeats 次，测平均 e2e + 输出 token 数 + 吞吐。"""
    times, out_toks = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ro = eng.chat_full(msgs, max_tokens=max_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        rm = _req_metrics(ro)
        n_out = rm.get("num_output_tokens") or len(ro.outputs[0].text.split())
        times.append(dt)
        out_toks.append(n_out)
    avg_e2e = sum(times) / len(times)
    avg_out = sum(out_toks) / len(out_toks)
    return {
        "avg_e2e_s": round(avg_e2e, 3),
        "avg_output_tokens": round(avg_out, 1),
        "throughput_tok_per_s": round(avg_out / avg_e2e, 1) if avg_e2e else None,
        "times": [round(t, 3) for t in times],
    }


def main() -> int:
    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU"); return 2
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    model = "/data/models/Qwen3-4B-Instruct-2507"
    results = {}
    for spec in [False, True]:
        label = "ngram_spec_on" if spec else "no_spec"
        print(f"\n=== {label} ===", flush=True)
        # 子进程隔离，确保每个引擎退出后 VRAM 完整释放给下一侧
        r = _run_side_subprocess(model, g.index, "1" if spec else "0")
        results[label] = r
        if "error" in r:
            print(f"  error: {r['error']}", flush=True)
        else:
            print(f"  avg_e2e={r['avg_e2e_s']}s out_tok={r['avg_output_tokens']} "
                  f"throughput={r['throughput_tok_per_s']} tok/s", flush=True)

    base = results.get("no_spec", {})
    spec_r = results.get("ngram_spec_on", {})
    speedup = None
    if isinstance(base, dict) and isinstance(spec_r, dict) and base.get("throughput_tok_per_s") and spec_r.get("throughput_tok_per_s"):
        speedup = round(spec_r["throughput_tok_per_s"] / base["throughput_tok_per_s"], 2)
    summary = {"model": Path(model).name, "speculative": "ngram(num_spec=5)", "no_spec": base,
               "ngram_spec_on": spec_r, "throughput_speedup_x": speedup}
    out = Path("benchmark_results"); out.mkdir(parents=True, exist_ok=True)
    jp = out / f"phase_specdecode_{Path(model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    nb = base.get("throughput_tok_per_s") if isinstance(base, dict) else None
    ns = spec_r.get("throughput_tok_per_s") if isinstance(spec_r, dict) else None
    print(f"\n吞吐: no_spec={nb} -> ngram={ns} tok/s (speedup {speedup}x)")
    print(f"JSON: {jp}")
    print("PHASE_SPECDECODE_OK")
    return 0


CHILD = r"""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
from benchmarks.harness import _req_metrics
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
spec = sys.argv[1] == "1"
extra = {"scheduling_policy": "mimir"}
if spec:
    extra["speculative_config"] = {"method": "ngram", "num_speculative_tokens": 5,
                                   "prompt_lookup_max": 4, "prompt_lookup_min": 2}
eng = VLLMEngineV1(EngineConfig(model=sys.argv[3], dtype="bfloat16", gpu_memory_utilization=0.90,
    enable_prefix_caching=True, max_model_len=4096, use_v1=True, extra=extra), device=0)
_ = eng.llm
msgs=[{"role":"system","content":"You are a helpful assistant. Answer thoroughly and reuse the structure of the question in your answer."},
      {"role":"user","content":"List the steps to manage KV cache memory for an agent, and for each step, explain the step, give an example of the step, and summarize the step. Be concrete and structured."}]
times, out_toks = [], []
for _ in range(3):
    t0=time.perf_counter(); ro=eng.chat_full(msgs, max_tokens=256, temperature=0.0); dt=time.perf_counter()-t0
    rm=_req_metrics(ro); n=rm.get("num_output_tokens") or len(ro.outputs[0].text.split())
    times.append(dt); out_toks.append(n)
avg_e2e=sum(times)/len(times); avg_out=sum(out_toks)/len(out_toks)
print("RESULT_JSON:"+json.dumps({"avg_e2e_s":round(avg_e2e,3),"avg_output_tokens":round(avg_out,1),
      "throughput_tok_per_s":round(avg_out/avg_e2e,1) if avg_e2e else None,
      "times":[round(t,3) for t in times]}))
"""


def _run_side_subprocess(model, gpu, spec_flag):
    import subprocess
    r = subprocess.run(["python", "-c", CHILD, spec_flag, str(gpu), model],
                       capture_output=True, text=True, env=dict(os.environ), timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[12:])
    return {"error": (r.stderr[-300:] or "no RESULT_JSON").replace("\r", "")}


if __name__ == "__main__":
    raise SystemExit(main())
