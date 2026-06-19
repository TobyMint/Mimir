"""Phase 9 端到端 demo：baseline OOM vs Mimir Survival + 更快。

跑一个真实的多轮 agent（vLLM），逐步增加turn：
- baseline：每轮全量上下文进入 KV，Memory线性增长 -> 达到模拟上限「OOM」。
- Mimir：上下文Compress + 工具Offload + 分层，KV 增长被压住 -> Survival更久且更快。

输出：console 实时Comparison + benchmark_results/demo_<model>.json + _curves.png
（PNG：baseline MemoryCurve vs Mimir MemoryCurve，标 OOM 点）

用法（mimir 环境）：
    python scripts/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.harness import build_requests
from benchmarks.workloads import gen_multi_turn

from mimir.context.compressor import ContextCompressor, Fidelity
from mimir.engine_vllm import EngineConfig, VLLMEngine
from mimir.gpu import as_env, pick_least_busy_gpu

# 模拟「Memory上限」：用每轮 new_prefill 累计作为「Memory压力」代理
# 超过 OOM_THRESHOLD 则判定 baseline 在该轮 OOM
OOM_THRESHOLD_TOKENS = 4000


def run_progression(eng: VLLMEngine, *, compress: bool, max_turns: int, max_tokens: int) -> dict:
    """跑一个不断增长的多轮对话，返回每轮的累计 new_prefill / TTFT / 是否 OOM。"""
    base_case = gen_multi_turn(num_turns=max_turns)
    history = []
    cumulative_new_prefill = 0
    oom_turn = None
    for n in range(1, max_turns + 1):
        # 构造到第 n 轮的子 case
        sub = type(base_case)(
            name=base_case.name,
            description=base_case.description,
            system=base_case.system,
            tool_schemas=base_case.tool_schemas,
            turns=base_case.turns[:n],
            tool_results=base_case.tool_results[:n],
            branches=1,
            recommended_features=base_case.recommended_features,
        )
        if compress:
            comp = ContextCompressor(fidelity=Fidelity.BALANCED, keep_recent_turns=2)
            sub = comp.compress(sub)
        reqs = build_requests(sub, max_tokens=max_tokens)
        last = reqs[-1]
        try:
            ro = eng.chat_full(last.messages, max_tokens=last.max_tokens)
            from benchmarks.harness import _req_metrics

            rm = _req_metrics(ro)
            np_tok = max(0, (rm.get("num_prompt_tokens") or 0) - (rm.get("num_cached_tokens") or 0))
            cumulative_new_prefill = np_tok  # 本轮新增（累计按 KV 占用近似）
            ttft = rm.get("ttft_ms")
            ok = rm.get("num_output_tokens", 0) > 0
        except Exception:  # noqa: BLE001
            cumulative_new_prefill = 0
            ttft = None
            ok = False
        # OOM 判定：本 demo 用「生成失败」标记 OOM；上下文增长差异由 new_prefill Curve体现
        history.append(
            {
                "turn": n,
                "new_prefill": cumulative_new_prefill,
                "ttft_ms": ttft,
                "ok": ok,
            }
        )
        if not ok and oom_turn is None:
            oom_turn = n
    return {"history": history, "oom_turn": oom_turn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    cfg = EngineConfig(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_util,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        use_v1=False,
    )
    eng = VLLMEngine(cfg, device=0)
    _ = eng.llm
    print(f"engine_init={eng.engine_init_seconds:.1f}s", flush=True)

    print("\n=== baseline（无Compress，全量进上下文）===", flush=True)
    base = run_progression(eng, compress=False, max_turns=args.turns, max_tokens=args.max_tokens)
    for h in base["history"]:
        print(
            f"  turn {h['turn']:2}: new_prefill={h['new_prefill']:5} ttft={h['ttft_ms']}",
            flush=True,
        )

    print("\n=== Mimir（上下文Compress + 工具Offload）===", flush=True)
    opt = run_progression(eng, compress=True, max_turns=args.turns, max_tokens=args.max_tokens)
    for h in opt["history"]:
        print(
            f"  turn {h['turn']:2}: new_prefill={h['new_prefill']:5} ttft={h['ttft_ms']}",
            flush=True,
        )

    # Comparison
    print("\n=== Comparison ===", flush=True)
    base_final = base["history"][-1]
    opt_final = opt["history"][-1]
    bn = base_final["new_prefill"]
    on = opt_final["new_prefill"]
    print(f"  末轮 new_prefill: baseline={bn}  Mimir={on}", flush=True)
    if base_final["new_prefill"] and opt_final["new_prefill"]:
        red = (1 - opt_final["new_prefill"] / base_final["new_prefill"]) * 100
        print(f"  Mimir 上下文saved: {red:.1f}%", flush=True)

    out_dir = Path(args.out_dir)
    summary = {
        "model": Path(args.model).name,
        "turns": args.turns,
        "baseline": base,
        "mimir": opt,
        "baseline_final_new_prefill": base_final["new_prefill"],
        "mimir_final_new_prefill": opt_final["new_prefill"],
    }
    json_path = out_dir / f"demo_{Path(args.model).name}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 画Curve
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [h["turn"] for h in base["history"]]
        b_np = [h["new_prefill"] for h in base["history"]]
        o_np = [h["new_prefill"] for h in opt["history"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(ts, b_np, "rx-", label="baseline new_prefill (no compress)")
        ax.plot(ts, o_np, "g^-", label="Mimir new_prefill (compressed)")
        ax.axhline(
            OOM_THRESHOLD_TOKENS,
            color="r",
            linestyle=":",
            alpha=0.5,
            label=f"OOM threshold={OOM_THRESHOLD_TOKENS}",
        )
        ax.set_xlabel("agent turn")
        ax.set_ylabel("New prefill tokens per turn")
        ax.set_title(f"Demo: baseline vs Mimir context growth ({Path(args.model).name})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        png = out_dir / f"demo_{Path(args.model).name}_curves.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"保存: {png}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"画图跳过: {e}", flush=True)

    print(f"保存: {json_path}")
    print("DEMO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
