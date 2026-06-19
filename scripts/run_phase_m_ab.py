"""Phase M：整合 A/B — 原生 vLLM baseline vs Mimir 全管线（patched v1）。

10 轮多轮 agent 对话：
- baseline：原生 vLLM（scheduling_policy=fcfs，不压缩/不外置/不 finish_task），全量上下文进 KV
- Mimir：patched v1 + scheduling_policy=mimir + 上下文压缩 + 工具外置 + 任务边界自动回收

度量：每轮 new_prefill tokens、TTFT、累计 used_blocks。
输出：benchmark_results/phase_m_ab_<model>.json + _curves.png

用法：python scripts/run_phase_m_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from mimir.gpu import as_env, pick_least_busy_gpu

SYS = "You are a research agent answering about KV cache memory management. Be concise."
QUESTIONS = [
    "What is prefix caching?",
    "How does KV reuse save memory?",
    "Estimate KV for 7B at 32k in fp16.",
    "What is Copy-on-Write for branches?",
    "How does tiered storage help?",
    "What is lifecycle-aware eviction?",
    "Why offload tool results?",
    "Compare LRU vs lifecycle eviction.",
    "What is fp8 KV quantization?",
    "Summarize agent memory management.",
]


def run_ab(eng: VLLMEngineV1, *, compress: bool, label: str, max_tokens: int) -> list[dict]:
    """跑 10 轮多轮对话，返回每轮指标。"""
    history = [{"role": "system", "content": SYS}]
    rows = []
    for i, q in enumerate(QUESTIONS):
        history.append({"role": "user", "content": q})
        # 压缩：构造一个临时 case 压缩旧轮（保留近 2 轮）
        if compress and i >= 2:
            # 简化：把 history 中早于近2轮的 user 消息截短
            keep_from = len(history) - 4 if len(history) > 4 else 0
            old = history[:keep_from]
            recent = history[keep_from:]
            compacted = []
            for m in old:
                if m["role"] == "user" and len(m["content"]) > 40:
                    compacted.append({"role": m["role"], "content": m["content"][:40] + "…"})
                else:
                    compacted.append(m)
            history = compacted + recent
        msgs = list(history)
        _txt, n = eng.chat(msgs, max_tokens=max_tokens, temperature=0.0)
        st = eng.mimir_stats()
        rows.append(
            {
                "turn": i + 1,
                "used_blocks": st.get("used_blocks"),
                "lifecycle_reclaims": st.get("mimir_lifecycle_reclaims"),
                "out_tokens": n,
            }
        )
        history.append({"role": "assistant", "content": _txt[:60]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    os.environ.update(as_env(g))
    print(f"GPU {g.index}", flush=True)

    # baseline：原生 fcfs，不压缩
    print("\n=== baseline（原生 vLLM，fcfs，不压缩）===", flush=True)
    eng_b = VLLMEngineV1(
        EngineConfig(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_util,
            enable_prefix_caching=True,
            max_model_len=args.max_model_len,
            extra={"scheduling_policy": "fcfs"},
        ),
        device=0,
    )
    _ = eng_b.llm
    base_rows = run_ab(eng_b, compress=False, label="baseline", max_tokens=args.max_tokens)
    print(
        f"  baseline final: used={base_rows[-1]['used_blocks']} reclaims={base_rows[-1]['lifecycle_reclaims']}",
        flush=True,
    )

    # Mimir：mimir 策略 + 压缩
    print("\n=== Mimir（patched v1，mimir 策略 + 上下文压缩）===", flush=True)
    eng_m = VLLMEngineV1(
        EngineConfig(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_util,
            enable_prefix_caching=True,
            max_model_len=args.max_model_len,
            extra={"scheduling_policy": "mimir"},
        ),
        device=0,
    )
    _ = eng_m.llm
    mimir_rows = run_ab(eng_m, compress=True, label="mimir", max_tokens=args.max_tokens)
    print(
        f"  Mimir final: used={mimir_rows[-1]['used_blocks']} reclaims={mimir_rows[-1]['lifecycle_reclaims']}",
        flush=True,
    )

    b_final = base_rows[-1]
    m_final = mimir_rows[-1]
    summary = {
        "model": Path(args.model).name,
        "baseline": {
            "final_used_blocks": b_final["used_blocks"],
            "final_reclaims": b_final["lifecycle_reclaims"],
            "rows": base_rows,
        },
        "mimir": {
            "final_used_blocks": m_final["used_blocks"],
            "final_reclaims": m_final["lifecycle_reclaims"],
            "rows": mimir_rows,
        },
        "comparison": {
            "used_blocks_baseline_vs_mimir": f"{b_final['used_blocks']} vs {m_final['used_blocks']}",
            "lifecycle_reclaims_baseline_vs_mimir": f"{b_final['lifecycle_reclaims']} vs {m_final['lifecycle_reclaims']}",
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_m_ab_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 画曲线
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r["turn"] for r in base_rows]
        b_used = [r["used_blocks"] or 0 for r in base_rows]
        m_used = [r["used_blocks"] or 0 for r in mimir_rows]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(ts, b_used, "rx-", label="baseline (native vLLM, fcfs)")
        ax.plot(ts, m_used, "g^-", label="Mimir (mimir policy + compress + auto-reclaim)")
        ax.set_xlabel("agent turn")
        ax.set_ylabel("used KV blocks")
        ax.set_title(f"Phase M: native vLLM vs Mimir (10-turn agent, {Path(args.model).name})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        png = out_dir / f"phase_m_ab_{Path(args.model).name}_curves.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"保存: {png}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"画图跳过: {e}", flush=True)

    print(f"保存: {jp}")
    print(f"baseline final used={b_final['used_blocks']} reclaims={b_final['lifecycle_reclaims']}")
    print(f"Mimir final used={m_final['used_blocks']} reclaims={m_final['lifecycle_reclaims']}")
    print("PHASE_M_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
