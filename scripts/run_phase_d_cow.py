"""Phase D 评测：v1 引擎实测的分支 CoW Reuse vs Mimir BranchTree 预测。

跑 N 个分支（共享 system+user 前缀 + 各分支独有 seed），度量：
- v1 引擎实测：mimir_cow_reuses（跨分支Reuse块数）
- Mimir BranchTree 预测：cow_savings 的Reuse比例
两者应趋势一致（引擎是 ground truth，BranchTree 是预测投影）。

输出：benchmark_results/phase_d_cow_<model>.json

用法：python scripts/run_phase_d_cow.py [--branches 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.branch.cow import build_tot_tree
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from mimir.gpu import as_env, pick_least_busy_gpu

BRANCH_SEEDS = [
    "Approach A: decompose the problem into sub-problems.",
    "Approach B: find an analogous solved problem and adapt.",
    "Approach C: brainstorm edge cases first, then generalize.",
    "Approach D: estimate then verify by working backwards.",
    "Approach E: simplify constants, solve symbolically, restore.",
    "Approach F: enumerate and prune infeasible options.",
    "Approach G: divide and conquer on the dominant variable.",
    "Approach H: greedy heuristic then check optimality.",
]
SYS = (
    "You are a careful reasoning agent. For the question, explore the given "
    "approach and produce a concise final answer."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    os.environ.update(as_env(g))
    print(f"GPU {g.index}", flush=True)

    eng = VLLMEngineV1(
        EngineConfig(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_util,
            enable_prefix_caching=True,
            max_model_len=args.max_model_len,
        ),
        device=0,
    )
    _ = eng.llm
    print(f"engine_init={eng.engine_init_seconds:.1f}s", flush=True)

    question = "Estimate peak KV-cache memory for a 7B model at 32k context in fp16."
    per_branch = []
    for b in range(args.branches):
        tid = f"branch_{b}"
        seed = BRANCH_SEEDS[b % len(BRANCH_SEEDS)]
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"{question}\n\nExplore: {seed}"},
        ]
        eng.set_current_task(tid)
        _txt, n = eng.chat(msgs, max_tokens=args.max_tokens, temperature=0.0)
        st = eng.mimir_stats()
        per_branch.append(
            {
                "branch": tid,
                "used_blocks": st.get("used_blocks"),
                "mimir_cow_reuses": st.get("mimir_cow_reuses"),
                "out_tokens": n,
            }
        )
        print(
            f"  {tid}: used={st.get('used_blocks')} cow_reuses={st.get('mimir_cow_reuses')}",
            flush=True,
        )

    final = eng.mimir_stats()
    engine_cow = final.get("mimir_cow_reuses", 0)

    # Mimir BranchTree 预测（同 prefix/own token 估算）
    prefix_tokens = 60
    own_per_branch = 20
    tree = build_tot_tree(prefix_tokens, args.branches, own_per_branch, depth=1)
    pred = tree.cow_savings()
    naive = pred["naive_kv_tokens"]
    cow = pred["cow_kv_tokens"]
    pred_reuse_pct = (1 - cow / naive) * 100 if naive else 0.0

    summary = {
        "model": Path(args.model).name,
        "branches": args.branches,
        "engine_measured": {
            "mimir_cow_reuses": engine_cow,
            "per_branch": per_branch,
            "final_used_blocks": final.get("used_blocks"),
            "total_blocks": final.get("total_blocks"),
        },
        "mimir_branchtree_prediction": pred,
        "comparison": {
            "engine_cow_reuses": engine_cow,
            "prediction_saved_tokens": pred["saved_tokens"],
            "prediction_reuse_pct": round(pred_reuse_pct, 1),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_d_cow_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"\nengine mimir_cow_reuses={engine_cow}, BranchTree predicted reuse {pred_reuse_pct:.1f}%",
        flush=True,
    )
    print(f"保存: {jp}")
    print("PHASE_D_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
