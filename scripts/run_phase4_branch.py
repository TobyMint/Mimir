"""Phase 4 evaluation：分支推理 CoW。

构造一棵 ToT 分支树，在真实 vLLM 上跑各分支（共享 system+user 前缀 + 分支独有 token），
度量：
1. Mimir ``BranchTree`` 的 CoW 记账saved（Naive vs CoW KV token）。
2. vLLM APC 的真实前缀命中（num_cached_tokens）—— 验证共享在引擎层生效。
3. 分支数对 TTFT / 新进 KV 的影响。
4. 剪枝Reclaim的账面收益。

Comparison：分支推理「顺序跑」(每分支独立请求) vs「CoW 树」(显式共享前缀)。
输出：benchmark_results/phase4_branch_<model>.json + _mem.png + _savings.json

用法（mimir 环境）：
    python scripts/run_phase4_branch.py [--branches 4] [--depth 2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.branch.cow import build_tot_tree
from mimir.engine_vllm import EngineConfig, VLLMEngine
from mimir.gpu import as_env, pick_least_busy_gpu
from mimir.metrics import RunMetrics, save_results
from mimir.plots import plot_kv_mem_comparison

# 分支独有的 prompt 片段（模拟不同推理路径的展开）
BRANCH_SEEDS = [
    "Approach A: decompose the problem into sub-problems and solve each.",
    "Approach B: find an analogous solved problem and adapt its solution.",
    "Approach C: brainstorm edge cases first, then derive a general rule.",
    "Approach D: estimate the answer, then verify by working backwards.",
    "Approach E: simplify the constants, solve symbolically, then restore.",
    "Approach F: enumerate all possibilities and prune infeasible ones.",
    "Approach G: use divide and conquer along the dominant variable.",
    "Approach H: apply the greedy heuristic and check optimality.",
]

SYSTEM = (
    "You are a careful reasoning agent. For the given question, explore multiple solution "
    "approaches, reason step by step, and produce a concise final answer."
)


def branch_messages(question: str, branch_idx: int) -> list[dict[str, str]]:
    """构造一个分支的请求：共享 system+user 前缀 + 该分支独有的 approach seed。"""
    seed = BRANCH_SEEDS[branch_idx % len(BRANCH_SEEDS)]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{question}\n\nExplore: {seed}"},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU：无单卡空闲 >=6GiB。请协调 GPU 后重试。")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index} ({g.name}), free {g.mem_free_gib:.1f}GiB", flush=True)

    tag = Path(args.model).name
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
    print(f"engine_init_seconds={eng.engine_init_seconds:.1f}", flush=True)

    question = "Estimate peak KV-cache memory for a 7B model at 32k context in fp16."

    # 1) CoW 记账（纯逻辑）
    # 估算前缀 token：system+user 共享部分约 60 token，每分支独有 approach seed 约 20 token
    prefix_tokens = 60
    own_per_branch = 20
    tree = build_tot_tree(prefix_tokens, args.branches, own_per_branch, args.depth)
    savings = tree.cow_savings()
    print(f"\n=== CoW 记账（{args.branches} 分支 × depth {args.depth}）===", flush=True)
    print(json.dumps(savings, ensure_ascii=False, indent=2), flush=True)

    # 剪枝一半分支，看Reclaim
    active = [nid for nid in tree.nodes if not tree.nodes[nid].pruned and nid != 0]
    pruned_reclaim = 0
    for nid in active[: len(active) // 2]:
        pruned_reclaim += tree.prune(nid)
    after_prune = tree.cow_savings()
    print(f"剪枝一半分支后Reclaim own tokens: {pruned_reclaim}", flush=True)
    print(json.dumps(after_prune, ensure_ascii=False, indent=2), flush=True)

    # 2) 真实 vLLM：跑各分支（共享前缀），度量 APC 命中 + TTFT
    print(f"\n=== 真实推理：{args.branches} 个分支 ===", flush=True)
    results: list[RunMetrics] = []
    from mimir.metrics import MetricsCollector

    total_cached = 0
    total_new = 0
    ttfts = []
    for b in range(args.branches):
        msgs = branch_messages(question, b)
        col = MetricsCollector(device=0)
        with col.track(f"branch_{b}") as c:
            ro = eng.chat_full(msgs, max_tokens=args.max_tokens)
            c.mark_first_token()  # 占位；TTFT 取自 vLLM metrics
            from benchmarks.harness import _req_metrics

            rm = _req_metrics(ro)
            c.add_output_tokens(rm.get("num_output_tokens", 0))
            c.success = True
            if rm.get("ttft_ms") is not None:
                ttfts.append(rm["ttft_ms"])
            total_cached += rm.get("num_cached_tokens", 0) or 0
            if rm.get("num_prompt_tokens"):
                total_new += max(0, rm["num_prompt_tokens"] - (rm.get("num_cached_tokens", 0) or 0))
        m = col.metrics()
        if rm.get("ttft_ms") is not None:
            m.ttft_ms = rm["ttft_ms"]
        m.extra = {"workload": "multi_stage", "branch": b, **rm}
        results.append(m)
        new_tok = rm.get("num_prompt_tokens", 0) - (rm.get("num_cached_tokens") or 0)
        print(
            f"  branch {b}: TTFT={rm.get('ttft_ms')}ms prompt={rm.get('num_prompt_tokens')} "
            f"cached={rm.get('num_cached_tokens')} new={new_tok}",
            flush=True,
        )

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
    print(
        f"\n汇总: avg_TTFT={avg_ttft}ms total_cached={total_cached} total_new_prefill={total_new}",
        flush=True,
    )
    tot = total_cached + total_new
    reuse_pct = (total_cached / tot * 100) if tot else 0.0
    print(f"APC 前缀Reuse率: {total_cached}/{tot} = {reuse_pct:.1f}%", flush=True)

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"phase4_branch_{tag}.json"
    save_results(results, json_path)
    summary = {
        "branches": args.branches,
        "depth": args.depth,
        "cow_accounting": savings,
        "after_prune_accounting": after_prune,
        "pruned_reclaimed_tokens": pruned_reclaim,
        "real_apc": {
            "total_cached_tokens": total_cached,
            "total_new_prefill_tokens": total_new,
            "prefix_reuse_pct": round(total_cached / (total_cached + total_new) * 100, 1)
            if (total_cached + total_new)
            else 0.0,
            "avg_ttft_ms": avg_ttft,
        },
    }
    (out_dir / f"phase4_branch_{tag}_savings.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_kv_mem_comparison(
        results,
        out_dir / f"phase4_branch_{tag}_mem.png",
        title="Phase 4 分支 CoW：各分支PeakMemory",
    )
    print(f"\n保存: {json_path}")
    print(f"保存: {out_dir / f'phase4_branch_{tag}_savings.json'}")
    print("PHASE4_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
