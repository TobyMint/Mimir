"""Phase C GPU 验证：v1 in-tree 生命周期Reclaim（mimir_finish_task）在真实引擎上端到端生效。

跑两个 agent Task（各含若干 chat 轮），每个Task结束后调 mimir_finish_task。
Comparison：
- baseline（无 finish_task）：Task结束后 KV 仍驻留（vLLM LRU 被动保留）
- mimir（调 finish_task）：Task结束立即Reclaim，mimir_lifecycle_reclaims 增长，空闲块增多

度量：mimir_lifecycle_reclaims、free_blocks、peak used_blocks。
输出：benchmark_results/phase_c_lifecycle_<model>.json

用法（mimir 环境 + activate_env.sh）：
    python scripts/run_phase_c_lifecycle.py
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


def run_task(eng: VLLMEngineV1, task_id: str, questions: list[str], max_tokens: int) -> int:
    """跑一个 agent Task（多轮 chat），返回该Task累计输出 token。"""
    eng.set_current_task(task_id)
    out_tokens = 0
    msgs = [{"role": "system", "content": f"You are agent {task_id}. Answer briefly."}]
    for q in questions:
        msgs.append({"role": "user", "content": q})
        _txt, n = eng.chat(msgs, max_tokens=max_tokens, temperature=0.0)
        out_tokens += n
    return out_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
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
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    cfg = EngineConfig(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_util,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
    )
    eng = VLLMEngineV1(cfg, device=0)
    _ = eng.llm
    print(f"engine_init={eng.engine_init_seconds:.1f}s", flush=True)

    QUESTIONS = [
        "What is 2+2?",
        "Name a primary color.",
        "What is the capital of France?",
    ]
    tasks = [("task_A", QUESTIONS), ("task_B", ["Explain recursion.", "What is an LLM?"])]

    # 单引擎 A/B：跑Task，记录「无 finish_task」的驻留块数；再手动 finish 各Task，
    # Comparison mimir_lifecycle_reclaims 增长。避免双引擎叠加Memory OOM。
    print("\n=== 跑两个Task（不调 finish_task）===", flush=True)
    for tid, qs in tasks:
        run_task(eng, tid, qs, args.max_tokens)
    pre = eng.mimir_stats()
    print(
        f"Task完成后（未Reclaim）: used_blocks={pre.get('used_blocks')} "
        f"total={pre.get('total_blocks')} reclaims={pre.get('mimir_lifecycle_reclaims')}",
        flush=True,
    )

    # 现在 Mimir 主动Reclaim每个Task
    print("\n=== Mimir 主动Reclaim（mimir_finish_task）===", flush=True)
    reclaims_total = 0
    snapshots = []
    for tid, _qs in tasks:
        reclaimed = eng.mimir_finish_task(tid)
        reclaims_total += reclaimed
        snap = eng.mimir_stats()
        snapshots.append({"task": tid, "reclaimed": reclaimed, "stats": snap})
        rb = snap.get("mimir_lifecycle_reclaims")
        print(
            f"after finish_task({tid}): reclaimed={reclaimed} "
            f"used_blocks={snap.get('used_blocks')} reclaims={rb}",
            flush=True,
        )
    final_stats = eng.mimir_stats()
    print(
        f"\nMimir total reclaims: {reclaims_total}  "
        f"counter: {final_stats.get('mimir_lifecycle_reclaims')}",
        flush=True,
    )
    print(
        f"used_blocks: {pre.get('used_blocks')} (pre) -> {final_stats.get('used_blocks')} (post)",
        flush=True,
    )

    summary = {
        "model": Path(args.model).name,
        "before_reclaim": {
            "used_blocks": pre.get("used_blocks"),
            "total_blocks": pre.get("total_blocks"),
            "mimir_lifecycle_reclaims": pre.get("mimir_lifecycle_reclaims", 0),
        },
        "after_reclaim": {
            "total_reclaimed": reclaims_total,
            "mimir_lifecycle_reclaims": final_stats.get("mimir_lifecycle_reclaims", 0),
            "used_blocks": final_stats.get("used_blocks"),
            "per_task": snapshots,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"phase_c_lifecycle_{Path(args.model).name}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {json_path}")
    print("PHASE_C_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
