"""Phase J 评测：lifecycle-aware 主动回收闭环（mimir_reclaim_evictable）。

场景：任务 A 跑完，部分块仍被引用 -> mimir_finish_task 把它们标记 EVICTABLE（无法立即回收）。
显存压力点调 mimir_reclaim_evictable()，把所有 EVICTABLE 且 ref_cnt==0 的块物理释放。
度量：reclaim_evictable 前后 used_blocks 与 mimir_lifecycle_reclaims。

输出：benchmark_results/phase_j_reclaim_evictable_<model>.json

用法：python scripts/run_phase_j_reclaim_evictable.py
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=24)
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

    SYS = "You are a research agent. Answer about KV cache memory management briefly."
    # 任务 A 跑（产生块；finish_task 把仍引用的块标记 EVICTABLE）
    eng.set_current_task("task_A")
    eng.chat(
        [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "What is prefix caching and how does it save memory?"},
        ],
        max_tokens=args.max_tokens,
    )
    pre = eng.mimir_stats()
    print(
        f"task_A done: used={pre.get('used_blocks')} reclaims={pre.get('mimir_lifecycle_reclaims')}",
        flush=True,
    )

    reclaimed_task = eng.mimir_finish_task("task_A")
    after_finish = eng.mimir_stats()
    n_evictable = sum(
        1 for v in eng.mimir_block_pool().mimir_block_lifecycle.values() if v == "evictable"
    )
    print(
        f"after finish_task(A): reclaimed_now={reclaimed_task} used={after_finish.get('used_blocks')} "
        f"reclaims={after_finish.get('mimir_lifecycle_reclaims')} evictable_marked={n_evictable}",
        flush=True,
    )

    # Phase J：主动扫描回收所有 EVICTABLE
    reclaimed_sweep = eng.mimir_reclaim_evictable()
    post = eng.mimir_stats()
    print(
        f"after mimir_reclaim_evictable(): swept={reclaimed_sweep} used={post.get('used_blocks')} "
        f"reclaims={post.get('mimir_lifecycle_reclaims')}",
        flush=True,
    )

    summary = {
        "model": Path(args.model).name,
        "task_a_used_blocks_before_finish": pre.get("used_blocks"),
        "finish_task_A_reclaimed_now": reclaimed_task,
        "evictable_marked_after_finish": n_evictable,
        "reclaim_evictable_swept": reclaimed_sweep,
        "final_used_blocks": post.get("used_blocks"),
        "total_mimir_lifecycle_reclaims": post.get("mimir_lifecycle_reclaims"),
        "interpretation": "mimir_reclaim_evictable closes the loop: blocks that finish_task could not immediately free (still referenced) get marked EVICTABLE and are swept at the pressure point.",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_j_reclaim_evictable_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {jp}")
    print("PHASE_J_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
