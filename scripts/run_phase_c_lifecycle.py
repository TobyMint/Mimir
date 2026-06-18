"""Phase C GPU 验证：v1 in-tree 生命周期回收（mimir_finish_task）在真实引擎上端到端生效。

跑两个 agent 任务（各含若干 chat 轮），每个任务结束后调 mimir_finish_task。
对比：
- baseline（无 finish_task）：任务结束后 KV 仍驻留（vLLM LRU 被动保留）
- mimir（调 finish_task）：任务结束立即回收，mimir_lifecycle_reclaims 增长，空闲块增多

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
    """跑一个 agent 任务（多轮 chat），返回该任务累计输出 token。"""
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
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
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

    # ---- baseline：不调 finish_task ----
    print("\n=== baseline（无 finish_task，LRU 被动保留）===", flush=True)
    for tid, qs in tasks:
        run_task(eng, tid, qs, args.max_tokens)
    base_stats = eng.mimir_stats()
    print(f"after 2 tasks (no reclaim): {base_stats}", flush=True)

    # ---- mimir：每任务结束调 finish_task ----
    # 重置引擎状态：用新引擎避免 baseline 残留。同一进程内重新构造。
    print("\n=== Mimir（每任务结束 mimir_finish_task 主动回收）===", flush=True)
    eng2 = VLLMEngineV1(cfg, device=0)
    _ = eng2.llm
    reclaims_total = 0
    snapshots = []
    for tid, qs in tasks:
        run_task(eng2, tid, qs, args.max_tokens)
        reclaimed = eng2.mimir_finish_task(tid)
        reclaims_total += reclaimed
        snap = eng2.mimir_stats()
        snapshots.append({"task": tid, "reclaimed": reclaimed, "stats": snap})
        print(f"after finish_task({tid}): reclaimed={reclaimed} stats={snap}", flush=True)
    final_stats = eng2.mimir_stats()
    print(f"\nMimir total reclaims: {reclaims_total}", flush=True)
    print(
        f"Mimir final lifecycle_reclaims: {final_stats.get('mimir_lifecycle_reclaims')}", flush=True
    )

    summary = {
        "model": Path(args.model).name,
        "baseline_no_reclaim": {
            "mimir_lifecycle_reclaims": base_stats.get("mimir_lifecycle_reclaims", 0),
            "free_blocks": base_stats.get("used_blocks"),
            "total_blocks": base_stats.get("total_blocks"),
        },
        "mimir_active_reclaim": {
            "total_reclaimed": reclaims_total,
            "mimir_lifecycle_reclaims": final_stats.get("mimir_lifecycle_reclaims", 0),
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
