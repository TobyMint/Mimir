"""Phase 6 evaluation：生命周期感知淘汰 vs 纯 LRU（对照 vLLM APC）。

跑一段多任务 agent 访问 trace，对比：
- LifecycleEvictor（任务结束主动回收）
- PureLRUEvictor（vLLM APC 风格，仅 LRU）

度量：命中率、回收块数、淘汰块数、容量压力下的存活。
输出：benchmark_results/phase6_lifecycle.json + _cmp.png

用法（纯 CPU，不需 GPU）：
    python scripts/run_phase6_lifecycle.py [--tasks 8] [--blocks-per-task 4] [--capacity 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.kv_cache.lifecycle import (
    LifecycleEvictor,
    PureLRUEvictor,
    simulate_agent_trace,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--blocks-per-task", type=int, default=4)
    ap.add_argument("--reuse", type=int, default=3)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    lc = LifecycleEvictor(capacity=args.capacity)
    lru = PureLRUEvictor(capacity=args.capacity)
    s_lc = simulate_agent_trace(lc, args.tasks, args.blocks_per_task, args.reuse, args.capacity)
    s_lru = simulate_agent_trace(lru, args.tasks, args.blocks_per_task, args.reuse, args.capacity)

    summary = {
        "config": {
            "tasks": args.tasks,
            "blocks_per_task": args.blocks_per_task,
            "reuse_within_task": args.reuse,
            "capacity": args.capacity,
        },
        "lifecycle": {
            "hits": s_lc.hits,
            "misses": s_lc.misses,
            "hit_rate": round(s_lc.hit_rate, 4),
            "evictions": s_lc.evictions,
            "lifecycle_reclaims": s_lc.lifecycle_reclaims,
        },
        "pure_lru": {
            "hits": s_lru.hits,
            "misses": s_lru.misses,
            "hit_rate": round(s_lru.hit_rate, 4),
            "evictions": s_lru.evictions,
            "lifecycle_reclaims": s_lru.lifecycle_reclaims,
        },
    }
    # 任务块总数（理论上若无容量限制会驻留的总块数）
    total_blocks = args.tasks * args.blocks_per_task
    summary["lifecycle_reclaim_pct"] = round(s_lc.lifecycle_reclaims / total_blocks * 100, 1)
    summary["lru_reclaim_pct"] = 0.0  # LRU 不主动回收
    summary["hit_rate_delta_pct"] = round((s_lc.hit_rate - s_lru.hit_rate) * 100, 2)

    print("=== Phase 6 生命周期淘汰 vs 纯 LRU ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    # 画对比图
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["命中率", "回收块数", "淘汰块数(被动)"]
        lc_vals = [s_lc.hit_rate * 100, s_lc.lifecycle_reclaims, s_lc.evictions]
        lru_vals = [s_lru.hit_rate * 100, s_lru.lifecycle_reclaims, s_lru.evictions]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, lc_vals, w, label="Mimir 生命周期淘汰")
        ax.bar(x + w / 2, lru_vals, w, label="纯 LRU (vLLM APC 风格)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Phase 6：生命周期感知淘汰 vs 纯 LRU")
        ax.legend(fontsize=9)
        for i, (a, b) in enumerate(zip(lc_vals, lru_vals, strict=False)):
            ax.text(i - w / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + w / 2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "phase6_lifecycle_cmp.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"保存图: {png}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"画图跳过: {e}", flush=True)

    out_dir = Path(args.out_dir)
    json_path = out_dir / "phase6_lifecycle.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {json_path}", flush=True)
    print("PHASE6_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
