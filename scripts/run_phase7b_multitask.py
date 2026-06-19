"""Phase 7b：多Task KV 协调评测（新Optimized方向，赛题「多模型/多Task」40 分子项）。

在单卡上模拟 N 个Concurrent agent Task，Comparison：
- coordinated（Mimir MultiTaskCoordinator）：共享前缀 pin + Task结束Reclaim
- uncoordinated（基线）：每Task各存前缀副本，无主动Reclaim

度量：共享saved %、Reclaim %、ConcurrentPeak。
输出：benchmark_results/phase7b_multitask.json + _cmp.png

用法（纯 CPU，不需 GPU）：
    python scripts/run_phase7b_multitask.py [--tasks 8] [--prefix 20] [--own 50]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.multitask import simulate_multi_task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=20)
    ap.add_argument("--own", type=int, default=50)
    ap.add_argument("--capacity", type=int, default=1000)
    ap.add_argument("--sweep", action="store_true", help="扫描不同Task数")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        # 扫描Task数 1..N，看协调收益随Concurrent增长
        rows = []
        for n in range(1, args.tasks + 1):
            r = simulate_multi_task(n, args.prefix, args.own, args.capacity)
            rows.append(
                {
                    "num_tasks": n,
                    "sharing_savings_pct": r["coordination_benefit"][
                        "sharing_savings_vs_naive_pct"
                    ],
                    "reclaim_pct": r["coordination_benefit"]["reclaim_vs_naive_pct"],
                    "coordinated_peak": r["coordinated"]["peak_resident_blocks"],
                    "naive_total": r["uncoordinated_baseline"][
                        "total_blocks_if_no_sharing_no_reclaim"
                    ],
                }
            )
        summary = {"sweep": rows}
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        json_path = out_dir / "phase7b_multitask_sweep.json"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        # 画扫描Curve
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ns = [r["num_tasks"] for r in rows]
            share = [r["sharing_savings_pct"] for r in rows]
            reclaim = [r["reclaim_pct"] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(ns, share, "g^-", label="共享前缀saved %")
            ax.plot(ns, reclaim, "b.--", label="Task结束Reclaim %")
            ax.set_xlabel("ConcurrentTask数")
            ax.set_ylabel("相对Naive基线saved %")
            ax.set_title("Phase 7b：多Task KV 协调收益随ConcurrentTask数增长")
            ax.legend(fontsize=9)
            fig.tight_layout()
            png = out_dir / "phase7b_multitask_sweep.png"
            fig.savefig(png, dpi=140)
            plt.close(fig)
            print(f"保存图: {png}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"画图跳过: {e}", flush=True)
        print(f"保存: {json_path}")
        print("PHASE7B_OK")
        return 0

    r = simulate_multi_task(args.tasks, args.prefix, args.own, args.capacity)
    print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
    json_path = out_dir / "phase7b_multitask.json"
    json_path.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {json_path}")
    print("PHASE7B_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
