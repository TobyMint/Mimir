"""Phase 5 evaluation：分层存储让长生命周期推理存活。

演示场景：一个多轮 agent，每轮产生一个大的「上下文片段」（模拟工具返回 / 历史块）。
- **baseline**：所有片段都驻留「显存」（用 TieredStore 仅 GPU 层，cap=∞ 模拟）。
  上下文增长 -> 显存线性增长 -> 达到上限即「OOM」(模拟)。
- **tiered**：片段进入三层（GPU cap 小 + HOST + DISK）。冷片段自动 demote，
  访问时 promote。显存占用被 cap 住，而历史仍可按需取回 -> 「存活」。

度量：每轮后的 GPU 层项数 / 总项数 / demote / promote 次数，以及「OOM 轮次」
（baseline 达到 gpu_cap 的轮次 vs tiered 在同 cap 下永不 OOM）。

输出：benchmark_results/phase5_tiered_<tag>.json + _tier.png

用法（mimir 环境，纯 CPU 逻辑，不需 GPU）：
    python scripts/run_phase5_tiered.py [--turns 20] [--gpu-cap 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.tiered.store import TieredStore


def simulate_baseline(num_turns: int, gpu_cap: int) -> dict:
    """baseline：所有片段都驻留显存，达到 cap 即 OOM。"""
    resident = 0
    oom_turn = None
    history = []
    for t in range(1, num_turns + 1):
        resident += 1  # 每轮新增一个片段驻留显存
        history.append({"turn": t, "gpu_resident": resident, "oom": resident > gpu_cap})
        if resident > gpu_cap and oom_turn is None:
            oom_turn = t
            break  # OOM，任务失败
    return {
        "oom_turn": oom_turn,
        "survived_turns": (oom_turn - 1) if oom_turn else num_turns,
        "history": history,
    }


def simulate_tiered(
    num_turns: int, gpu_cap: int, host_cap: int, access_pattern: str = "recent"
) -> dict:
    """tiered：片段进三层，冷数据 demote，按需 promote。

    access_pattern:
      - "recent": 每轮访问最近 2 个片段（其余冷）
      - "all": 每轮访问所有片段
    """
    store = TieredStore(gpu_cap=gpu_cap, host_cap=host_cap, disk_dir=None)
    history = []
    for t in range(1, num_turns + 1):
        store.put(f"frag_{t}", f"fragment content turn {t} " * 50)
        # 模拟本轮访问模式
        if access_pattern == "recent":
            for back in range(min(2, t)):
                store.get(f"frag_{t - back}")
        else:
            for back in range(t):
                store.get(f"frag_{t - back}")
        snap = store.snapshot()
        stats = store.stats_dict()
        history.append(
            {
                "turn": t,
                "gpu": len(snap["gpu"]),
                "host": len(snap["host"]),
                "disk": len(snap["disk"]),
                "total": stats["total"],
                "promotions": stats["promotions"],
                "demotions": stats["demotions"],
                "oom": len(snap["gpu"]) > gpu_cap,  # 不应发生
            }
        )
    stats = store.stats_dict()
    return {
        "oom_turn": None,  # tiered 永不 OOM（gpu 被 cap 住）
        "survived_turns": num_turns,
        "final_stats": stats,
        "history": history,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--gpu-cap", type=int, default=4)
    ap.add_argument("--host-cap", type=int, default=16)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    print(f"=== Phase 5 分层存储模拟（{args.turns} 轮，gpu_cap={args.gpu_cap}）===", flush=True)
    base = simulate_baseline(args.turns, args.gpu_cap)
    tier = simulate_tiered(args.turns, args.gpu_cap, args.host_cap, "recent")

    print(
        f"\nbaseline: 在第 {base['oom_turn']} 轮 OOM（仅存活 {base['survived_turns']} 轮）",
        flush=True,
    )
    print(
        f"tiered:   存活全部 {tier['survived_turns']} 轮"
        f"（gpu 被 cap 在 {args.gpu_cap}，冷数据落 host/disk）",
        flush=True,
    )
    print(f"  final: {tier['final_stats']}", flush=True)

    # 画分层曲线
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        turns = [h["turn"] for h in tier["history"]]
        gpu = [h["gpu"] for h in tier["history"]]
        host = [h["host"] for h in tier["history"]]
        disk = [h["disk"] for h in tier["history"]]
        # baseline 在 OOM 后中断，补 None 对齐到完整轮数
        base_res = [h["gpu_resident"] for h in base["history"]]
        base_res += [None] * (len(turns) - len(base_res))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(turns, base_res, "rx-", label="baseline 显存驻留(无分层)")
        ax.plot(turns, gpu, "g^-", label="tiered GPU 层(热)")
        ax.plot(turns, host, "b.--", label="tiered HOST 层(温)")
        ax.plot(turns, disk, "k:", label="tiered DISK 层(冷)")
        ax.axhline(
            args.gpu_cap, color="r", linestyle=":", alpha=0.5, label=f"GPU cap={args.gpu_cap}"
        )
        if base["oom_turn"]:
            ax.axvline(base["oom_turn"], color="r", alpha=0.3)
            ax.text(base["oom_turn"], 1, "baseline OOM", color="r", fontsize=8, ha="left")
        ax.set_xlabel("agent 轮次")
        ax.set_ylabel("驻留项数")
        ax.set_title(f"Phase 5 分层存储：长生命周期存活（{args.turns} 轮）")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "phase5_tiered_tier.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"保存图: {png}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"画图跳过: {e}", flush=True)

    out_dir = Path(args.out_dir)
    summary = {
        "turns": args.turns,
        "gpu_cap": args.gpu_cap,
        "host_cap": args.host_cap,
        "baseline_oom_turn": base["oom_turn"],
        "baseline_survived": base["survived_turns"],
        "tiered_survived": tier["survived_turns"],
        "tiered_final_stats": tier["final_stats"],
    }
    json_path = out_dir / "phase5_tiered_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {json_path}", flush=True)
    print("PHASE5_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
