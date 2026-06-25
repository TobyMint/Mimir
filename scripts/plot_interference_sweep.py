# ruff: noqa: E501
"""画干扰强度 sweep 曲线。

读 benchmark_results/interference_sweep.json,画 3 子图(x 轴=干扰档 none/weak/medium/strong):
  - hit_ratio: native 命中率随压力退化曲线 + pin/pin+SSC 各档稳定领先(核心论据)
  - total_time_s: 诚实展示 pin 总时间可能反超 native(保活占显存降吞吐——非单调)
  - mean_ttft_ms: 诚实展示 pin+SSC 的 TTFT 被 reload 拉高(非单调)
none 档只跑了 native,pin/pinsc 缺失自动跳过。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # Agg 后端,须在 import pyplot 前设
import matplotlib.pyplot as plt

PROFILES_ORDER = ["none", "weak", "medium", "strong"]
MODES = ["native", "pin", "ssc"]
MODE_STYLE = {"native": ("o-", "tab:red"), "pin": ("s-", "tab:blue"), "ssc": ("^-", "tab:green")}


def main():
    path = Path("benchmark_results/interference_sweep.json")
    if not path.exists():
        print(f"{path} not found; 先跑 scripts/run_interference_sweep.py", file=sys.stderr)
        return 1
    sweep = json.loads(path.read_text(encoding="utf-8"))
    profiles = sweep["profiles"]
    present = [p for p in PROFILES_ORDER if p in profiles]
    xs = list(range(len(present)))

    def intf_tok(p):
        c = profiles[p]["config"]
        return c["intf_n"] * (c["intf_ctx"] + c["intf_maxtok"])

    xlabels = [f"{p}\n{intf_tok(p) // 1000}K干扰" for p in present]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    specs = [("hit_ratio", "命中率", axes[0]), ("total_time_s", "总时间 (s)", axes[1]),
             ("mean_ttft_ms", "mean TTFT (ms)", axes[2])]
    for key, title, ax in specs:
        for mode in MODES:
            ys = []
            for p in present:
                r = profiles[p]["results"].get(mode)
                ys.append(r[key] if r and r.get(key) is not None else None)
            xx = [x for x, y in zip(xs, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            if xx:
                mk, col = MODE_STYLE[mode]
                ax.plot(xx, yy, mk, color=col, label=mode, markersize=7, linewidth=2)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Interference sweep: native hit degrades with pressure; pin/pin+SSC stable",
                 fontsize=12)
    fig.tight_layout()
    out = Path("benchmark_results/interference_sweep.png")
    fig.savefig(out, dpi=130)
    print(f"PNG → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
