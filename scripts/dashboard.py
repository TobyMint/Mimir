"""实时内存仪表盘（TUI，基于 rich）—— Phase 9 博眼球层。

在 SSH 会话里实时展示 Mimir 内存管理状态：
- 三层存储（GPU/HOST/DISK）的冷热分布
- KV 块生命周期分布（ACTIVE/EVICTABLE/PINNED）
- 分支树结构 + CoW 节省
- baseline vs optimized 显存/TTFT 对比

两种模式：
- ``--demo``：用内置模拟数据驱动，可离线看效果（无需 GPU）。
- ``--once``：渲染一帧到文件（PNG/文本），供报告截图。

用法（mimir 环境）：
    python scripts/dashboard.py --demo
    python scripts/dashboard.py --demo --once frame.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mimir.branch.cow import build_tot_tree
from mimir.kv_cache.lifecycle import (
    LifecycleEvictor,
)
from mimir.tiered.store import TieredStore


def _bar(frac: float, width: int = 24, fill: str = "█", empty: str = "░") -> str:
    n = int(max(0.0, min(1.0, frac)) * width)
    return fill * n + empty * (width - n)


def _line(label: str, n: int, total: int) -> str:
    pct = n / total * 100
    return f"{label} {_bar(n / total)}  {n:>3}  ({pct:4.1f}%)"


def tier_panel(store: TieredStore) -> Panel:
    snap = store.snapshot()
    st = store.stats_dict()
    total = max(1, st["total"])
    lines = [
        _line("GPU  (热)", len(snap["gpu"]), total),
        _line("HOST (温)", len(snap["host"]), total),
        _line("DISK (冷)", len(snap["disk"]), total),
        "",
        f"promotions 冷→热: {st['promotions']}   "
        f"demotions 热→冷: {st['demotions']}   disk_reads: {st['disk_reads']}",
    ]
    return Panel("\n".join(lines), title="三层存储 (GPU/HOST/DISK)", border_style="cyan")


def lifecycle_panel(evictor: LifecycleEvictor) -> Panel:
    snap = evictor.snapshot()
    by = snap["by_lifecycle"]
    total = max(1, snap["total"])
    active = by.get("active", 0)
    evictable = by.get("evictable", 0)
    pinned = by.get("pinned", 0)
    lines = [
        f"ACTIVE    {_bar(active / total)}  {active:>3}  (进行中任务)",
        f"EVICTABLE {_bar(evictable / total)}  {evictable:>3}  (可回收/任务已结束)",
        f"PINNED    {_bar(pinned / total)}  {pinned:>3}  (常驻 system 前缀)",
        "",
        f"lifecycle_reclaims(主动): {evictor.stats.lifecycle_reclaims}  "
        f"evictions(被动): {evictor.stats.evictions}  "
        f"hit_rate: {evictor.stats.hit_rate * 100:.1f}%",
    ]
    return Panel("\n".join(lines), title="KV 块生命周期", border_style="magenta")


def branch_panel(tree_kind: str = "8×2") -> Panel:
    tree = build_tot_tree(root_prefix_len=60, num_branches=8, own_tokens_per_branch=20, depth=2)
    s = tree.cow_savings()
    lines = [
        f"分支树: {tree_kind}  ({s['active_branches']} 活跃分支)",
        f"朴素 KV (无共享): {s['naive_kv_tokens']} tokens",
        f"CoW  KV (共享前缀): {s['cow_kv_tokens']} tokens",
        f"节省: {s['saved_tokens']} tokens  ({s['savings_pct']}%)",
        "",
        f"{_bar(1 - s['cow_kv_tokens'] / max(1, s['naive_kv_tokens']))}  CoW 相对朴素 KV 占用",
    ]
    return Panel("\n".join(lines), title="分支 CoW 树", border_style="green")


def comparison_panel() -> Panel:
    """baseline vs Mimir 汇总（来自已落盘的真实结果）。"""

    rows = [
        ("上下文压缩", "tool_call TTFT", "307ms", "27ms", "-91%"),
        ("工具数据外置", "tool_call TTFT", "304ms", "29ms", "-90%"),
        ("分支 CoW", "KV tokens", "7040", "1500", "-78.7%"),
        ("分层存储", "存活轮次", "4/20", "20/20", "OOM→存活"),
        ("生命周期淘汰", "主动回收", "0%", "100%", "+100%"),
        ("fp8 KV 量化", "KV 容量", "1772块", "3659块", "2.06x"),
        ("Phase M A/B", "10轮 used_blocks", "69 (原生)", "0 (Mimir)", "-100%"),
        ("Phase O 并发", "3agent峰值", "14 (原生)", "0 (Mimir)", "-100%"),
    ]
    t = Table(title="baseline vs Mimir (真实数据)", expand=True)
    t.add_column("方向", style="cyan")
    t.add_column("指标", style="white")
    t.add_column("baseline", style="red")
    t.add_column("Mimir", style="green")
    t.add_column("Δ", style="yellow")
    for d, m, b, o, dlt in rows:
        t.add_row(d, m, b, o, dlt)
    return Panel(t, title="优化效果汇总", border_style="yellow")


def make_layout(store: TieredStore, evictor: LifecycleEvictor) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(Group(tier_panel(store), lifecycle_panel(evictor))),
        Layout(Group(branch_panel(), comparison_panel())),
    )
    header = Align.center(
        Text("Mimir — 面向智能体的内存管理系统  |  实时内存仪表盘", style="bold white on blue"),
        vertical="middle",
    )
    layout["header"].update(Panel(header, border_style="blue"))
    layout["footer"].update(
        Panel(
            Align.center(
                "vLLM 0.10.2 · Qwen3-4B-Instruct-2507 · 单卡 RTX 3090  ·  (--demo 模拟数据)",
                vertical="middle",
            ),
            border_style="dim",
        )
    )
    return layout


def demo_loop(console: Console, *, frames: int, interval: float, once: str | None) -> None:
    """用模拟数据驱动仪表盘：逐步放入数据，看分层迁移与生命周期变化。"""
    store = TieredStore(gpu_cap=6, host_cap=12, disk_dir=None)
    evictor = LifecycleEvictor(capacity=10)

    # 预置一个 pinned system 前缀
    evictor.add("sys_prefix", "__shared__", pinned=True)

    with console.screen() if once is None else _nullcontext():
        for f in range(frames):
            # 模拟：每帧放几个块进 tiered + lifecycle，偶尔触发 finish_task
            store.put(f"frag_{f}", f"data {f}")
            evictor.add(f"task{f}_b0", task_id=f"task_{f}")
            evictor.add(f"task{f}_b1", task_id=f"task_{f}")
            if f > 0 and f % 3 == 0:
                store.get(f"frag_{f - 1}")  # 访问旧块 -> promote
                evictor.finish_task(f"task_{f - 1}")  # 任务结束 -> 回收
            layout = make_layout(store, evictor)
            if once:
                rec = Console(record=True, width=100)
                rec.print(layout)
                Path(once).write_text(rec.export_text(), encoding="utf-8")
                return
            console.print(layout)
            time.sleep(interval)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="模拟数据驱动")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--interval", type=float, default=0.8)
    ap.add_argument("--once", default=None, help="渲染一帧到文件后退出")
    args = ap.parse_args()

    console = Console()
    if args.once:
        demo_loop(console, frames=6, interval=0, once=args.once)
        print(f"已渲染一帧到 {args.once}")
        return 0
    if args.demo:
        demo_loop(console, frames=args.frames, interval=args.interval, once=None)
        return 0
    # 默认：渲染一帧静态快照到 stdout
    store = TieredStore(gpu_cap=6, host_cap=12, disk_dir=None)
    for i in range(10):
        store.put(f"f{i}", i)
    for i in range(3):
        store.get(f"f{i}")
    evictor = LifecycleEvictor(capacity=10)
    evictor.add("sys", "__shared__", pinned=True)
    for t in range(2):
        for b in range(3):
            evictor.add(f"t{t}_b{b}", task_id=f"task_{t}")
        evictor.finish_task(f"task_{t}")
    console.print(make_layout(store, evictor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
