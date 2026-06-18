"""测试报告图表生成（matplotlib，不依赖 GPU）。

把 ``RunMetrics`` 结果绘制为 baseline vs optimized 对比图，供测试报告嵌入。
所有函数输入均为 ``RunMetrics`` 列表，输出 PNG 路径。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt  # noqa: E402

from mimir.metrics import RunMetrics  # noqa: E402


def _maybe_float(v: object) -> float | None:
    try:
        if v is None:
            return None
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _group(results: Sequence[RunMetrics]) -> dict[tuple[str, str], RunMetrics]:
    """按 (workload, label) 索引，便于 baseline/optimized 配对。"""
    out: dict[tuple[str, str], RunMetrics] = {}
    for r in results:
        wl = r.extra.get("workload", "?")
        out[(str(wl), r.label)] = r
    return out


def plot_kv_mem_comparison(
    results: Sequence[RunMetrics],
    out_path: str | Path,
    *,
    title: str = "KV Cache 显存占用对比",
) -> str:
    """分组柱状图：每个工作流的 baseline vs optimized 峰值 KV 显存。"""
    grouped = _group(results)
    workloads = sorted({k[0] for k in grouped})
    labels = sorted({k[1] for k in grouped})
    x = range(len(workloads))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(workloads)), 4.2))
    for i, lab in enumerate(labels):
        vals = []
        for wl in workloads:
            r = grouped.get((wl, lab))
            v = r.extra.get("peak_kv_used_gib") if r else None
            v = _maybe_float(v)
            vals.append(v if v is not None else 0.0)
        ax.bar([xi + (i - (len(labels) - 1) / 2) * width for xi in x], vals, width, label=lab)

    ax.set_xticks(list(x))
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.set_ylabel("峰值 KV 显存 (GiB)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def plot_latency_comparison(
    results: Sequence[RunMetrics],
    out_path: str | Path,
    *,
    title: str = "延迟对比 (TTFT / E2E)",
) -> str:
    """每个工作流 baseline vs optimized 的 TTFT 与 E2E 延迟分组柱状图。"""
    grouped = _group(results)
    workloads = sorted({k[0] for k in grouped})
    labels = sorted({k[1] for k in grouped})
    x = range(len(workloads))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, 1.8 * len(workloads)), 7), sharex=True)
    for i, lab in enumerate(labels):
        ttfts, e2es = [], []
        for wl in workloads:
            r = grouped.get((wl, lab))
            ttfts.append(_maybe_float(r.ttft_ms if r else None) or 0.0)
            e2es.append(_maybe_float(r.e2e_latency_s if r else None) or 0.0)
        offs = [xi + (i - (len(labels) - 1) / 2) * width for xi in x]
        ax1.bar(offs, ttfts, width, label=lab)
        ax2.bar(offs, e2es, width, label=lab)
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title(title)
    ax1.legend()
    ax2.set_ylabel("E2E 延迟 (s)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(workloads, rotation=15, ha="right")
    ax2.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def plot_ablation_curve(
    ablation: list[tuple[str, float | None, float | None, float | None]],
    out_path: str | Path,
    *,
    title: str = "消融：逐步启用优化",
) -> str:
    """消融实验折线图。

    ``ablation`` 元素: ``(特性名, 峰值KV显存GiB, E2E延迟s, 成功率0~1)``。
    缺值允许 None。
    """
    names = [a[0] for a in ablation]
    mem = [a[1] for a in ablation]
    lat = [a[2] for a in ablation]
    succ = [a[3] for a in ablation]
    x = range(len(names))

    fig, ax1 = plt.subplots(figsize=(max(7, 1.1 * len(names)), 4.6))
    ax1.plot(list(x), mem, "o-", color="tab:blue", label="峰值 KV 显存 (GiB)")
    ax1.set_ylabel("峰值 KV 显存 (GiB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(list(x), lat, "s--", color="tab:orange", label="E2E 延迟 (s)")
    ax2.set_ylabel("E2E 延迟 (s)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_title(title)
    succ_str = ", ".join(
        f"{n}: {('%.0f%%' % (s * 100)) if s is not None else '—'}" for n, s in zip(names, succ)
    )
    fig.text(0.5, 0.01, f"任务成功率  {succ_str}", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)
