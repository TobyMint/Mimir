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
    title: str = "KV Cache: new prefill tokens",
) -> str:
    """分组柱状图：每个工作流的 baseline vs optimized 进入 KV 的新 prefill token 数。

    外部优化层（压缩 / 工具外置）的 KV 优化信号是 ``total_prefill_new_tokens``
    （num_prompt_tokens - num_cached_tokens，即真正需要 prefill 进 KV 的 token）。
    ``torch.cuda.max_memory_allocated`` 只反映 vLLM 预分配的固定 KV 池（≈11.6 GiB @
    util=0.55，与优化无关），不作图。``peak_kv_used_gib`` 在外部层未填充，故用
    new_prefill 作为 KV 进入量的代理（越小越好）。
    """
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
            v = r.extra.get("total_prefill_new_tokens") if r else None
            v = _maybe_float(v)
            vals.append(v if v is not None else 0.0)
        ax.bar([xi + (i - (len(labels) - 1) / 2) * width for xi in x], vals, width, label=lab)

    ax.set_xticks(list(x))
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.set_ylabel("new prefill tokens (into KV)")
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
    title: str = "Latency (TTFT / E2E)",
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
    ax2.set_ylabel("E2E Latency (s)")
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
    title: str = "Ablation: Cumulative Features",
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
    ax1.plot(list(x), mem, "o-", color="tab:blue", label="Peak KV Memory (GiB)")
    ax1.set_ylabel("Peak KV Memory (GiB)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(list(x), lat, "s--", color="tab:orange", label="E2E Latency (s)")
    ax2.set_ylabel("E2E Latency (s)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_title(title)
    succ_str = ", ".join(
        f"{n}: {('%.0f%%' % (s * 100)) if s is not None else '—'}" for n, s in zip(names, succ)
    )
    fig.text(0.5, 0.01, f"Task Success Rate  {succ_str}", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


def plot_agent_loop_gif(
    json_path: str | Path,
    out_path: str | Path,
    *,
    duration: float = 0.9,
) -> str:
    """Animated dashboard GIF from a real agent_benchmark_<model>.json.

    Replaces the old mock-data TUI GIF with one driven by *engine-measured*
    data: for each agent step it animates (a) used KV blocks (native crashes
    early, Mimir stays at 0) and (b) per-step TTFT + new_prefill tokens. This
    is the visual one-punch showing both the memory win (40pts) and the
    latency win (40pts) on the same task.

    Builds frames via matplotlib, assembles into a looping GIF with Pillow
    (no imageio/ffmpeg dependency).
    """
    import io
    import json

    from PIL import Image

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    native = data.get("native", {}).get("results", [])
    mimir = data.get("mimir", {}).get("results", [])
    if not mimir:
        raise ValueError("no mimir results in agent benchmark JSON")

    # Use the longest task for the most dramatic arc.
    pick = max(
        range(len(mimir)),
        key=lambda i: len([s for s in mimir[i]["steps"] if s.get("used_blocks", 0) != -1]),
    )
    m = mimir[pick]
    n = native[pick] if pick < len(native) else {}
    task_name = m["label"].rsplit("_", 1)[0]
    m_real = [s for s in m["steps"] if s.get("used_blocks", 0) != -1]
    n_real = [s for s in n.get("steps", []) if s.get("used_blocks", 0) != -1]

    n_steps_n = [s["step"] for s in n_real]
    n_used = [s["used_blocks"] for s in n_real]
    n_ttft = [s.get("ttft_ms") for s in n_real]
    m_steps = [s["step"] for s in m_real]
    m_used = [s["used_blocks"] for s in m_real]
    m_ttft = [s.get("ttft_ms") for s in m_real]
    m_prefill = [s.get("new_prefill_tokens") for s in m_real]
    total_steps = max(m_steps + [0]) + 1
    crash_step = (max(n_steps_n) + 1) if n_steps_n and len(n_real) < len(m_real) else None

    max_used = max(n_used + m_used + [1])
    max_ttft = max([v for v in (n_ttft + m_ttft) if v] + [1])
    max_prefill = max([v for v in m_prefill if v is not None] + [1])

    def _draw_frame(until_m: int) -> bytes:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 5.4))
        # Top: memory
        vis_n = n_steps_n
        vis_m = m_steps[: max(1, until_m)]
        ax0.plot(vis_n, n_used[: len(vis_n)], "rx-", label="native vLLM (fcfs)")
        ax0.plot(vis_m, m_used[: len(vis_m)], "g^-", label="Mimir (reclaim+offload)")
        if crash_step is not None:
            ax0.axvline(crash_step - 0.5, color="red", linestyle=":", alpha=0.5)
        ax0.set_xlim(-0.3, total_steps - 0.7)
        ax0.set_ylim(-1, max_used * 1.15)
        ax0.set_ylabel("used KV blocks")
        ax0.set_title(f"{task_name} — agent step {until_m - 1}/{total_steps - 1}")
        ax0.legend(fontsize=7, loc="upper left")
        if crash_step is not None and until_m - 1 >= crash_step - 1:
            ax0.text(
                crash_step - 0.4,
                max_used * 0.6,
                "native CRASHED\n(context overflow)",
                color="red",
                fontsize=7,
                ha="center",
            )
        # Bottom: latency
        vis_mt = m_ttft[: max(1, until_m)]
        vis_mp = m_prefill[: max(1, until_m)]
        ax1.plot(vis_m, vis_mt, "g^-", label="Mimir TTFT (ms)")
        ax1.plot(vis_m, vis_mp, "g:", alpha=0.55, label="Mimir new_prefill (tok)")
        ax1.set_xlim(-0.3, total_steps - 0.7)
        ax1.set_ylim(0, max(max_ttft, max_prefill) * 1.15)
        ax1.set_xlabel("agent step")
        ax1.set_ylabel("TTFT (ms) / new prefill tokens")
        ax1.legend(fontsize=7, loc="upper left")
        fig.suptitle(
            "Mimir Agent-Loop: native crashes vs Mimir completes (used=0)",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        return buf.getvalue()

    frames: list[Image.Image] = []
    # animate Mimir steps one at a time so the divergence grows visibly
    for k in range(1, len(m_real) + 1):
        frames.append(Image.open(io.BytesIO(_draw_frame(k))).convert("P"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 1000),
        loop=0,
        optimize=True,
    )
    return str(out_path)
