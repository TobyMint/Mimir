"""Benchmark 运行入口。

定义三类典型智能体工作流与评测指标，并对优化前后进行对比。本文件为评测脚手架：
实际推理（调用 vLLM / llama.cpp）随实现推进接入，但**评测框架与指标口径**已固化，
保证优化前后在同一硬件配置下口径一致、可复现。

运行::

    python -m benchmarks.run
    python -m benchmarks.run --workloads multi_turn --features prefix_cache lifecycle
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------- #
# 评测指标定义（口径固定，对应 docs/测试报告.md §4）
# --------------------------------------------------------------------------- #


@dataclass
class Metrics:
    """单次运行的评测结果。"""

    peak_gpu_memory_gb: float | None = None
    ttft_ms: float | None = None  # 首 token 延迟
    e2e_latency_s: float | None = None  # 端到端延迟
    throughput_tok_per_s: float | None = None
    task_success_rate: float | None = None  # 0.0 ~ 1.0


# --------------------------------------------------------------------------- #
# 典型智能体工作流定义（对应赛题要求）
# --------------------------------------------------------------------------- #


@dataclass
class Workload:
    """一类典型智能体工作流。"""

    name: str
    description: str
    features: list[str]  # 该场景推荐启用的 Mimir 特性
    run: Callable[[list[str]], Metrics] = field(default=lambda f: Metrics())


def _stub_run(workload_name: str) -> Callable[[list[str]], Metrics]:
    """实际推理接入前的占位 runner，提示后续接入点。"""

    def _run(features: list[str]) -> Metrics:
        # TODO(ai): 接入 vLLM / llama.cpp 后端执行真实推理并采集指标。
        #   - 优化前后须使用相同硬件配置与模型（见 docs/赛题说明.md 赛题要求）
        #   - 采集: nvidia-smi 峰值显存、TTFT、端到端延迟、吞吐、任务成功率
        print(f"      [stub] {workload_name}: 实际推理待接入后端，启用特性 = {features}")
        return Metrics()

    return _run


WORKLOADS: dict[str, Workload] = {
    "multi_turn": Workload(
        name="multi_turn",
        description="多轮对话（上下文持续累积）",
        features=["prefix_cache", "lifecycle"],
        run=_stub_run("multi_turn"),
    ),
    "tool_call": Workload(
        name="tool_call",
        description="工具调用 / ReAct（多次 function calling）",
        features=["tool_offload", "context_compress"],
        run=_stub_run("tool_call"),
    ),
    "multi_stage": Workload(
        name="multi_stage",
        description="多阶段决策 / Tree-of-Thought（分支推理）",
        features=["branch_cow", "prefix_cache"],
        run=_stub_run("multi_stage"),
    ),
}


# --------------------------------------------------------------------------- #
# 对比评测：baseline（未优化） vs optimized（Mimir）
# --------------------------------------------------------------------------- #


def _row(label: str, m: Metrics) -> str:
    def fmt(v: float | None, suffix: str = "") -> str:
        return "—" if v is None else f"{v:g}{suffix}"

    return (
        f"      {label:<22} "
        f"显存={fmt(m.peak_gpu_memory_gb, 'GB'):<10} "
        f"TTFT={fmt(m.ttft_ms, 'ms'):<10} "
        f"E2E={fmt(m.e2e_latency_s, 's'):<8} "
        f"成功率={m.task_success_rate if m.task_success_rate is not None else '—'}"
    )


def run_comparison(workloads: list[str], features: list[str]) -> None:
    """对给定工作流执行 baseline vs optimized 对比并打印。"""
    print("=" * 72)
    print("Mimir Benchmark — baseline (未优化) vs optimized (Mimir)")
    print(f"  工作流: {workloads}")
    print(f"  启用特性: {features or '(无)'}")
    print("=" * 72)
    for name in workloads:
        wl = WORKLOADS[name]
        print(f"\n[{name}] {wl.description}  (推荐特性: {wl.features})")
        print("  baseline:")
        print(_row("baseline", wl.run([])))
        print("  optimized:")
        print(_row("optimized", wl.run(features)))
    print("\n" + "=" * 72)
    print("注: 当前为评测脚手架（stub）。接入推理后端后，结果将写入 benchmark_results/。")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Mimir Benchmark")
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=list(WORKLOADS),
        choices=list(WORKLOADS),
        help="选择要运行的工作流（默认全部）",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=[],
        help="optimized 路径启用的 Mimir 特性开关",
    )
    args = parser.parse_args()
    run_comparison(args.workloads, args.features)


if __name__ == "__main__":
    main()
