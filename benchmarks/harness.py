"""Benchmark 编排：工作流 → 请求 → 驱动 vLLM 引擎 → 采集指标。

纯编排层：依赖 ``mimir.engine_vllm``（引擎适配）与 ``mimir.metrics``（指标），
以及 ``benchmarks.workloads``（工作流定义）。本身不含 vLLM 依赖，便于单测请求构建。
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.workloads import WorkloadCase
from mimir.engine_vllm import VLLMEngine
from mimir.metrics import MetricsCollector, RunMetrics


@dataclass
class RequestSpec:
    """一次推理请求（纯数据，可单测）。"""

    messages: list[dict[str, str]]
    max_tokens: int = 256
    label: str = ""


def build_requests(case: WorkloadCase, *, max_tokens: int = 256) -> list[RequestSpec]:
    """把 ``WorkloadCase`` 转为 ``RequestSpec`` 列表（纯函数，不依赖 vLLM）。

    - **multi_turn**：每轮一个请求，上下文累积（前缀可被 vLLM APC 复用）。
    - **tool_call**：每轮含大工具返回（baseline 全量进入上下文）。
    - **multi_stage**：N 个分支，共享 system+user 前缀。
    """
    sys_content = case.system
    if case.tool_schemas:
        sys_content += "\n\nAvailable tools:\n" + "\n".join(case.tool_schemas)
    sys_msg = {"role": "system", "content": sys_content}
    reqs: list[RequestSpec] = []

    if case.name == "multi_stage":
        base = [sys_msg, {"role": "user", "content": case.turns[0].content}]
        for b in range(case.branches):
            reqs.append(
                RequestSpec(
                    messages=[dict(m) for m in base],
                    max_tokens=max_tokens,
                    label=f"branch_{b}",
                )
            )
        return reqs

    history: list[dict[str, str]] = [sys_msg]
    for i, turn in enumerate(case.turns):
        history.append({"role": "user", "content": turn.content})
        if i < len(case.tool_results):
            tr = case.tool_results[i]
            history.append({"role": "assistant", "content": f"[tool: {tr.name}]"})
            history.append({"role": "tool", "content": tr.content})
        reqs.append(
            RequestSpec(
                messages=[dict(m) for m in history],
                max_tokens=max_tokens,
                label=f"turn_{i}",
            )
        )
    return reqs


def run_workload(
    engine: VLLMEngine,
    case: WorkloadCase,
    *,
    max_tokens: int = 256,
    label: str = "run",
) -> RunMetrics:
    """驱动引擎跑完一条工作流，返回含 KV 峰值的 ``RunMetrics``。"""
    reqs = build_requests(case, max_tokens=max_tokens)
    col = MetricsCollector(device=engine.device)
    peak_kv_blocks = 0
    peak_kv_gib: float | None = None
    peak_kv_util: float | None = None

    with col.track(label) as c:
        ok = True
        for i, r in enumerate(reqs):
            try:
                _txt, n = engine.chat(r.messages, max_tokens=r.max_tokens)
                if i == 0:
                    c.mark_first_token()
                c.add_output_tokens(n)
            except Exception as e:  # noqa: BLE001
                ok = False
                c.set_extra(error=str(e))
                break
            kv = engine.kv_usage()
            ub = kv.get("used_blocks")
            if ub is not None:
                peak_kv_blocks = max(peak_kv_blocks, ub)
            if kv.get("used_gib") is not None:
                peak_kv_gib = max(peak_kv_gib or 0.0, kv["used_gib"])
            if kv.get("utilization") is not None:
                peak_kv_util = max(peak_kv_util or 0.0, kv["utilization"])
        c.success = ok

    m = col.metrics()
    m.extra["workload"] = case.name
    m.extra["num_requests"] = len(reqs)
    m.extra["peak_kv_used_blocks"] = peak_kv_blocks or None
    m.extra["peak_kv_used_gib"] = peak_kv_gib
    m.extra["peak_kv_utilization"] = peak_kv_util
    return m
