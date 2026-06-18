"""Benchmark 编排：工作流 → 请求 → 驱动 vLLM 引擎 → 采集指标。

纯编排层：依赖 ``mimir.engine_vllm``（引擎适配）与 ``mimir.metrics``（指标），
以及 ``benchmarks.workloads``（工作流定义）。本身不含 vLLM 依赖，便于单测请求构建。

指标采集
--------
- **TTFT**：在每次 ``engine.chat`` 之前重置计时起点，调用结束后立即标记首个 token
  （单请求离线 generate 下，TTFT≈prefill 时间；多 token 时偏低估，但 baseline/optimized
  口径一致，可用于对比）。
- **KV 峰值块**：每个请求后查 ``kv_usage()``，取工作流全程的 ``used_blocks`` 峰值。
- **显存峰值**：由 ``MetricsCollector`` 的 ``torch.cuda.max_memory_allocated`` 给出
  （仅 v0 单进程有意义；见 engine_vllm 模块说明）。
"""

from __future__ import annotations

import time
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
    """驱动引擎跑完一条工作流，返回含 KV 峰值的 ``RunMetrics``。

    TTFT 取第一个请求的 prefill 时间（请求开始→生成结束）。
    """
    reqs = build_requests(case, max_tokens=max_tokens)
    col = MetricsCollector(device=engine.device)
    peak_kv_blocks = 0
    peak_kv_gib: float | None = None
    peak_kv_util: float | None = None

    with col.track(label) as c:
        ok = True
        for i, r in enumerate(reqs):
            # 每个 request 独立计时，第一个 request 的耗时近似 TTFT
            t_req = time.perf_counter()
            try:
                _txt, n = engine.chat(r.messages, max_tokens=r.max_tokens)
                if i == 0:
                    # 用首个 request 的生成完成时刻近似「首 token」
                    c.mark_first_token_custom(t_req)
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
    m.extra["engine_init_s"] = engine.engine_init_seconds
    return m
