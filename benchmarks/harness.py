"""Benchmark 编排：工作流 → 请求 → 驱动 vLLM 引擎 → 采集指标。

纯编排层：依赖 ``mimir.engine_vllm``（引擎适配）与 ``mimir.metrics``（指标），
以及 ``benchmarks.workloads``（工作流定义）。本身不含 vLLM 依赖，便于单测请求构建。

指标采集（精确口径）
--------------------
直接读取 vLLM ``RequestOutput`` 的 per-request 指标，避免近似与竞态：

- **TTFT**：``metrics.first_token_time - metrics.arrival_time``（vLLM 实测）。
- **E2E**：工作流 wall-clock（首个请求 arrival → 末请求 finished）。
- **吞吐**：总输出 token / E2E。
- **前缀命中**：``num_cached_tokens``（vLLM APC 命中数，衡量 prefix 复用）。
- **新进 KV 的 prompt token**：``num_prompt_tokens - num_cached_tokens``（实际 prefill）。
- **KV 块峰值**：每请求前后采样 block_manager，取全程峰值（best-effort，单进程 v0）。
- **显存峰值**：``torch.cuda.max_memory_allocated``（单进程 v0）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _req_metrics(ro: Any) -> dict[str, Any]:
    """从 vLLM ``RequestOutput`` 提取 per-request 指标（容错）。"""
    out: dict[str, Any] = {}
    try:
        out["num_prompt_tokens"] = len(ro.prompt_token_ids)
    except Exception:
        out["num_prompt_tokens"] = None
    try:
        out["num_output_tokens"] = sum(len(getattr(o, "token_ids", []) or []) for o in ro.outputs)
    except Exception:
        out["num_output_tokens"] = None
    try:
        out["num_cached_tokens"] = int(getattr(ro, "num_cached_tokens", 0) or 0)
    except Exception:
        out["num_cached_tokens"] = 0
    m = getattr(ro, "metrics", None)
    ttft = e2e = None
    if m is not None:
        arrival = getattr(m, "arrival_time", None)
        first_tok = getattr(m, "first_token_time", None)
        finished = getattr(m, "finished_time", None)
        if arrival is not None and first_tok is not None:
            ttft = (first_tok - arrival) * 1000.0
        if arrival is not None and finished is not None:
            e2e = finished - arrival
    out["ttft_ms"] = ttft
    out["req_e2e_s"] = e2e
    return out


def run_workload(
    engine: VLLMEngine,
    case: WorkloadCase,
    *,
    max_tokens: int = 256,
    label: str = "run",
) -> RunMetrics:
    """驱动引擎跑完一条工作流，返回含 per-request 精确指标的 ``RunMetrics``。"""
    reqs = build_requests(case, max_tokens=max_tokens)
    col = MetricsCollector(device=engine.device)
    peak_kv_blocks = 0
    peak_kv_gib: float | None = None
    per_req: list[dict[str, Any]] = []
    total_out_tokens = 0
    total_cached = 0
    total_prefill_new = 0
    first_ttft: float | None = None
    avg_ttft_sum = 0.0
    avg_ttft_n = 0

    with col.track(label) as c:
        ok = True
        for i, r in enumerate(reqs):
            try:
                ro = engine.chat_full(r.messages, max_tokens=r.max_tokens)
                rm = _req_metrics(ro)
                per_req.append({"label": r.label, **rm})
                if rm.get("num_output_tokens"):
                    total_out_tokens += rm["num_output_tokens"]
                total_cached += rm.get("num_cached_tokens", 0) or 0
                if rm.get("num_prompt_tokens") is not None:
                    total_prefill_new += max(
                        0, rm["num_prompt_tokens"] - (rm.get("num_cached_tokens", 0) or 0)
                    )
                if rm.get("ttft_ms") is not None:
                    avg_ttft_sum += rm["ttft_ms"]
                    avg_ttft_n += 1
                    if i == 0:
                        first_ttft = rm["ttft_ms"]
                # 块峰值采样（单进程 v0，生成后块已释放，但并发/长上下文仍可观测）
                kv = engine.kv_usage()
                ub = kv.get("used_blocks")
                if ub is not None:
                    peak_kv_blocks = max(peak_kv_blocks, ub)
                if kv.get("used_gib") is not None:
                    peak_kv_gib = max(peak_kv_gib or 0.0, kv["used_gib"])
            except Exception as e:  # noqa: BLE001
                ok = False
                c.set_extra(error=str(e))
                break
        c.add_output_tokens(total_out_tokens)
        c.success = ok

    m = col.metrics()
    m.extra["workload"] = case.name
    m.extra["num_requests"] = len(reqs)
    m.extra["peak_kv_used_blocks"] = peak_kv_blocks or None
    m.extra["peak_kv_used_gib"] = peak_kv_gib
    m.extra["total_cached_tokens"] = total_cached
    m.extra["total_prefill_new_tokens"] = total_prefill_new
    m.extra["first_ttft_ms"] = first_ttft
    m.extra["avg_ttft_ms"] = (avg_ttft_sum / avg_ttft_n) if avg_ttft_n else None
    m.extra["per_request"] = per_req
    m.extra["engine_init_s"] = engine.engine_init_seconds
    # 用 vLLM 实测 TTFT 覆盖近似值（更准）
    if first_ttft is not None:
        m.ttft_ms = first_ttft
    return m
