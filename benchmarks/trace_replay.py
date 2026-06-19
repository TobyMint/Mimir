"""在固定 DeepSeek 轨迹上回放引擎，逐步测量 KV / TTFT / new_prefill。

与 ``agent_loop.run_agent_loop`` 的区别：
- ``run_agent_loop`` 让**本地引擎**逐步生成（轨迹随机、依赖弱模型 agent 能力）。
- ``replay_trace`` 用 **DeepSeek 预生成的固定轨迹**，native 与 Mimir 跑同一份上下文，
  A/B 干净（同 token、同工具结果），且轨迹反映前沿模型的真实 agent 行为。

回放语义：轨迹是一串消息 [sys, user, assistant_0, tool_result_0, assistant_1, ...]。
每到一个「即将生成 assistant_k」的位置，把当前前缀发给引擎做一次 prefill+短生成，
记录 used_blocks / ttft / new_prefill（前缀越长，KV 越大、prefill 越大）。
Mimir 模式下，历史工具结果经 ``ToolDataStore.put`` 外置，上下文只留引用。
"""

from __future__ import annotations

import time
from typing import Any

from benchmarks.agent_loop import AgentRunResult, AgentStepResult
from benchmarks.harness import _req_metrics


def _is_tool_result_msg(m: dict[str, str]) -> bool:
    return m["role"] == "user" and m["content"].startswith("[TOOL_RESULT ")


def _tool_result_name(m: dict[str, str]) -> str:
    # "[TOOL_RESULT search]\n..."
    head = m["content"].split("]", 1)[0]
    return head.replace("[TOOL_RESULT ", "").strip()


def replay_trace(
    eng: Any,
    trace: dict[str, Any],
    *,
    policy: str = "mimir",
    tool_offload: bool = True,
    max_tokens: int = 8,
) -> AgentRunResult:
    """回放一条 DeepSeek 轨迹，逐步测量。

    ``max_tokens``：每步只生成极少 token（我们关心的是 prefill/KV，不是生成质量）。
    """
    from mimir.tools.offload import ToolDataStore

    store = ToolDataStore() if tool_offload else None
    label = f"{trace['task']}_{policy}"
    result = AgentRunResult(label=label, policy=policy, tool_offload=tool_offload)

    msgs = trace["messages"]
    # 找出所有「assistant 消息出现的位置」——每个位置是一个测量步（在该 assistant 之前的
    # 前缀就是要 prefill 的上下文）。
    assistant_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]

    step_i = 0
    for ai in assistant_idx:
        # 构造前缀 [0:ai]，必要时把工具结果外置
        prefix: list[dict[str, str]] = []
        for m in msgs[:ai]:
            if _is_tool_result_msg(m):
                raw = m["content"].split("]\n", 1)[1] if "]\n" in m["content"] else m["content"]
                if store is not None:
                    ctx_content = store.put(_tool_result_name(m), raw)
                else:
                    ctx_content = raw
                prefix.append(
                    {"role": "user",
                     "content": f"[TOOL_RESULT {_tool_result_name(m)}]\n{ctx_content}"}
                )
            else:
                prefix.append(dict(m))

        task_id = f"{trace['task']}_step_{step_i}"
        set_task = getattr(eng, "set_current_task", None)
        if callable(set_task):
            set_task(task_id)

        t0 = time.perf_counter()
        crashed_msg: str | None = None
        try:
            if hasattr(eng, "chat_full"):
                ro = eng.chat_full(prefix, max_tokens=max_tokens, temperature=0.0)
                ttft = (_req_metrics(ro)).get("ttft_ms")
                np_tok = (_req_metrics(ro)).get("num_prompt_tokens")
                nc_tok = (_req_metrics(ro)).get("num_cached_tokens", 0) or 0
                new_prefill = max(0, np_tok - nc_tok) if np_tok is not None else None
            else:
                _, _ = eng.chat(prefix, max_tokens=max_tokens, temperature=0.0)
                ttft = None
                new_prefill = None
        except Exception as exc:  # noqa: BLE001
            crashed_msg = str(exc)[:240] or exc.__class__.__name__
            ttft = None
            new_prefill = None
        t1 = time.perf_counter()

        if crashed_msg is not None:
            result.final_answer = f"(crashed: {crashed_msg})"
            result.steps.append(
                AgentStepResult(
                    step=step_i, text="[CRASHED]", tool_called=None, tool_result_bytes=0,
                    used_blocks=-1, ttft_ms=ttft, new_prefill_tokens=new_prefill,
                    e2e_s=t1 - t0, total_context_msgs=len(prefix),
                )
            )
            break

        stats = {}
        get_stats = getattr(eng, "mimir_stats", None)
        if callable(get_stats):
            stats = get_stats()
        used = stats.get("used_blocks", 0) or 0
        result.peak_used_blocks = max(result.peak_used_blocks, used)

        # 累计工具数据（用于报外置量）
        if store is not None:
            st = store.stats()
            result.total_tool_data_bytes = st.get("offloaded_chars", 0)
        else:
            result.total_tool_data_bytes = sum(
                len(m["content"]) for m in prefix if _is_tool_result_msg(m)
            )

        result.steps.append(
            AgentStepResult(
                step=step_i, text="", tool_called=None, tool_result_bytes=0,
                used_blocks=used, ttft_ms=ttft, new_prefill_tokens=new_prefill,
                e2e_s=t1 - t0, total_context_msgs=len(prefix),
            )
        )
        step_i += 1

    if result.final_answer == "":
        result.final_answer = "(trace replay complete)"
    return result
