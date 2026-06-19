"""真实 Agent Loop Benchmark 框架。

不是预构造消息，而是让 LLM 真正驱动一个多步任务循环：
  LLM 生成 → 解析 tool_call → 执行 mock 工具 → 结果喂回 → LLM 再生成 → ...

每步的上下文真实增长（上一步 LLM 输出 + 工具结果累积），显存压力真实。
工具返回大小可控（模拟不同压力场景）。

A/B 对照：同一任务/种子/卡
  - native: fcfs, 工具结果全量进 KV, 不回收
  - Mimir:  mimir 策略 + tool_offload + 每步自动回收

指标（逐步记录）：used_blocks / ttft_ms / e2e_latency / tool_data_bytes
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStepResult:
    """一步 agent loop 的结果。"""

    step: int
    text: str
    tool_called: str | None
    tool_result_bytes: int
    used_blocks: int
    ttft_ms: float | None
    e2e_s: float
    total_context_msgs: int


@dataclass
class AgentRunResult:
    """一次完整 agent run 的结果。"""

    label: str
    policy: str
    tool_offload: bool
    steps: list[AgentStepResult] = field(default_factory=list)
    final_answer: str = ""
    peak_used_blocks: int = 0
    total_tool_data_bytes: int = 0

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "policy": self.policy,
            "tool_offload": self.tool_offload,
            "num_steps": self.num_steps,
            "peak_used_blocks": self.peak_used_blocks,
            "total_tool_data_bytes": self.total_tool_data_bytes,
            "final_answer": self.final_answer[:200],
            "steps": [
                {
                    "step": s.step,
                    "text": s.text[:100],
                    "tool_called": s.tool_called,
                    "tool_result_bytes": s.tool_result_bytes,
                    "used_blocks": s.used_blocks,
                    "ttft_ms": s.ttft_ms,
                    "e2e_s": round(s.e2e_s, 3),
                    "context_msgs": s.total_context_msgs,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Mock 工具：返回可控大小的结构化数据
# ---------------------------------------------------------------------------


def mock_search(query: str, result_size_kb: int = 5) -> str:
    """模拟搜索工具：返回 result_size_kb 的 JSON 结果。"""
    n = max(1, result_size_kb * 10)  # ~100 bytes per entry
    entries = [
        json.dumps(
            {
                "title": f"Search result {i} for '{query[:30]}'",
                "url": f"https://example.com/result/{i}",
                "snippet": f"This result discusses {query[:50]} with relevant details. " * 3,
                "score": round(0.95 - i * 0.05, 2),
            }
        )
        for i in range(n)
    ]
    return "[" + ", ".join(entries) + "]"


def mock_calculator(expression: str) -> str:
    """模拟计算器工具。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"expression": expression, "result": result})
    except Exception:
        return json.dumps({"expression": expression, "error": "invalid expression"})


MOCK_TOOLS: dict[str, Any] = {
    "search": mock_search,
    "calculator": mock_calculator,
}

TOOL_DESCRIPTIONS = [
    "search(query, result_size_kb) -> list[dict]: Search a knowledge base.",
    "calculator(expression: str) -> dict: Evaluate a numeric expression and return the result.",
]

# 任务集：每个任务是一个 system prompt + 一个 user 请求
TASKS = [
    {
        "name": "research_kv_cache",
        "system": (
            "You are a research agent. Use the search tool to find information, "
            "then synthesize an answer. Call tools by writing "
            "[TOOL: search('query')] or [TOOL: calculator('expr')]. "
            "When you have enough info, write [FINAL: your answer]."
        ),
        "user": (
            "Research how KV cache memory management works in LLM inference, "
            "then estimate the peak KV memory for a 7B model at 32k context "
            "in fp16."
        ),
        "expected_tools": ["search", "calculator"],
        "max_steps": 8,
    },
    {
        "name": "compare_frameworks",
        "system": (
            "You are a technical analyst. Use the search tool to gather data, "
            "then compare. Call tools by writing [TOOL: search('query')]. "
            "When done, write [FINAL: your comparison]."
        ),
        "user": (
            "Compare vLLM and llama.cpp in terms of memory efficiency and "
            "agent support. Search for relevant information first."
        ),
        "expected_tools": ["search"],
        "max_steps": 6,
    },
    {
        "name": "multi_step_estimate",
        "system": (
            "You are a systems engineer. Use search to gather specs, use "
            "calculator for math. Call tools by writing "
            "[TOOL: search('query')] or [TOOL: calculator('expr')]. "
            "When done, write [FINAL: your estimate]."
        ),
        "user": (
            "Estimate how many concurrent 4B-model agents can run on a single "
            "24GB GPU at 8k context. Search for model specs, then calculate."
        ),
        "expected_tools": ["search", "calculator"],
        "max_steps": 10,
    },
]


# 解析 LLM 输出中的 tool_call
TOOL_CALL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\(([^)]*)\)\s*\]")
FINAL_RE = re.compile(r"\[FINAL:\s*(.*?)\]", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str | None, str | None, str | None]:
    """从 LLM 输出解析 tool_call 和 final answer。

    Returns:
        (tool_name, tool_arg, final_answer) — tool_name 为 None 表示无工具调用，
        final_answer 为 None 表示未完成。
    """
    final_match = FINAL_RE.search(text)
    if final_match:
        return None, None, final_match.group(1).strip()

    tool_match = TOOL_CALL_RE.search(text)
    if tool_match:
        name = tool_match.group(1)
        arg = tool_match.group(2).strip().strip("'\"")
        return name, arg, None

    return None, None, None


def run_agent_loop(
    eng: Any,
    task: dict[str, Any],
    *,
    policy: str = "mimir",
    tool_offload: bool = True,
    tool_result_kb: int = 5,
    max_tokens: int = 128,
) -> AgentRunResult:
    """跑一个真实的 agent loop。

    LLM 生成 → 解析 tool_call → 执行 mock 工具 → 结果喂回 → 再生成。
    每步记录 used_blocks / ttft / e2e。
    """
    from mimir.tools.offload import ToolDataStore

    store = ToolDataStore() if tool_offload else None
    label = f"{task['name']}_{policy}"
    result = AgentRunResult(label=label, policy=policy, tool_offload=tool_offload)

    system = task["system"]
    # 把工具描述附加到 system prompt
    system += "\n\nAvailable tools:\n" + "\n".join(TOOL_DESCRIPTIONS)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["user"]},
    ]

    max_steps = task.get("max_steps", 8)
    import time

    for step_i in range(max_steps):
        # 设当前步骤的 task_id（mimir 策略下会自动回收上一步的 KV）
        task_id = f"{task['name']}_step_{step_i}"
        set_task = getattr(eng, "set_current_task", None)
        if callable(set_task):
            set_task(task_id)

        # LLM 生成
        t0 = time.perf_counter()
        text, n_tok = eng.chat(messages, max_tokens=max_tokens, temperature=0.0)
        t1 = time.perf_counter()

        # 读引擎统计
        stats = {}
        get_stats = getattr(eng, "mimir_stats", None)
        if callable(get_stats):
            stats = get_stats()

        used = stats.get("used_blocks", 0) or 0
        result.peak_used_blocks = max(result.peak_used_blocks, used)

        # 解析 tool_call / final
        tool_name, tool_arg, final = parse_tool_call(text)
        step_result = AgentStepResult(
            step=step_i,
            text=text[:200],
            tool_called=tool_name,
            tool_result_bytes=0,
            used_blocks=used,
            ttft_ms=stats.get("first_ttft_ms") or None,
            e2e_s=t1 - t0,
            total_context_msgs=len(messages),
        )

        if final is not None:
            # 任务完成
            result.final_answer = final
            result.steps.append(step_result)
            break

        if tool_name is not None and tool_name in MOCK_TOOLS:
            # 执行 mock 工具
            if tool_name == "search":
                tool_result = MOCK_TOOLS[tool_name](tool_arg or "", result_size_kb=tool_result_kb)
            else:
                tool_result = MOCK_TOOLS[tool_name](tool_arg or "")
            step_result.tool_result_bytes = len(tool_result)
            result.total_tool_data_bytes += len(tool_result)

            # 工具外置：大结果存 store，上下文只放引用
            if store is not None:
                ctx_content = store.put(tool_name, tool_result)
            else:
                ctx_content = tool_result

            # 把 LLM 输出和工具结果加入上下文
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "tool", "content": ctx_content})
        else:
            # LLM 没调工具也没给 final — 把它的输出加入上下文继续
            messages.append({"role": "assistant", "content": text})
            # 如果连续两步没调工具也没 final，强制提示
            if step_i >= 1 and result.steps and result.steps[-1].tool_called is None:
                messages.append(
                    {"role": "user", "content": "Please use a tool or provide [FINAL: answer]."}
                )

        result.steps.append(step_result)
    else:
        # 达到 max_steps 仍未完成
        result.final_answer = "(max steps reached without final answer)"

    return result
