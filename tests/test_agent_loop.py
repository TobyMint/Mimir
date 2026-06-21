"""Agent loop benchmark framework unit tests (mock engine, no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.agent_loop import (
    TASKS,
    mock_calculator,
    mock_search,
    parse_tool_call,
    run_agent_loop,
)


def test_parse_tool_call_search() -> None:
    name, arg, final = parse_tool_call('Let me search. [TOOL: search("KV cache memory")]')
    assert name == "search"
    assert arg == "KV cache memory"
    assert final is None


def test_parse_tool_call_calculator() -> None:
    name, arg, final = parse_tool_call('Let me calculate. [TOOL: calculator("7 * 32 * 1024")]')
    assert name == "calculator"
    assert arg == "7 * 32 * 1024"
    assert final is None


def test_parse_final() -> None:
    name, arg, final = parse_tool_call(
        "The answer is 3.5GB. [FINAL: Peak KV memory is 3.5GB for 7B at 32k fp16.]"
    )
    assert name is None
    assert final is not None
    assert "3.5GB" in final


def test_parse_no_match() -> None:
    name, arg, final = parse_tool_call("I need more information to answer this.")
    assert name is None
    assert final is None


def test_mock_search_size() -> None:
    result = mock_search("test query", result_size_kb=2)
    assert len(result) > 1000  # ~2KB
    assert "test query" in result
    assert result.startswith("[")


def test_mock_calculator() -> None:
    result = mock_calculator("2 + 3")
    import json

    d = json.loads(result)
    assert d["result"] == 5


class _MockEngine:
    """Mock v1 engine for agent loop testing (no GPU).

    仅用于驱动 agent loop 的多步循环逻辑（解析 tool_call / 累积消息 / crash 捕获），
    used_blocks 给单调占位值即可（不模拟回收——回收机制已删除）。
    """

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0
        self._current_task = None
        self._step = 0

    def set_current_task(self, task_id: str) -> None:
        self._current_task = task_id

    def chat(self, messages, *, max_tokens=128, temperature=0.0):
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp, len(resp.split())

    def mimir_stats(self):
        # 单调占位 used（不模拟回收）；供 agent_loop 读 used_blocks 不报错
        used = self._step * 5
        self._step += 1
        return {"used_blocks": used, "mimir_cow_reuses": 0}

    def mimir_block_pool(self):
        return SimpleNamespace(
            num_gpu_blocks=100,
            get_num_free_blocks=lambda: 100,
            mimir_block_task={},
        )


def test_agent_loop_completes_with_final() -> None:
    """Agent loop runs 2 tool calls then gives final answer."""
    responses = [
        'Let me search. [TOOL: search("KV cache memory")]',
        'Now let me calculate. [TOOL: calculator("7 * 32 * 1024")]',
        "Based on my research, the answer is clear. [FINAL: Peak KV memory is 3.5GB]",
    ]
    eng = _MockEngine(responses)
    task = TASKS[0]  # research_kv_cache
    result = run_agent_loop(eng, task, policy="mimir", tool_offload=True, max_tokens=128)
    assert result.final_answer == "Peak KV memory is 3.5GB"
    assert result.num_steps == 3
    # Step 0: tool=search, Step 1: tool=calculator, Step 2: final
    assert result.steps[0].tool_called == "search"
    assert result.steps[1].tool_called == "calculator"
    assert result.steps[2].tool_called is None  # final answer, no tool


def test_agent_loop_max_steps() -> None:
    """Agent loop reaches max_steps without final answer."""
    responses = ["I need more info but don't know what to do."] * 20
    eng = _MockEngine(responses)
    task = TASKS[1]  # compare_frameworks, max_steps=6
    result = run_agent_loop(eng, task, policy="fcfs", tool_offload=False, max_tokens=64)
    assert result.num_steps == 6  # reached max
    assert "max steps" in result.final_answer


def test_agent_loop_tool_offload_reduces_context() -> None:
    """When tool_offload=True, tool result should NOT be in context (store has it)."""
    responses = [
        'Search first. [TOOL: search("test")]',
        "Done. [FINAL: answer]",
    ]
    eng = _MockEngine(responses)
    task = TASKS[2]  # multi_step_estimate
    result = run_agent_loop(
        eng, task, policy="mimir", tool_offload=True, tool_result_kb=5, max_tokens=64
    )
    # Tool was called, result was big (5KB), but offloaded
    assert result.steps[0].tool_called == "search"
    assert result.steps[0].tool_result_bytes > 3000  # ~5KB produced
    assert result.total_tool_data_bytes > 3000
