"""``benchmarks.harness.build_requests`` 的单元测试（纯逻辑，不依赖 vLLM / GPU）。"""

from __future__ import annotations

from benchmarks.harness import build_requests
from benchmarks.workloads import gen_multi_stage, gen_multi_turn, gen_tool_call


def test_multi_turn_builds_accumulating_requests() -> None:
    case = gen_multi_turn(num_turns=3)
    reqs = build_requests(case, max_tokens=64)
    assert len(reqs) == 3
    # 上下文应逐轮累积：每轮比上一轮多至少一条消息
    assert len(reqs[0].messages) < len(reqs[1].messages) < len(reqs[2].messages)
    # 首条消息应是 system（含工具描述）
    assert reqs[0].messages[0]["role"] == "system"
    assert "Available tools" in reqs[0].messages[0]["content"]
    assert reqs[0].max_tokens == 64


def test_tool_call_includes_large_tool_results() -> None:
    case = gen_tool_call(num_calls=2)
    reqs = build_requests(case)
    # 应存在 role=tool 的消息，且内容较大（用于体现外置价值）
    tool_msgs = [m for r in reqs for m in r.messages if m["role"] == "tool"]
    assert len(tool_msgs) >= 2
    assert all(len(m["content"]) > 100 for m in tool_msgs)


def test_multi_stage_branches_share_prefix() -> None:
    case = gen_multi_stage(num_branches=4)
    reqs = build_requests(case)
    assert len(reqs) == 4
    # 所有分支共享相同的前缀消息
    first = reqs[0].messages
    assert all(r.messages == first for r in reqs)
    assert first[0]["role"] == "system"
    assert first[1]["role"] == "user"


def test_max_tokens_propagated() -> None:
    case = gen_multi_turn(num_turns=2)
    reqs = build_requests(case, max_tokens=128)
    assert all(r.max_tokens == 128 for r in reqs)
