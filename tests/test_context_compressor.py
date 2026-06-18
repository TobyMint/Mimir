"""``mimir.context.compressor`` 单元测试（纯逻辑，不依赖 GPU）。"""

from __future__ import annotations

from benchmarks.workloads import (
    gen_multi_turn,
    gen_tool_call,
)

from mimir.context.compressor import (
    ContextCompressor,
    Fidelity,
    _summarize_tool_result,
    compress_workload,
)


def test_lossless_does_not_compress() -> None:
    case = gen_tool_call(num_calls=4)
    out, stats = compress_workload(case, fidelity=Fidelity.LOSSLESS)
    # LOSSLESS：字符数不变，无工具摘要
    assert stats.compressed_chars == stats.original_chars
    assert stats.tool_results_summarized == 0
    assert len(out.turns) == len(case.turns)


def test_balanced_compresses_old_turns_and_tool_results() -> None:
    case = gen_tool_call(num_calls=5)
    c = ContextCompressor(fidelity=Fidelity.BALANCED, keep_recent_turns=2)
    out = c.compress(case)
    # 应有压缩
    assert c.stats.compressed_chars < c.stats.original_chars
    assert c.stats.char_reduction_pct > 0
    # 近 2 轮原文保留（未被摘要）
    assert out.turns[-1].content == case.turns[-1].content
    assert out.turns[-2].content == case.turns[-2].content
    # 至少一个工具返回被精简
    assert c.stats.tool_results_summarized >= 1
    # 精简后的工具返回更短
    assert len(out.tool_results[0].content) <= len(case.tool_results[0].content)


def test_aggressive_compresses_more_than_balanced() -> None:
    case = gen_multi_turn(num_turns=6)
    bal = ContextCompressor(fidelity=Fidelity.BALANCED, keep_recent_turns=2)
    agg = ContextCompressor(fidelity=Fidelity.AGGRESSIVE, keep_recent_turns=2)
    bal.compress(case)
    agg.compress(case)
    assert agg.stats.compressed_chars <= bal.stats.compressed_chars


def test_summarize_tool_result_json_list() -> None:
    import json as _json

    items = [{"name": f"x{i}", "score": 0.9, "snippet": "lorem ipsum " * 8} for i in range(8)]
    items.append({"name": "y", "url": "u"})
    big = _json.dumps(items)
    s = _summarize_tool_result(big, max_chars=120)
    assert len(s) <= 200
    assert "list" in s  # 结构化标记


def test_summarize_tool_result_short_passthrough() -> None:
    short = '{"a": 1}'
    assert _summarize_tool_result(short, max_chars=240) == short


def test_static_prefix_preserved_across_fidelity() -> None:
    case = gen_multi_turn(num_turns=4)
    for f in Fidelity:
        _out, stats = compress_workload(case, fidelity=f)
        assert stats.static_prefix_chars > 0  # system + tool schemas
