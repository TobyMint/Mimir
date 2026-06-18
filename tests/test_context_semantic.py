"""``mimir.context.semantic`` 单元测试（纯逻辑，无 LLM）。"""

from __future__ import annotations

from benchmarks.workloads import ConversationTurn, WorkloadCase

from mimir.context.compressor import Fidelity
from mimir.context.semantic import LLMSemanticCompressor


def _case(n: int) -> WorkloadCase:
    return WorkloadCase(
        name="multi_turn",
        description="t",
        system="sys",
        turns=[ConversationTurn(role="user", content=f"question {i} " * 40) for i in range(n)],
    )


def test_lossless_returns_unchanged() -> None:
    case = _case(6)
    comp = LLMSemanticCompressor(fidelity=Fidelity.LOSSLESS)
    out, st = comp.compress(case)
    assert out is case
    assert st.llm_calls == 0
    assert st.reduction_pct == 0.0


def test_balanced_summarizes_old_turns_truncation_fallback() -> None:
    case = _case(6)
    comp = LLMSemanticCompressor(
        fidelity=Fidelity.BALANCED, keep_recent_turns=2, turns_per_summary=2
    )
    out, st = comp.compress(case)
    # 旧 4 轮被分组摘要（2 组），近 2 轮保留
    assert st.summarized_turn_groups == 2
    assert len(out.turns) == 3  # 1 summary + 2 recent
    assert out.turns[0].role == "system"  # 摘要作为 system 前缀
    assert st.original_chars > st.summarized_chars


def test_summarize_fn_called_and_cached() -> None:
    calls = []

    def fake_summarize(prompt: str) -> str:
        calls.append(prompt)
        return "COMPACT SUMMARY"

    case = _case(6)
    comp = LLMSemanticCompressor(
        summarize_fn=fake_summarize,
        fidelity=Fidelity.BALANCED,
        keep_recent_turns=2,
        turns_per_summary=2,
    )
    _out, st = comp.compress(case)
    assert st.llm_calls == 2  # 2 组
    assert st.cache_hits == 0
    # 再跑一次同样输入 -> 命中缓存
    case2 = _case(6)
    comp.compress(case2)
    assert comp.stats.cache_hits == 2


def test_aggressive_keeps_fewer_recent() -> None:
    case = _case(8)
    bal = LLMSemanticCompressor(fidelity=Fidelity.BALANCED, keep_recent_turns=3)
    agg = LLMSemanticCompressor(fidelity=Fidelity.AGGRESSIVE, keep_recent_turns=3)
    out_b, _ = bal.compress(case)
    out_a, _ = agg.compress(case)
    # aggressive 保留更少近轮 -> 摘要更多 -> 输出轮数更少
    assert len(out_a.turns) <= len(out_b.turns)


def test_short_conversation_not_summarized() -> None:
    case = _case(2)
    comp = LLMSemanticCompressor(fidelity=Fidelity.BALANCED, keep_recent_turns=2)
    out, _ = comp.compress(case)
    assert out is case  # 轮数 <= keep_recent，不摘要
