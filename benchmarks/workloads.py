"""典型智能体工作流生成器（纯数据，不依赖推理后端）。

为 Benchmark 提供确定性的输入序列，覆盖赛题三类场景：
- 多轮对话（上下文持续累积）
- 工具调用 / ReAct（多次 function calling，含大规模中间数据）
- 多阶段决策 / Tree-of-Thought（分支推理）

内容固定（无随机），保证优化前后可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ToolResult:
    """一次工具调用返回。``content`` 可能很大（用于测试工具数据外置）。"""

    name: str
    content: str
    tokens_approx: int = 0


@dataclass
class WorkloadCase:
    """一类工作流的可执行描述。"""

    name: str
    description: str
    system: str
    tool_schemas: list[str] = field(default_factory=list)  # 工具描述（可冗余）
    turns: list[ConversationTurn] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    branches: int = 1  # ToT 分支数
    recommended_features: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 静态内容（模拟 agent 常见冗余：system prompt + 工具描述）
# --------------------------------------------------------------------------- #

_SYSTEM_AGENT = (
    "You are a meticulous research agent. Break the user's request into steps, "
    "use the available tools when needed, and verify each intermediate result "
    "before producing the final answer. Always cite sources."
)

# 一组较长的工具 schema 描述，重复出现在多轮中，用于体现「上下文冗余 / 压缩」价值
_TOOL_SCHEMAS = [
    (
        'search(query: str, top_k: int) -> list[dict]: "Search a knowledge base '
        "and return ranked passages with title, url, snippet, score fields. "
        'Use for factual lookups."'
    ),
    (
        'python_run(code: str) -> str: "Execute a sandboxed Python snippet and '
        'return stdout/stderr (truncated to 4096 chars). Use for computation."'
    ),
    (
        'sql_query(database: str, sql: str) -> list[dict]: "Run read-only SQL on '
        'the given database, return rows as JSON. Use for structured retrieval."'
    ),
    (
        'calculator(expression: str) -> float: "Evaluate a numeric expression '
        'with units support. Use for unit conversions and arithmetic."'
    ),
]

_QUESTIONS = [
    "Summarize the latest progress on memory management for LLM agents.",
    "Compare KV cache reuse vs CPU offloading in terms of peak memory.",
    "How does prefix caching interact with multi-turn agent context?",
    "Estimate peak memory for a 7B model with 32k context at fp16.",
    "Design a Copy-on-Write scheme for branch reasoning.",
    "What are the trade-offs of disk-backed KV tiers?",
    "Propose an eviction policy that is aware of agent step boundaries.",
    "Explain why tool call results should not fully enter the KV cache.",
]


def _make_large_tool_result(idx: int) -> ToolResult:
    """构造一个「大」工具返回（模拟搜索/SQL 返回的长 JSON）。"""
    chunk = (
        '{"passage": "KV cache block reuse reduces redundant prefill '
        f'computation across turns. Segment {idx}.", "score": 0.8' + "}"
    )
    content = "[" + ", ".join([chunk] * 64) + "]"  # 故意膨胀，约数 KB
    return ToolResult(
        name="search",
        content=content,
        tokens_approx=1200,  # 粗估 token 数
    )


# --------------------------------------------------------------------------- #
# 三类工作流
# --------------------------------------------------------------------------- #


def gen_multi_turn(num_turns: int = 8) -> WorkloadCase:
    """多轮对话：system + 工具描述 + 逐轮累积的用户提问。"""
    turns = [
        ConversationTurn(role="user", content=_QUESTIONS[i % len(_QUESTIONS)])
        for i in range(num_turns)
    ]
    return WorkloadCase(
        name="multi_turn",
        description=f"多轮对话（{num_turns} 轮，上下文持续累积）",
        system=_SYSTEM_AGENT,
        tool_schemas=list(_TOOL_SCHEMAS),
        turns=turns,
        recommended_features=["prefix_cache", "lifecycle"],
    )


def gen_tool_call(num_calls: int = 6) -> WorkloadCase:
    """工具调用 / ReAct：每次提问伴随一次大工具返回。"""
    turns = [
        ConversationTurn(role="user", content=_QUESTIONS[i % len(_QUESTIONS)])
        for i in range(num_calls)
    ]
    results = [_make_large_tool_result(i) for i in range(num_calls)]
    return WorkloadCase(
        name="tool_call",
        description=f"工具调用 / ReAct（{num_calls} 次，含大返回）",
        system=_SYSTEM_AGENT,
        tool_schemas=list(_TOOL_SCHEMAS),
        turns=turns,
        tool_results=results,
        recommended_features=["tool_offload", "context_compress"],
    )


def gen_multi_stage(num_branches: int = 4, depth: int = 3) -> WorkloadCase:
    """多阶段决策 / ToT：共享前缀的多分支推理。"""
    return WorkloadCase(
        name="multi_stage",
        description=f"多阶段决策 / ToT（{num_branches} 分支 × {depth} 深度）",
        system=_SYSTEM_AGENT,
        tool_schemas=list(_TOOL_SCHEMAS),
        turns=[ConversationTurn(role="user", content=_QUESTIONS[0])],
        branches=num_branches,
        recommended_features=["branch_cow", "prefix_cache"],
    )


def all_workloads() -> dict[str, WorkloadCase]:
    return {
        "multi_turn": gen_multi_turn(),
        "tool_call": gen_tool_call(),
        "multi_stage": gen_multi_stage(),
    }
