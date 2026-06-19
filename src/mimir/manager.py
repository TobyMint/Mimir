"""统一的 MemoryManager —— 把各优化方向编排成一条 agent 内存管线。

把上下文压缩 / 工具外置 / 分层存储 / 生命周期 / 分支 CoW / 多任务协调 串起来，
对一条 ``WorkloadCase`` 跑「Mimir 完整管线」：

  workload
     │
     ├─[context_compress]→ 压缩历史轮次 + 精简工具描述
     ├─[tool_offload]→ 大工具返回外置（进分层 / 落盘），上下文留引用
     ├─[prefix_cache]→ 静态前缀去重（复用指纹）
     │
     ▼
   引擎（vLLM）—— KV 复用由 APC、共享由 CoW、回收由 lifecycle、容量由 fp8/tiered

``MemoryManager`` 是「编排 + 特性开关」，便于消融实验与渐进集成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.workloads import WorkloadCase

from mimir.context.compressor import ContextCompressor, Fidelity
from mimir.context.semantic import LLMSemanticCompressor
from mimir.kv_cache.lifecycle import LifecycleEvictor
from mimir.tiered.store import TieredStore
from mimir.tools.offload import ToolDataStore


@dataclass
class PipelineStep:
    """管线一步的执行记录。"""

    name: str
    enabled: bool
    metric: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """一次管线运行的产出：变换后的 case + 各步统计。"""

    case: WorkloadCase
    steps: list[PipelineStep] = field(default_factory=list)

    def step(self, name: str, enabled: bool, **metric: Any) -> None:
        self.steps.append(PipelineStep(name=name, enabled=enabled, metric=dict(metric)))


class MemoryManager:
    """统一内存管理入口（特性开关编排各优化方向）。

    例::

        mm = MemoryManager(features=["context_compress", "tool_offload", "tiered"])
        result = mm.apply(case)            # 返回变换后的 case + 统计
        # ...把 result.case 喂给 vLLM 引擎跑...
        mm.finish_task("task_1")          # 任务结束 -> 回收
    """

    SUPPORTED_FEATURES = frozenset(
        {
            "prefix_cache",  # 静态前缀去重（APC 在引擎层）
            "lifecycle",  # 生命周期感知淘汰
            "branch_cow",  # 分支 CoW（引擎层 + BranchTree 记账）
            "context_compress",  # 启发式上下文压缩
            "semantic_compress",  # LLM 语义压缩
            "tool_offload",  # 工具数据外置
            "tiered",  # 分层存储
            "fp8_kv",  # KV fp8 量化（引擎层）
            "multitask",  # 多任务协调
        }
    )

    def __init__(
        self,
        features: list[str] | None = None,
        *,
        fidelity: Fidelity = Fidelity.BALANCED,
        keep_recent_turns: int = 2,
        tiered_gpu_cap: int = 8,
        tiered_host_cap: int = 32,
        summarize_fn: Any = None,
    ) -> None:
        self.features: set[str] = set(features or [])
        unknown = self.features - self.SUPPORTED_FEATURES
        if unknown:
            raise ValueError(f"未知的特性开关: {sorted(unknown)}")
        self.fidelity = fidelity
        self.keep_recent_turns = keep_recent_turns
        self.summarize_fn = summarize_fn

        # 持有各模块实例（按需）
        self._tiered: TieredStore | None = None
        if "tiered" in self.features:
            self._tiered = TieredStore(gpu_cap=tiered_gpu_cap, host_cap=tiered_host_cap)
        self._offload: ToolDataStore | None = None
        if "tool_offload" in self.features:
            self._offload = ToolDataStore(tiered=self._tiered)
        self._evictor: LifecycleEvictor | None = None
        if "lifecycle" in self.features:
            self._evictor = LifecycleEvictor(capacity=tiered_gpu_cap)

    @property
    def enabled(self) -> set[str]:
        return set(self.features)

    def has(self, feature: str) -> bool:
        return feature in self.features

    def apply(self, case: WorkloadCase, *, task_id: str = "default") -> PipelineResult:
        """对一条 workload 跑完整管线，返回变换后的 case + 各步统计。"""
        result = PipelineResult(case=case)
        cur = case

        # 1) 静态前缀去重（记账：system+tools 长度，复用指纹）
        prefix_len = len(case.system) + sum(len(s) for s in case.tool_schemas)
        result.step("prefix_cache", self.has("prefix_cache"), static_prefix_chars=prefix_len)

        # 2) 语义压缩（LLM）—— 若启用，先于启发式
        if self.has("semantic_compress"):
            comp = LLMSemanticCompressor(
                summarize_fn=self.summarize_fn,
                fidelity=self.fidelity,
                keep_recent_turns=self.keep_recent_turns,
            )
            cur, st = comp.compress(cur)
            result.step(
                "semantic_compress", True, llm_calls=st.llm_calls, reduction_pct=st.reduction_pct
            )
        else:
            result.step("semantic_compress", False)

        # 3) 启发式上下文压缩
        if self.has("context_compress"):
            comp = ContextCompressor(
                fidelity=self.fidelity, keep_recent_turns=self.keep_recent_turns
            )
            cur = comp.compress(cur)
            result.step(
                "context_compress",
                True,
                original_chars=comp.stats.original_chars,
                compressed_chars=comp.stats.compressed_chars,
                reduction_pct=comp.stats.char_reduction_pct,
            )
        else:
            result.step("context_compress", False)

        # 4) 工具数据外置（若启用，重写 case.tool_results 为引用）
        if self.has("tool_offload") and self._offload is not None and cur.tool_results:
            from benchmarks.workloads import ToolResult

            new_results = []
            for r in cur.tool_results:
                ctx = self._offload.put(r.name, r.content)
                new_results.append(
                    ToolResult(name=r.name, content=ctx, tokens_approx=len(ctx) // 4)
                )
            cur = WorkloadCase(
                name=cur.name,
                description=cur.description,
                system=cur.system,
                tool_schemas=cur.tool_schemas,
                turns=cur.turns,
                tool_results=new_results,
                branches=cur.branches,
                recommended_features=cur.recommended_features,
            )
            result.step("tool_offload", True, **self._offload.stats())
        else:
            result.step("tool_offload", False)

        # 5) 分层 / 生命周期 / fp8 / CoW / 多任务 —— 记账（引擎层行为）
        result.step(
            "tiered", self.has("tiered"), **(self._tiered.stats_dict() if self._tiered else {})
        )
        result.step("lifecycle", self.has("lifecycle"))
        result.step("fp8_kv", self.has("fp8_kv"))
        result.step("branch_cow", self.has("branch_cow"))
        result.step("multitask", self.has("multitask"))

        result.case = cur
        return result

    def finish_task(self, task_id: str) -> int:
        """任务结束：触发生命周期回收。"""
        if self._evictor is not None:
            return self._evictor.finish_task(task_id)
        return 0

    def run_turn_with_engine(
        self,
        eng: Any,
        case: WorkloadCase,
        *,
        task_id: str,
        max_tokens: int = 128,
    ) -> dict[str, Any]:
        """统一一条 agent 轮：外部管线变换 + vLLM(in-tree patched) 引擎执行。

        串起两层优化：先用 ``apply()`` 做请求侧变换（压缩/外置/prefix），
        再设引擎当前任务并执行（mimir 策略下请求完成会自动回收 KV）。
        返回该轮的指标（含外部管线 step + 引擎 mimir_stats）。
        """
        # 1. 外部变换
        pr = self.apply(case, task_id=task_id)
        # 2. 引擎执行（若 eng 是 VLLMEngineV1，设当前任务）
        set_task = getattr(eng, "set_current_task", None)
        if callable(set_task):
            set_task(task_id)
        chat = getattr(eng, "chat", None)
        text, n_tok = "", 0
        if callable(chat):
            # 用变换后 case 构造多轮最后一条请求
            from benchmarks.harness import build_requests

            reqs = build_requests(pr.case, max_tokens=max_tokens, offload_store=self._offload)
            last = reqs[-1] if reqs else None
            if last is not None:
                text, n_tok = chat(last.messages, max_tokens=last.max_tokens)
        # 3. 引擎指标
        stats = {}
        get_stats = getattr(eng, "mimir_stats", None)
        if callable(get_stats):
            stats = get_stats()
        return {
            "task_id": task_id,
            "text": text[:80],
            "out_tokens": n_tok,
            "pipeline_steps": [
                {"name": s.name, "enabled": s.enabled, "metric": s.metric} for s in pr.steps
            ],
            "engine_stats": stats,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryManager(features={sorted(self.features)})"
