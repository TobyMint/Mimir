"""vLLM v1 引擎适配器（InprocClient 单进程，可观测 block_pool）。

与 ``engine_vllm.py``（v0）并列。v1 默认把引擎核心跑在子进程（``SyncMPClient``），
父进程看不到 ``block_pool``/显存。我们用 ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` 切到
``InprocClient``，引擎核心在主进程内，父进程可直接遍历到 scheduler.kv_cache_manager.block_pool。

遍历路径：
    llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
                ^LLMEngine  ^InprocClient  ^EngineCore   ^Scheduler

启用前需 ``source scripts/activate_env.sh``（设 LD_LIBRARY_PATH 让 _C 找到 libtorch.so）。
"""

from __future__ import annotations

import os
from typing import Any

from mimir.engine_vllm import VLLMEngine, _dtype_bytes


def _resolve_v1_block_pool(llm: Any) -> Any | None:
    """从 v1 LLM 遍历到 BlockPool（InprocClient 下），失败返回 None。"""
    try:
        eng = llm.llm_engine  # LLMEngine (v1)
        ec = eng.engine_core  # InprocClient（单进程）或 SyncMPClient（子进程）
        inner = getattr(ec, "engine_core", ec)  # InprocClient.engine_core = EngineCore
        sched = inner.scheduler
        return sched.kv_cache_manager.block_pool
    except Exception:
        return None


def _resolve_v1_scheduler(llm: Any) -> Any | None:
    try:
        return llm.llm_engine.engine_core.engine_core.scheduler
    except Exception:
        return None


def v1_kv_usage(llm: Any) -> dict[str, Any]:
    """查询 v1 BlockPool 的 KV 块使用（优化敏感指标）。"""
    out: dict[str, Any] = {
        "used_blocks": None,
        "total_blocks": None,
        "utilization": None,
        "used_gib": None,
    }
    bp = _resolve_v1_block_pool(llm)
    if bp is None:
        return out
    try:
        total = bp.num_gpu_blocks
        free = bp.get_num_free_blocks()
        used = total - free
        out["total_blocks"] = total
        out["used_blocks"] = used
        out["utilization"] = (
            bp.get_usage() if hasattr(bp, "get_usage") else (used / total if total else None)
        )
        # best-effort 字节估算
        try:
            cc = llm.llm_engine.vllm_config.cache_config
            mc = llm.llm_engine.vllm_config.model_config
            block_size = cc.block_size
            num_layers = mc.get_num_layers()
            num_kv_heads = mc.get_total_num_kv_heads()
            head_size = mc.get_head_size()
            dpb = _dtype_bytes(getattr(mc, "dtype", "bfloat16"))
            bytes_per_token = num_layers * 2 * num_kv_heads * head_size * dpb
            out["used_gib"] = (used * block_size * bytes_per_token) / (1024**3)
        except Exception:
            pass
    except Exception:
        pass
    return out


class VLLMEngineV1(VLLMEngine):
    """vLLM v1 引擎（InprocClient 单进程），接口与 v0 ``VLLMEngine`` 一致。"""

    def _init_engine(self) -> None:
        import time

        from vllm import LLM

        # 强制 v1 单进程（必须在 import vllm 后、构造前设好；activate_env.sh 已设默认）
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        c = self.config
        kwargs: dict[str, Any] = {
            "model": c.model,
            "dtype": c.dtype,
            "gpu_memory_utilization": c.gpu_memory_utilization,
            "enable_prefix_caching": c.enable_prefix_caching,
            "max_model_len": c.max_model_len,
            "enforce_eager": c.enforce_eager,
            "tensor_parallel_size": c.tensor_parallel_size,
            "seed": c.seed,
        }
        if c.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = c.kv_cache_dtype
        # Keep v1's request-level stat pipeline alive so per-request timing
        # (arrival/first_token/scheduled/last_token) is tracked — needed for
        # TTFT observability (the in-tree Phase R patch attaches these to
        # RequestOutput.metrics). Stock LLM() force-disables log_stats; undo it.
        kwargs.setdefault("disable_log_stats", False)
        kwargs.update(c.extra)
        t0 = time.perf_counter()
        self._llm = LLM(**kwargs)
        self._engine_init_s = time.perf_counter() - t0

    def kv_usage(self) -> dict[str, Any]:
        if self._llm is None:
            return {
                "used_blocks": None,
                "total_blocks": None,
                "utilization": None,
                "used_gib": None,
            }
        return v1_kv_usage(self._llm)

    def mimir_scheduler(self) -> Any | None:
        """返回 v1 Scheduler（供 Mimir 调用 block_pool 的 mimir_* 方法 / 读 stats）。"""
        return _resolve_v1_scheduler(self._llm)

    def mimir_block_pool(self) -> Any | None:
        return _resolve_v1_block_pool(self._llm)

    def _engine_core(self) -> Any | None:
        try:
            return self._llm.llm_engine.engine_core.engine_core
        except Exception:
            return None

    def set_current_task(self, task_id: str | None) -> None:
        """设置后续请求归属的 agent 任务 id（注入 engine_core._mimir_current_task）。"""
        ec = self._engine_core()
        if ec is not None:
            ec._mimir_current_task = task_id  # noqa: SLF001

    def chat_task(
        self,
        messages: list[dict[str, str]],
        *,
        task_id: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> tuple[str, int]:
        """带任务标记的 chat：先把 task_id 设为当前，再 chat。返回 (text, n_tokens)。"""
        self.set_current_task(task_id)
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def _block_size(self) -> int:
        try:
            return int(self._llm.llm_engine.vllm_config.cache_config.block_size)
        except Exception:
            return 16

    def _compute_block_classes(self, messages: list[dict[str, str]]) -> list[str]:
        """按消息角色把 prompt 的每个块打语义类别标签（block-class 创新的核心注入）。

        用 tokenizer 把每条消息编码，按累积 token 数对齐到块边界（block_size），给落在
        该消息区间内的块打类别：
          - role=system                 -> "system"
          - role=assistant              -> "reasoning"  （模型推理中间态，低价值优先淘汰）
          - 以 "[TOOL_RESULT" 开头的 user -> "tool_result"（高价值，保留到任务结束）
          - 其余 user                    -> "user"
        返回 per-block-index 的类别列表。失败时返回空列表（block_pool 退回 "unknown"）。
        """
        try:
            tok = self._llm.get_tokenizer()  # vLLM LLM 提供
        except Exception:
            return []
        block_size = max(1, self._block_size())

        def role_class(role: str, content: str) -> str:
            if role == "system":
                return "system"
            if role == "assistant":
                return "reasoning"
            if role == "user" and content.lstrip().startswith("[TOOL_RESULT"):
                return "tool_result"
            return "user"

        # 用 chat 模板里每条消息的 token 长度近似对齐（不考虑模板拼接的少量特殊 token，
        # 误差 ≤ 1 块，对「类别感知」足够）。
        classes: list[str] = []
        cum = 0
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            try:
                n = len(tok.encode(content))
            except Exception:
                n = max(1, len(content) // 3)
            start_block = cum // block_size
            end_block = (cum + n + block_size - 1) // block_size  # ceil
            for bi in range(start_block, max(start_block + 1, end_block)):
                # 该块若已被前一条消息占满会重复——用首条覆盖它的消息定类
                while len(classes) <= bi:
                    classes.append(role_class(role, content))
                classes[bi] = role_class(role, content)
            cum += n
        return classes

    def chat_full(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Any:
        """单轮 chat，返回完整 ``RequestOutput``。注入 block-class 标签（创新核心）。"""
        # 计算 per-block 类别并经 engine_core 挂到 Request（core.py Phase-C 站点旁路读取）
        try:
            classes = self._compute_block_classes(messages)
            ec = self._engine_core()
            if ec is not None and classes:
                ec._mimir_block_classes = classes  # noqa: SLF001
            elif ec is not None:
                ec._mimir_block_classes = []  # noqa: SLF001
        except Exception:
            pass
        sp = self._make_sp(max_tokens, temperature)
        outs = self.llm.chat([messages], sp, use_tqdm=False)
        return outs[0]

    def chat_batch(
        self,
        msgs_list: list[list[dict[str, str]]],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        task_ids: list[str] | None = None,
    ) -> list[Any]:
        """真批量并发 chat：一次把 N 个请求交给 vLLM 同 batch 处理（N 请求同时 prefill/decode）。

        用于并发压测：N 个 agent 同时提交，测峰值 used_blocks / 是否退化（LRU 淘汰活跃块）/ OOM。
        不同于逐个 chat_full（同步阻塞单请求，假并发），这里 ``self.llm.chat(msgs_list)``
        让 vLLM 内部把 N 个请求放进同一 scheduling batch 真并发处理。

        ``task_ids``：每个请求的 agent task id（mimir 策略下可在请求间区分任务、触发回收）。
        注意：block-class 标签经 ``_mimir_block_classes`` 单字段注入，批量下会被覆盖——
        并发压测关心 used_blocks/退化而非分类标签，此限制可接受（Mimir 优势主要靠 lifecycle 回收）。
        """
        sp = self._make_sp(max_tokens, temperature)
        # 批量提交前给每个请求打 task_id（mimir 调度策略据此区分任务）
        ec = self._engine_core()
        if task_ids and ec is not None:
            # 批量下 _mimir_current_task 单字段只能代表"当前"——这里设为最后一个，
            # 真正的 per-request task_id 由 core.py 在 from_engine_core_request 时取该字段。
            # 并发压测场景足够（回收按 task_id 批量清理）。
            ec._mimir_current_task = task_ids[-1]  # noqa: SLF001
        outs = self.llm.chat(msgs_list, sp, use_tqdm=False)
        return list(outs)

    def mimir_finish_task(self, task_id: str) -> int:
        """任务结束：调 block_pool.mimir_finish_task 主动回收该任务 KV。返回回收块数。"""
        bp = self.mimir_block_pool()
        if bp is None:
            return 0
        fn = getattr(bp, "mimir_finish_task", None)
        if callable(fn):
            try:
                return int(fn(task_id))
            except Exception:
                return 0
        return 0

    def mimir_reclaim_evictable(self) -> int:
        """Phase J：主动回收所有 EVICTABLE（已结束任务残留）块。返回回收数。"""
        bp = self.mimir_block_pool()
        if bp is None:
            return 0
        fn = getattr(bp, "mimir_reclaim_evictable", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                return 0
        return 0

    def mimir_pin_task_blocks(self, task_id: str) -> int:
        """pin 某任务当前拥有的块（Phase E）。返回 pin 数。"""
        bp = self.mimir_block_pool()
        if bp is None:
            return 0
        get_ids = getattr(bp, "mimir_get_task_block_ids", None)
        pin = getattr(bp, "mimir_pin_blocks", None)
        if callable(get_ids) and callable(pin):
            try:
                return int(pin(get_ids(task_id)))
            except Exception:
                return 0
        return 0

    def mimir_stats(self) -> dict[str, Any]:
        """读取 scheduler 上由 in-tree patch 暴露的 Mimir 统计（Phase B）。"""
        sched = self.mimir_scheduler()
        if sched is None:
            return {}
        getter = getattr(sched, "get_mimir_stats", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return {}
        return {}
