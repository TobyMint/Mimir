# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mimir in-tree patch: tool-call-boundary TTL 估计器（移植自 Continuum）。

来源：Continuum（arXiv 2511.02230, Hanchen Li et al., UC Berkeley），
``Hanchenli/vllm-continuum`` 释放版（simplified，不带论文 cost-benefit CDF 估算，
固定 TTL 阈值）。我们移植其核心机制并入 Mimir 的 ``"mimir"`` 调度策略——
工具调用暂停时给 KV 挂 TTL 保留，同一 job 下一步进来时 KV 还在、免重 prefill。

本文件是 Continuum 的 ``vllm/v1/core/estimate_with_func.py`` 的照搬 + 轻量适配；
原作者 ``# NOTE (Hanchen)`` / ``# TODO (Hanchen)`` 注释保留以利追溯。
Mimir 侧适配：adapter（``engine_vllm_v1``）直接在 ``SamplingParams.extra_args``
里传 ``this_func_call``，故 ``set_up_pin`` 走 ``request.this_func_call``，
``ToolCallParser``（mini-swe-agent 专用 bash-codeblock 解析）保留但默认不走。
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer
from vllm.v1.request import Request

logger = init_logger(__name__)

FIXED_THRESHOLD_CONTINUUM = 2.0  # seconds


class Continuum_Recorder:
    """Continuum 调度事件记录器（移植自 Continuum，用于离线分析 JSON 落盘）。"""

    def __init__(self):
        self.job_id_to_history = {}
        # Track scheduling operation timing
        self.scheduling_times = []  # List of {start_time, end_time, duration}

    def print_history(self):
        import json

        # Per-run output directory (set by launcher); fallback to default
        output_dir = os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")
        os.makedirs(output_dir, exist_ok=True)

        # Atomic write to avoid partial reads by other processes
        final_path = os.path.join(output_dir, "scheduler_timestamps")
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.job_id_to_history, f, indent=2)
        os.replace(tmp_path, final_path)

    def request_arrives(self, request: Request):
        if request.job_id not in self.job_id_to_history:
            self.job_id_to_history[request.job_id] = []
        self.job_id_to_history[request.job_id].append(
            {"Request_arrival_time": time.time()})

    def request_finished(self, request: Request):
        self.job_id_to_history[request.job_id].append(
            {"Request_departure_time": time.time()})

    def request_evicted_from_running_queue(self, request: Request):
        self.job_id_to_history[request.job_id].append(
            {"Request_evicted_from_running_queue_time": time.time()})

    def request_pinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"pinned_time": time.time()})

    def request_unpinned(self, request: Request):
        self.job_id_to_history[request.job_id].append({"unpinned_time": time.time()})

    def request_waiting_to_running(self, request: Request, prompt_length: int,
                                   hit_length: int = 0):
        self.job_id_to_history[request.job_id].append({
            "waiting_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })

    def request_evicted_to_running(self, request: Request, prompt_length: int,
                                   hit_length: int):
        self.job_id_to_history[request.job_id].append({
            "evicted_to_running": time.time(),
            "prompt_length": prompt_length,
            "hit_length": hit_length
        })


class ToolCallParser:
    """Parser for extracting function calls from LLM output.

    Uses the same parsing logic as mini-swe-agent to extract bash commands
    from markdown code blocks and identify the function call.

    This can be extended for different datasets with different parsing logic.
    NOTE (Hanchen): Mimir 适配——在 mini-swe-agent 的 bash-codeblock 之外，
    额外支持 Mimir agent-loop 的 ``[TOOL: name(args)]`` 与 ``[FINAL: ...]`` 格式。
    """

    # Mimir agent-loop 格式（[TOOL: name(args)] / [FINAL: answer]）
    _TOOL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\(([^)]*)\)\s*\]")
    _FINAL_RE = re.compile(r"\[FINAL:\s*(.*?)\]", re.DOTALL)

    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        Returns the function call name, or None if not found (含末步 [FINAL:]）。
        """
        # Mimir agent-loop 工具调用
        m = self._TOOL_RE.search(text)
        if m:
            return m.group(1)
        # 原始 Continuum mini-swe-agent 路径：```bash ... ``` 解析首个命令词
        actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)
        if len(actions) == 1:
            bash_action = actions[0].strip()
            words = bash_action.split()
            if words:
                return words[0]
        return None

    def is_final(self, text: str) -> bool:
        """是否为程序末步（Mimir agent-loop 的 [FINAL: ...]）。"""
        return self._FINAL_RE.search(text) is not None


class ToolCallEstimator:
    def __init__(
        self,
        tokenizer: Optional[AnyTokenizer] = None,
        model_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_revision: Optional[str] = None,
        parser: Optional[ToolCallParser] = None,
    ):
        self.func_call_to_exec_time: dict[str, float] = {}
        self.record_func_call_to_exec_time: dict[str, list[float]] = {}

        self.job_to_history: dict[str, list[dict[str, float]]] = {}

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model_name is not None:
            try:
                self.tokenizer = get_tokenizer(
                    tokenizer_name=model_name,
                    tokenizer_mode=tokenizer_mode,
                    trust_remote_code=trust_remote_code,
                    revision=tokenizer_revision,
                )
                logger.info(f"Initialized tokenizer for model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer for {model_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize parser (can be customized for different datasets)
        self.parser = parser if parser is not None else ToolCallParser()

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        """Mimir adapter 在引擎起好后把已加载的 tokenizer 注入，启用 detokenize-parse 路径。"""
        self.tokenizer = tokenizer

    def get_func_call_exec_time(self, func: str) -> Optional[float]:
        if func not in self.func_call_to_exec_time:
            return None
        return self.func_call_to_exec_time[func]

    #TODO Hanchen This is currently just an average
    def update_func_call_exec_time(self, job_id: str) -> None:
        #this is called when the func call is back again in scheduler.py, update the exec time with last_func_call
        last_departure_time = self.job_to_history[job_id][-1]["departure_time"]
        func = self.job_to_history[job_id][-1]["func_call"]
        exec_time = time.time() - last_departure_time

        if func not in self.record_func_call_to_exec_time:
            self.record_func_call_to_exec_time[func] = [exec_time]
        else:
            self.record_func_call_to_exec_time[func].append(exec_time)
        self.func_call_to_exec_time[func] = (sum(
            self.record_func_call_to_exec_time[func])
            / len(self.record_func_call_to_exec_time[func]))
        return

    def _parse_func_call(self, request: Request) -> Optional[str]:
        """从请求输出解析工具名（Mimir 适配：支持 adapter 直传 + detokenize-parse 双路径）。

        1) 若 request.this_func_call 已由 adapter 经 extra_args 传入 → 直接用；
        2) 否则用 tokenizer detokenize 输出 + parser 解析（兼容 Continuum 原始路径）。
        返回工具名；None 表示无工具调用（可能是末步 / 无 tool 文本）。
        """
        if getattr(request, "this_func_call", None) is not None:
            return request.this_func_call
        if self.tokenizer is None or len(request.output_token_ids) == 0:
            return None
        try:
            output_text = self.tokenizer.decode(
                request.output_token_ids, skip_special_tokens=True)
            return self.parser.parse(output_text)
        except Exception as e:
            logger.warning(
                f"Error detokenizing/parsing output for request {request.request_id}: {e}"
            )
            return None

    #Functions below will be called by outside functions
    def set_up_pin(self, request: Request) -> float:
        if request.this_func_call is None:
            return 0

        this_func_call_exec_time = (self.get_func_call_exec_time(
            request.this_func_call) or 0.0)

        if this_func_call_exec_time > FIXED_THRESHOLD_CONTINUUM:
            return 0

        return FIXED_THRESHOLD_CONTINUUM

    def request_arrives(self, request: Request) -> None:
        logger.info(f"Request job id arriving: {request.job_id}, time is {time.time()}")
        # this is called when a job arrives in scheduler.py, if job is new, create an entry,
        if request.job_id not in self.job_to_history:
            self.job_to_history[request.job_id] = []
            assert request.last_func_call is None
            self.job_to_history[request.job_id].append(
                {"arrival_time": request.arrival_time})
            return
        request.last_func_call = self.job_to_history[request.job_id][-1]["func_call"]
        logger.info(
            f"Request job id: {request.job_id}, last func call: {request.last_func_call}"
        )

        self.update_func_call_exec_time(request.job_id)

        self.job_to_history[request.job_id].append({"arrival_time": request.arrival_time})
        return

    def request_finished(self, request: Request) -> None:
        logger.info(f"Request job id finishing: {request.job_id}, time is {time.time()}")
        if request.job_id is None:
            # 非 agent 程序请求（无 job_id）——不参与 TTL 记账，直接返回。
            return

        # 解析工具调用（adapter 直传优先，否则 detokenize-then-parse）。
        # 与 Continuum 原版不同：原版只在 tokenizer 可用时解析；Mimir 即使
        # tokenizer 不可用也尊重 adapter 已传的 this_func_call。
        this_func_call = self._parse_func_call(request)
        request.this_func_call = this_func_call

        # 末步判定：adapter 已标 is_last_step=True 则尊重；否则若输出含 [FINAL:]
        # 也判为末步（Mimir agent-loop 终结标记）。
        if not request.is_last_step:
            if self.tokenizer is not None and len(request.output_token_ids) > 0:
                try:
                    output_text = self.tokenizer.decode(
                        request.output_token_ids, skip_special_tokens=True)
                    if self.parser.is_final(output_text):
                        request.is_last_step = True
                except Exception:
                    pass

        if this_func_call:
            logger.info(f"Extracted func_call: {this_func_call} from output")

        self.job_to_history[request.job_id].append({
            "departure_time": time.time(),
            "func_call": request.this_func_call
        })
        return
