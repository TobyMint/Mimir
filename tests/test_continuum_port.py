"""Continuum TTL 移植（in-tree patch）的确定性单元测试。

不依赖 GPU：直接测
1. ToolCallParser 解析 Mimir agent-loop 的 [TOOL: ...] / [FINAL: ...] 格式。
2. ToolCallEstimator.set_up_pin：无工具→0；末步不 pin；工具历史平均 ≤2s→pin 2s；
   工具历史平均 >2s→不 pin（TTL=0）。
3. ToolCallEstimator 在线学习（update_func_call_exec_time）影响 set_up_pin。
4. MimirRequestQueue 的 program-FCFS + pinned-job 优先选择。
5. Scheduler 的 pin/unpin/unpin_requests_regular 死锁兜底逻辑（用最小 Scheduler 构造）。
"""

from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")  # 跳过无 vllm 环境
from vllm.v1.core.estimate_with_func import (  # noqa: E402
    FIXED_THRESHOLD_CONTINUUM,
    ToolCallEstimator,
    ToolCallParser,
)
from vllm.v1.core.sched.request_queue import (  # noqa: E402
    MimirRequestQueue,
    SchedulingPolicy,
)

# ---------- 1. Parser ----------

def test_parser_extracts_tool_name():
    p = ToolCallParser()
    assert p.parse("Let me search.\n[TOOL: search(query='kv cache')]\n") == "search"
    assert p.parse("[TOOL: read_file(path='/a/b')]") == "read_file"


def test_parser_final_is_not_a_tool_call():
    p = ToolCallParser()
    assert p.parse("All done. [FINAL: the answer is 42]") is None
    assert p.is_final("All done. [FINAL: the answer is 42]") is True
    assert p.is_final("[TOOL: search(q='x')]") is False


# ---------- 2. set_up_pin 决策 ----------

class _FakeReq:
    """最小请求替身：只带 estimator 关心的字段。"""

    def __init__(self, *, job_id, this_func_call=None, is_last_step=None,
                 arrival_time=0.0, output_token_ids=None, request_id="r0"):
        self.job_id = job_id
        self.this_func_call = this_func_call
        self.is_last_step = is_last_step
        self.last_func_call = None
        self.arrival_time = arrival_time
        self.output_token_ids = output_token_ids or []
        self.request_id = request_id


def test_set_up_pin_no_tool_returns_zero():
    est = ToolCallEstimator(tokenizer=None)
    req = _FakeReq(job_id="j1", this_func_call=None, is_last_step=False)
    assert est.set_up_pin(req) == 0


def test_set_up_pin_unknown_tool_pins_default():
    """从未见过的工具：无历史 → pin 固定 TTL（simplified 释放版）。"""
    est = ToolCallEstimator(tokenizer=None)
    req = _FakeReq(job_id="j1", this_func_call="search", is_last_step=False)
    assert est.set_up_pin(req) == FIXED_THRESHOLD_CONTINUUM


def test_set_up_pin_slow_tool_no_pin():
    """工具历史平均执行 >2s → 不 pin（避免长期占显存）。"""
    est = ToolCallEstimator(tokenizer=None)
    # 人工塞历史：search 平均 3.0s
    est.func_call_to_exec_time["search"] = 3.0
    est.record_func_call_to_exec_time["search"] = [3.0]
    req = _FakeReq(job_id="j1", this_func_call="search", is_last_step=False)
    assert est.set_up_pin(req) == 0


def test_set_up_pin_fast_tool_pins():
    """工具历史平均 ≤2s → pin。"""
    est = ToolCallEstimator(tokenizer=None)
    est.func_call_to_exec_time["search"] = 0.5
    est.record_func_call_to_exec_time["search"] = [0.5]
    req = _FakeReq(job_id="j1", this_func_call="search", is_last_step=False)
    assert est.set_up_pin(req) == FIXED_THRESHOLD_CONTINUUM


def test_online_learning_updates_average():
    """update_func_call_exec_time 累积样本并更新均值。"""
    import time as _time

    est = ToolCallEstimator(tokenizer=None)
    # 构造一个有 departure 记录的 job 历史
    est.job_to_history["j1"] = [{"departure_time": _time.time(), "func_call": "search"}]
    # monkey-patch time.time 让 exec_time 可控（≈0.5s）
    base = _time.time()
    est.job_to_history["j1"][-1]["departure_time"] = base
    orig = _time.time
    try:
        _time.time = lambda: base + 0.5  # type: ignore[assignment]
        est.update_func_call_exec_time("j1")
    finally:
        _time.time = orig  # type: ignore[assignment]
    assert est.func_call_to_exec_time["search"] == pytest.approx(0.5)
    req = _FakeReq(job_id="j1", this_func_call="search", is_last_step=False)
    assert est.set_up_pin(req) == FIXED_THRESHOLD_CONTINUUM


# ---------- 4. MimirRequestQueue 选择 ----------

def _real_req(job_id, arrival_time, rid):
    """用真实 vLLM Request 构造一个最小请求（供队列选择）。

    Request 需要 sampling_params；用一个最小 SamplingParams 满足构造。
    """
    from vllm import SamplingParams
    from vllm.v1.request import Request

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    sp.extra_args = {"job_id": job_id}
    return Request(
        request_id=rid,
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
        eos_token_id=0,
        arrival_time=arrival_time,
    )


def test_queue_pinned_job_prioritized():
    q = MimirRequestQueue()
    q.set_pinned_job_ids(set())
    # job B 先到、但 job A 被 pin
    rb = _real_req("B", 0.0, "b1")
    ra = _real_req("A", 1.0, "a1")
    q.add_request(rb)
    q.add_request(ra)
    q.pinned_job_ids = {"A"}
    # peek 应返回 A 的请求（pinned 优先），哪怕它后到
    assert q.peek_request().job_id == "A"


def test_queue_falls_back_to_job_fcfs():
    q = MimirRequestQueue()
    q.set_pinned_job_ids(set())
    rb = _real_req("B", 0.0, "b1")
    ra = _real_req("A", 1.0, "a1")
    q.add_request(rb)
    q.add_request(ra)
    # 无 pin → 选最早入队 job 的请求
    assert q.peek_request().job_id == "B"


def test_queue_empty_raises():
    q = MimirRequestQueue()
    q.set_pinned_job_ids(set())
    with pytest.raises(IndexError):
        q.peek_request()


# ---------- 5. Scheduler pin/unpin/deadlock 兜底 ----------
# Scheduler 构造重（需要 vllm_config 等），这里只测我们能独立触达的 pin/unpin 逻辑
# 方法：构造 Scheduler 太重，改为直接验证 pin_request/unpin_request 在一个最小
# stub 上的行为（方法的副作用只依赖 self.pinned_requests / self.pinned_job_ids /
# continuum_recorder，与 Scheduler 其它状态解耦）。

class _SchedulerStub:
    """复用 Scheduler 的 pin/unpin 方法但绕过重构造：只挂需要的属性 + 绑方法。"""

    def __init__(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        self.pinned_requests = []
        self.pinned_job_ids = set()
        self.continuum_recorder = None
        self.policy = SchedulingPolicy.MIMIR
        self.kv_cache_manager = None
        self.requests = {}
        self.waiting = MimirRequestQueue()
        self.waiting.set_pinned_job_ids(self.pinned_job_ids)
        # 绑定真实方法
        self.pin_request = Scheduler.pin_request.__get__(self)
        self.unpin_request = Scheduler.unpin_request.__get__(self)
        self._unpin_job = Scheduler._unpin_job.__get__(self)
        self.unpin_requests_regular = Scheduler.unpin_requests_regular.__get__(self)
        self.pop_running_request_based_on_last_step = \
            Scheduler.pop_running_request_based_on_last_step.__get__(self)


def test_pin_and_unpin_track_job_ids():
    s = _SchedulerStub()
    r = _real_req("A", 0.0, "a1")
    s.pin_request(r, 2.0)
    assert len(s.pinned_requests) == 1
    assert "A" in s.pinned_job_ids
    # unpin
    end_time = s.pinned_requests[0][1]
    s.unpin_request(r, end_time)
    assert len(s.pinned_requests) == 0
    assert "A" not in s.pinned_job_ids


def test_unpin_job_clears_all_pins_for_job():
    s = _SchedulerStub()
    r1 = _real_req("A", 0.0, "a1")
    r2 = _real_req("A", 1.0, "a2")
    s.pin_request(r1, 2.0)
    s.pin_request(r2, 2.0)
    s._unpin_job("A")
    assert len(s.pinned_requests) == 0
    assert "A" not in s.pinned_job_ids


def test_deadlock_victim_is_latest_expiry():
    s = _SchedulerStub()
    r1 = _real_req("A", 0.0, "a1")
    r2 = _real_req("B", 0.0, "b1")
    s.pin_request(r1, 1.0)   # 早到期
    s.pin_request(r2, 5.0)   # 晚到期
    victim = s.pop_running_request_based_on_last_step(r1)
    # 选 end_time 最大的（r2）作为驱逐对象
    assert victim is not None
    assert victim[0].job_id == "B"
    assert len(s.pinned_requests) == 1
