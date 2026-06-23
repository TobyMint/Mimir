# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    MIMIR = "mimir"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.MIMIR:
        return MimirRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")


# ---- Mimir in-tree patch (Phase G + Continuum TTL 移植): mimir 调度策略 --- #
class MimirRequestQueue(FCFSRequestQueue):
    """Mimir 调度队列：Continuum 风格的 program-level FCFS + pinned-job 优先。

    逻辑移植自 Continuum 的 ContinuumRequestQueue（arXiv 2511.02230）：
    - 先看队列里有没有 job 在当前 pinned 集合中（被 TTL pin 着、下一步会复用 KV），
      有则在该批 pinned-job 中选最早入队的优先调度，让"快回来"的请求插队保连续性；
    - 否则退化为 job-level FCFS（按 job_id 首次入队时间出队）。
    pinned 集合由 scheduler 在起引擎时经 ``set_pinned_job_ids`` 注入一个可变的 set 引用。

    与 Continuum 原版的区别：原版 peek/pop 带 ``pinned_requests`` 参数、由 scheduler
    在 CONTINUUM 分支显式传参；Mimir 把 pinned 集合挂在队列上、peek/pop 保持无参签名，
    避免改动 schedule loop 中大量 peek/pop 调用点。
    """

    def __init__(self) -> None:
        super().__init__()
        # job_id -> 首次入队时间（program-level FCFS 排序键）
        self.job_id_first_entry_time: dict[str, float] = {}
        # scheduler 注入的可变 pinned job_id 集合引用（默认空——无 pin 时退化为纯 FCFS）
        self.pinned_job_ids: set[str] = set()

    def set_pinned_job_ids(self, pinned: set[str]) -> None:
        """scheduler 在 __init__ 时注入自身 pinned_requests 对应的 job_id 集合引用。"""
        self.pinned_job_ids = pinned

    def add_request(self, request: Request) -> None:
        if request.job_id is not None and request.job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.append(request)

    def prepend_request(self, request: Request) -> None:
        if request.job_id is not None and request.job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            jid = getattr(request, "job_id", None)
            if jid is not None and jid not in self.job_id_first_entry_time:
                self.job_id_first_entry_time[jid] = request.arrival_time
        self.extendleft(reversed(requests))

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        return self._select()

    def pop_request(self) -> Request:
        if not self:
            raise IndexError("pop from an empty queue")
        request = self._select()
        self.remove(request)
        return request

    def _select(self) -> Request:
        """Continuum 选择策略:pinned-job 优先 → 否则 job-level FCFS。"""
        # 1) 队列里有 job 在 pinned 集合中? 选其中最早入队的
        earliest_request: Request | None = None
        earliest_entry_time = float("inf")
        for request in self:
            jid = getattr(request, "job_id", None)
            if jid is not None and jid in self.pinned_job_ids:
                job_entry_time = self.job_id_first_entry_time.get(
                    jid, request.arrival_time)
                if job_entry_time < earliest_entry_time:
                    earliest_entry_time = job_entry_time
                    earliest_request = request
        if earliest_request is not None:
            return earliest_request

        # 2) 否则 job-level FCFS:最早首次入队的 job 的请求
        earliest_request = None
        earliest_entry_time = float("inf")
        for request in self:
            jid = getattr(request, "job_id", None)
            job_entry_time = (self.job_id_first_entry_time.get(jid, request.arrival_time)
                              if jid is not None else request.arrival_time)
            if job_entry_time < earliest_entry_time:
                earliest_entry_time = job_entry_time
                earliest_request = request
        # earliest_request 必非空(队列非空已在校验)
        assert earliest_request is not None
        return earliest_request
# ---- Mimir patch end ------------------------------------------------------ #
