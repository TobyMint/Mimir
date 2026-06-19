# vLLM 0.10.2 In-Tree Patch 清单（Mimir）

> 本文件记录 Mimir 对 vLLM v0.10.2 源码（普通目录 `third_party/vllm_flat`（拍平自 v0.10.2 fork）
> 分支）所做的全部 **纯 Python** in-tree patch。
> 所有 patch 不重编 `_C`（详见 [`VLLM_EDITABLE_SETUP.md`](VLLM_EDITABLE_SETUP.md)）。
> 这是「我们修改了 vLLM 内核」的可复现证据。

## 安装与复现

```bash
source scripts/activate_env.sh          # conda + LD_LIBRARY_PATH + VLLM_USE_V1=1 InprocClient
（vLLM 已拍平为 third_party/vllm_flat 普通目录，无需 submodule checkout）
```
（详见 `VLLM_EDITABLE_SETUP.md`：`source scripts/activate_env.sh` 即可。拍平目录 + .pth + dist-info，不重编 `_C`。）

## Patch 总览

| Phase | 文件 | 改动 | 验证结果 |
| --- | --- | --- | --- |
| B | `vllm/v1/core/sched/scheduler.py` | 新增 `Scheduler.get_mimir_stats()`：导出块级 KV 使用（used/total/utilization）+ Mimir 计数器 | v1 父进程可读 total_blocks=1780 |
| C | `vllm/v1/core/block_pool.py` | 新增 `mimir_block_task`/`mimir_block_lifecycle`/`mimir_used_blocks`/`mimir_lifecycle_reclaims` + `mimir_finish_task()`（任务边界主动回收）/ `mimir_pin_blocks`/`mimir_unpin_task`；`cache_full_blocks` 标记块→任务归属 | 2 个 agent 任务的 10 个 KV 块被主动回收（used_blocks 10→0） |
| C | `vllm/v1/engine/core.py` | `Request.from_engine_core_request` 后注入 `req.mimir_task_id`（从 `engine_core._mimir_current_task`） | task_id 流入 block_pool |
| D | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` 命中复用时统计跨分支 CoW 复用 → `block_pool.mimir_cow_reuses` | 4 分支测得 9 次跨分支复用 |
| E | `vllm/v1/core/block_pool.py`（Phase C 同文件） | `mimir_pin_blocks`/`mimir_unpin_task`（per-block pin） | agent_A 3/3 pinned 块在 agent_B 压力下存活 |
| F | `vllm/engine/arg_utils.py` | v1 oracle 检测 fp8 不支持时，从 `raise NotImplementedError` 改为 warn + 回退 bf16 | RTX 3090 上 fp8 请求不再崩溃，降级 bf16 跑通 |
| G | `vllm/config/scheduler.py` | `SchedulerPolicy` Literal += `"mimir"` | — |
| G | `vllm/v1/core/sched/request_queue.py` | `SchedulingPolicy.MIMIR` + `MimirRequestQueue`（FCFS 子类） | — |
| G | `vllm/v1/core/sched/scheduler.py` | 调度策略分发 `"mimir"` → `SchedulingPolicy.MIMIR` + 日志 | 引擎日志 "Mimir scheduling policy active" |
| I | `vllm/v1/core/block_pool.py` | `mimir_pin_hits` 计数器（pin 阻止回收时增） | pin 阻止回收可观测 |
| J | `vllm/v1/core/block_pool.py` | `mimir_reclaim_evictable()` 主动扫描回收所有 EVICTABLE 块（闭环：finish_task 标记 + reclaim 扫描） | 验证：finish_task 回收 3 块 used 3→0 |
| L | `vllm/v1/core/sched/scheduler.py` | mimir 策略下 `_free_blocks` 自动调 `mimir_finish_task`（任务完成即回收，自驱动） | auto_reclaim_works=True（reclaims=2，无需外部调用） |

## 与同实验室 Continuum 的区别

Continuum（`vllm-continuum`/`vllm-diff`，同为 v0.10.2 fork）做的是 **工具调用暂停时
的 KV-pin（time-bounded 估计 + whole-request）**。Mimir 在内存管理上的差异化：

| 维度 | Continuum | Mimir |
| --- | --- | --- |
| KV-pin 触发 | 解析输出中的工具调用文本 + 估计工具执行时长 | agent 轮边界（Mimir 知道何时暂停，无需文本解析） |
| pin 边界 | time-bounded（`end_time = now + est`） | lifecycle-bounded（pin 到同 agent 下一轮，无时间猜测） |
| pin 粒度 | whole-request | per-block（仅 system+history 前缀；中间 scratch 仍可淘汰） |
| 任务边界回收 | 无（pin 与淘汰独立） | **有**（Phase C `mimir_finish_task` 主动回收，Continuum 不做） |
| 与淘汰协同 | 独立 pinned 列表 | PINNED 是 `BlockLifecycle` 一态，lifecycle evictor 跳过 |
| CoW 记账 | 无 | **有**（Phase D 跨分支复用计数） |
| fp8 容错 | 无 | **有**（Phase F 优雅降级） |
| 调度策略 | `--scheduling-policy continuum`（独立队列 + 轮转） | `--scheduling-policy mimir`（FCFS 队列 + 内存管线协同） |

## Mimir 侧适配器

`src/mimir/engine_vllm_v1.py`：`VLLMEngineV1`（强制 v1 InprocClient）暴露
- `kv_usage()` / `mimir_stats()`：读 v1 block_pool + scheduler 的 Mimir 计数
- `set_current_task(task_id)` / `chat_task(...)`：注入 `engine_core._mimir_current_task`
- `mimir_finish_task(task_id)`：触发 block_pool 主动回收
- `mimir_pin_task_blocks(task_id)`：per-block pin

## 验证脚本

| 脚本 | 验证 |
| --- | --- |
| `scripts/run_phase_c_lifecycle.py` | lifecycle 主动回收（used_blocks 10→0） |
| `scripts/run_phase_d_cow.py` | CoW 跨分支复用（9 次） |
| `scripts/run_phase_e_pin.py` | per-block pin 存活（3/3） |
| `scripts/run_phase7_fp8kv.py`（改用 v1）| fp8 回退 |
| `scripts/run_phase_j_reclaim_evictable.py` | reclaim_evictable 闭环 |
| `scripts/run_phase_k_multimodel.py` | 多模型规模泛化（1.7B/4B）|
| `scripts/run_phase_m_ab.py` | 决定性 A/B（单 agent 10 轮 used 69→0） |
| `scripts/run_phase_o_concurrent.py` | 并发多 agent A/B（3 agent used 14→0）|

结果落盘：`benchmark_results/phase_{c,d,e,f,g}_*.json`。
