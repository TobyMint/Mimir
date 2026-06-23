# vLLM 0.10.2 In-Tree Patch 清单（Mimir）

> 本文件记录 Mimir 对 vLLM v0.10.2 源码（普通目录 `third_party/vllm`（拍平自 v0.10.2 fork）
> 分支）所做的全部 **纯 Python** in-tree patch。
> 所有 patch 不重编 `_C`（详见 [`VLLM_EDITABLE_SETUP.md`](VLLM_EDITABLE_SETUP.md)）。
> 这是「我们修改了 vLLM 内核」的可复现证据。

## 安装与复现

```bash
source scripts/activate_env.sh          # conda + LD_LIBRARY_PATH + VLLM_USE_V1=1 InprocClient
（vLLM 已拍平为 third_party/vllm 普通目录，无需 submodule checkout）
```
（详见 `VLLM_EDITABLE_SETUP.md`：`source scripts/activate_env.sh` 即可。拍平目录 + .pth + dist-info，不重编 `_C`。）

## Patch 总览

> **重要更新**：原 lifecycle 主动回收机制（Phase C/E/I/J/L 的 `mimir_finish_task` / per-block pin / `reclaim_evictable` / mimir 策略自驱动回收）经自审系「used_blocks→0 偷换概念」（推理时该占仍占、回收重算拖慢服务），**已从代码彻底删除**。下表保留这些条目仅为可追溯，标注【已删】。保留的真 patch：B（统计）/ D（CoW）/ F（fp8 降级）/ R（TTFT 回填）/ Continuum-TTL（已 port）/ BC（block-class，已弃用为核心，留备查）。
>
> **核心方向（三篇融合,均已落地）**：工具调用边界 TTL 保留（Continuum，已 port）+ LMCache 分层 offload（已集成）+ CacheGen KV 编解码压缩（已验证集成），组成 agent 工具调用场景下的统一 KV 放置管线（见下「三篇融合」「融合进度」与 [`技术方案.md`](技术方案.md) §3.7）。不自造机制,做三篇融合 + 场景化 + 诚实归因。

| Phase | 文件 | 改动 | 验证结果 |
| --- | --- | --- | --- |
| B | `vllm/v1/core/sched/scheduler.py` | 新增 `Scheduler.get_mimir_stats()`：导出块级 KV 使用（used/total/utilization） | v1 父进程可读 total_blocks |
| ~~C~~【已删】 | `vllm/v1/core/block_pool.py` | 原 `mimir_finish_task` 任务边界主动回收——已删除（used_blocks→0 偷换） | （已删，见顶部说明） |
| C | `vllm/v1/engine/core.py` | `Request.from_engine_core_request` 后注入 `req.mimir_task_id`（CoW 记账用，非回收） | task_id 流入 block_pool |
| D | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` 命中复用时统计跨分支 CoW 复用 → `block_pool.mimir_cow_reuses` | 4 分支测得 9 次跨分支复用 |
| ~~E~~【已删】 | `vllm/v1/core/block_pool.py` | 原 per-block pin——已删除 | （已删） |
| F | `vllm/engine/arg_utils.py` | v1 oracle 检测 fp8 不支持时，从 `raise NotImplementedError` 改为 warn + 回退 bf16 | RTX 3090 上 fp8 请求不再崩溃，降级 bf16 跑通 |
| G | `vllm/config/scheduler.py` | `SchedulerPolicy` Literal += `"mimir"` | — |
| G | `vllm/v1/core/sched/request_queue.py` | `SchedulingPolicy.MIMIR` + `MimirRequestQueue`（FCFS 子类） | — |
| G | `vllm/v1/core/sched/scheduler.py` | 调度策略分发 `"mimir"` → `SchedulingPolicy.MIMIR` + 日志 | 引擎日志 "Mimir scheduling policy active" |
| ~~I~~【已删】 | `vllm/v1/core/block_pool.py` | 原 `mimir_pin_hits` 计数器——已删 | （已删） |
| ~~J~~【已删】 | `vllm/v1/core/block_pool.py` | 原 `mimir_reclaim_evictable()`——已删 | （已删） |
| ~~L~~【已删】 | `vllm/v1/core/sched/scheduler.py` | 原 mimir 策略自驱动回收——已删 | （已删） |
| R | `vllm/v1/engine/output_processor.py` | `_new_request_output` 用 v1 `RequestState.stats`（arrival/first_token/scheduled/last_token）构造 `RequestMetrics` 挂到 `RequestOutput.metrics`（v1 原本恒为 None，无法观测 TTFT） | v1 每请求 TTFT/prefill/e2e 可读；agent-loop 每步 ttft 真实落盘 |
| R | `src/mimir/engine_vllm_v1.py` | 构造时 `disable_log_stats=False`（v1 `LLM()` 默认强制 `True`，会关闭整个 stat pipeline，使 `RequestState.stats=None`） | 配合上面 patch，stats pipeline 保活 → TTFT 可观测 |
| **BC** | `vllm/v1/core/block_pool.py` | **【创新核心】** 新增 `mimir_block_class`（block_id→语义类别）+ `mimir_class_aware_evict()`（按 `reasoning>user>tool_result>system` 优先级主动淘汰，对比 vLLM 原生 LRU 盲选）；`cache_full_blocks` 给每个缓存块打类别标签；`get_new_blocks` 容量紧张时先 `mimir_class_aware_evict` | 演示：20 system/114 reasoning/99 tool_result 块，evict(57) 只淘汰 reasoning 57、tool_result/system 0 存活；5 个确定性单测覆盖优先级与 ref_cnt 守卫 |
| **BC** | `vllm/v1/engine/core.py` | 注入 `req.mimir_block_classes`（从 `engine_core._mimir_block_classes`） | adapter 计算的 per-block 类别流入 block_pool |
| **BC** | `src/mimir/engine_vllm_v1.py` | 新增 `chat_full()` 重写 + `_compute_block_classes()`：按消息角色（system/assistant→reasoning/`[TOOL_RESULT`→tool_result/user）把 prompt 块对齐打类别 | 真实 Qwen3-4B 上标签注入成功（block_class_counts 可读） |
| **BC** | `vllm/v1/core/sched/scheduler.py` | `get_mimir_stats()` 附加 `mimir_class_stats()`（类别块数 + 按类别淘汰数） | 类别感知淘汰可观测、可报告 |

## 三篇融合：Continuum + LMCache + CacheGen

Mimir **不自造机制，做三篇融合**——把三篇互补的 prior work 在 agent 工具调用场景下串成统一 KV 放置管线（详见 [`技术方案.md`](技术方案.md) §3.7）：

| 环节 | 论文 | 贡献 | 引擎 |
| --- | --- | --- | --- |
| **何时**留/放 | Continuum（arXiv 2511.02230） | 工具调用边界 + TTL 保留 + cost-benefit | vLLM 0.10.2 |
| **搬去哪/怎么搬** | LMCache（arXiv 2510.09665） | GPU↔CPU↔磁盘分层 offload、layer-wise 重叠、pin/lookup | vLLM 0.10.2 |
| **怎么压/怎么传得快** | CacheGen（arXiv 2310.07240, SIGCOMM'24） | delta+分层量化+算术编码压 KV 成 bitstream（3.5–4.3× 压） | HF transformers |

**融合管线**：工具调用暂停 → Continuum TTL pin KV → 显存紧 → LMCache offload 到 CPU（**CacheGen 编码成 bitstream**）→ 下一步回来 → LMCache reload + CacheGen 解码 → 快于重 prefill、也快于搬原始张量。**三篇无人串成这条 agent 工具调用管线**——Mimir 贡献：集成 + 场景化 + 诚实定位。CacheGen 是关键拼图：naive Continuum+LMCache 搬未压缩张量慢，CacheGen 压缩使低带宽下 reload 仍赢 prefill。

KVFlow（arXiv 2507.07400）走第四轴（"哪个 agent 该淘汰"，steps-to-execution），作 related work 提及，不做集成。早期 block-class「按语义角色淘汰」经重审不再作核心创新（见 §3.7 注），代码暂留备查。

> 注：此前文档「无论文做过 agent 感知 KV 淘汰」的表述不准确——KVFlow/Continuum 均为公开 prior work。

## 融合进度：三篇均已落地验证

| 来源 | 内容 | 状态 |
| --- | --- | --- |
| Continuum（`Hanchenli/vllm-continuum`，v0.10.2 fork） | 工具调用边界 TTL 保留：`Request` 加 `job_id`/`is_last_step`/`this_func_call` 字段（搭便车走 vanilla `SamplingParams.extra_args`）、`ToolCallEstimator`（simplified 释放版固定 2.0s TTL）、`pin_request`/`unpin_requests_regular`/死锁兜底、program-level FCFS。并入 `"mimir"` 策略，对外仍叫 Mimir。 | **已 port 验证**（commit d9070f3，13 单测 + 真引擎冒烟） |
| LMCache（`LMCache/LMCache`，vLLM 0.10.2 兼容） | GPU↔CPU↔磁盘分层 offload 底座：chunk 级批量搬运、layer-wise 与计算重叠、reload>recompute、pin/lookup API。作为 TTL 到期/显存紧时的 offload 后端。 | **已集成验证**（commit 97f882f，`src/mimir/lmcache_compat.py`：otel provider 兜底 + connector 自注册 + 可用性报告；engine 按 `extra["lmcache"]=True` 接入；真引擎三件套共存冒烟） |
| CacheGen（`UChi-JCL/CacheGen`，SIGCOMM'24） | KV 编解码：delta+分层量化+算术编码把 KV 张量压成 bitstream（论文真实 KV 3.5–4.3× 压）。**编解码器已随 LMCache 0.4.7 一并发布**（`lmcache.v1.storage_backend.naive_serde.cachegen_encoder.CacheGenSerializer`，作为 storage serde 后端，`remote_serde="cachegen"` 选用）。 | **已验证集成**（`tests/test_cachegen_serde.py`：Qwen3-4B 真实 KV 形状 round-trip + 压缩 2.88× + 小于 naive；模型名走 AutoConfig 默认回退，36 层） |
| Mimir（自有编排） | 把三者串成工具调用场景的统一管线：TTL 触发 → LMCache offload → CacheGen 编码/解码。`engine_vllm_v1` 适配层接线。 | 三篇接线均已落地；并发多轮 agent 端到端收益量化 benchmark 为下一步 |

**诚实边界**：三篇机制均已跑通，但收益依赖显存争用（并发多轮 agent），单 agent 顺序跑无争用则 vLLM 自带 APC 命中、LMCache/CacheGen 不触发——须用并发多轮场景才体现量化收益。CacheGen 压缩比测的是合成随机 KV（2.88×），论文真实 KV 因 token-wise locality 达 3.5–4.3×。三篇为 credited prior work，移植集成不占为己有。

## Mimir 侧适配器

`src/mimir/engine_vllm_v1.py`：`VLLMEngineV1`（强制 v1 InprocClient）暴露
- `kv_usage()` / `mimir_stats()`：读 v1 block_pool + scheduler 的块统计（used/total/CoW）
- `set_current_task(task_id)` / `chat_task(...)`：注入 `engine_core._mimir_current_task`（CoW 记账用）
- `chat_full()` / `chat_batch()`：Continuum TTL 元数据注入（`extra_args` 的 `job_id`/`this_func_call`/`is_last_step`）+ 真批量并发
- LMCache 按 `extra["lmcache"]=True` 经 `lmcache_compat.ensure_lmcache()` 接入（修 otel + 注册 connector + 注入 `kv_transfer_config`）

## 验证脚本

| 脚本 | 验证 |
| --- | --- |
| `scripts/run_phase_d_cow.py` | CoW 跨分支复用（9 次） |
| `scripts/run_phase7_fp8kv.py`（改用 v1）| fp8 回退 |
| `scripts/run_phase_k_multimodel.py` | 多模型规模泛化（1.7B/4B CoW + block-class）|
| `scripts/run_phase_blockclass.py` | **【创新】block-class 标签注入 + 类别感知淘汰**（evict 只淘汰 reasoning） |
| `tests/test_block_class.py` | block-class 确定性单测（优先级序 + ref_cnt 跳过，5 例）|
| `scripts/gen_deepseek_traces.py` | DeepSeek V4 Pro 产真实 agent 轨迹 |
| `scripts/run_trace_benchmark.py` | DeepSeek 轨迹 A/B（native 崩 vs Mimir 跑完）|
| `scripts/run_llm_judge.py` | DeepSeek-judge 保真 A/B（压缩无损正确性）|

结果落盘：`benchmark_results/phase_{d,f,k,blockclass}_*.json` + `trace_bench_*` + `llm_judge_*`。
（原 phase_c/e/m/o/p/q 与 concurrent_press 的回收相关结果随机制删除而废弃，保留备查。）
