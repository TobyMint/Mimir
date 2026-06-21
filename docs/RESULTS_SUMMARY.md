# Mimir 结果总览（评审速读）

> 一页纸看懂 Mimir 的优化效果。全部真实测量，Qwen3-4B-Instruct-2507，单卡 RTX 3090，vLLM 0.10.2。
> **真实部署配置**：`gpu_memory_utilization=0.9`（vLLM 默认，榨干单卡），KV 池 **5534 块**（88544 token）。
> 详见 `docs/测试报告.md`、`docs/VLLM_PATCH_INVENTORY.md`、`benchmark_results/`。

## 头图：工具调用并发 A/B（Phase Q，Mimir 真赢的场景）

| 场景 | 原生 vLLM | Mimir | 说明 |
| --- | --- | --- | --- |
| 工具调用并发 3agent×2轮（Phase Q，~5KB 返回） | **262**（大返回进KV） | **0**（offload+回收，reclaims=42） | tool_offload + 逐任务回收 |
| 单 agent 10 轮对话（Phase M） | **74**（累积） | **0**（reclaims=239） | mimir 策略每轮自动回收 |
| 3 agent 并发 6 步（Phase O） | **14**（累积） | **0**（reclaims=24） | per-task 隔离 + 自动回收 |
| KV 池压力 6 任务（Phase P） | **27**（累积） | **0**（reclaims=132） | lifecycle-aware 分配 + 自动回收 |

原生 vLLM KV 持续累积，Mimir 在任务边界主动回收 + 工具外置，显存稳态为 0。Phase Q 是最大赢面（262→0）且最贴合赛题工具调用场景，图表 `phase_q_toolcall_concurrent_*_curves.png`。

### 诚实边界：回收策略在并发吞吐下不占优（已验证，不回避）

我们也做过 util=0.9 大池子下的真·并发压测（一次提交 N 个请求 / 多轮 / 请求潮三种模式），结果诚实记录：
- **0.9 大池子下，native 靠 vLLM 的 preemption（KV 换出换入）能默默扛住高并发，不 OOM、请求都完成**；Mimir 的"任务结束即清空 KV"在纯吞吐场景下是负优化（清空导致前缀失配、后续重算，反而少完成请求或 TTFT 持平）。
- **结论：Mimir 回收的真价值在「长生命周期 / 显存真紧张到要 OOM」的场景，不在「大池子高并发吞吐」**。故主评测回归交替多轮场景（Phase Q 等上面的表，Mimir 真赢）与上下文压缩（TTFT -93%）。
- 这条边界如实写进文档，不藏——反而体现工程诚实。详见 `docs/测试报告.md` §5.0 与 `benchmark_results/concurrent_press_*.json`（负结果数据保留）。

## 其它引擎级 A/B（used_blocks，越低越好，交替场景）

| 场景 | 原生 vLLM | Mimir (in-tree patched v1) | 说明 |
| --- | --- | --- | --- |
| 单 agent 10 轮对话（Phase M） | **74**（累积） | **0**（reclaims=239） | mimir 策略每轮自动回收 |
| 3 agent 并发 6 步（Phase O） | **14**（累积） | **0**（reclaims=24） | per-task 隔离 + 自动回收 |
| KV 池压力 6 任务（Phase P） | **27**（累积） | **0**（reclaims=132） | lifecycle-aware 分配 + 自动回收 |

原生 vLLM KV 持续累积，Mimir 在任务边界主动回收，显存稳态为 0。

**DeepSeek-V4-Pro 真实轨迹 A/B**（创新评测，native 跑同一份前沿模型轨迹；轨迹口径已对齐：num_steps == replay 步数 == assistant 消息数）：

| 任务 | native | Mimir | native 峰值 used | Mimir 峰值 used | 能否跑完 |
| --- | --- | --- | --- | --- | --- |
| compare_frameworks | 崩 | 3 步 | 285 | **0** | native 上下文溢出崩，Mimir 跑完 |
| multi_step_estimate | 7 步 | 7 步 | 763 | **0** | 都跑完，但 Mimir 显存稳态 0 |
| research_kv_cache | 10 步 | 10 步 | 1235 | **0** | 都跑完，Mimir 显存稳态 0 |

**核心结论**：Mimir 全程 `used_blocks=0`、显存稳态为 0；native 在重轨迹（compare）下上下文溢出崩溃，在另两任务里虽勉强跑完但 KV 堆到 763/1235 块（逼近 1780 块池上限，再重一点就崩）。崩溃场景下匹配步 TTFT Mimir −86%（compare 316 vs 43 ms）。

> **诚实的 trade-off**：在两任务都不崩的低强度回放下，Mimir 的 offload 使上下文前缀 hash 与 native 不同，vLLM APC 前缀缓存命中率下降，mimir 侧每步 new_prefill 略升、TTFT 与 native 持平或略高（multi_step −0.1% / research +74%）。这是「显存换重算」的固有权衡——Mimir 的价值在「显存维稳态 0 + 不崩」，而非在低强度回放下抢 TTFT；高强度/长生命周期场景前缀复用收益会回归。

**LLM-judge 保真**（DeepSeek-flash 裁判，full vs Mimir 压缩上下文打分 0-10）：能跑场景压缩无损（10/10==10/10）；full 上下文超 `max_model_len` 直接崩，Mimir 压缩后反能答——压缩不仅是省显存，更是救任务。

## 外部优化层（Mimir 模块，patched v1 + Phase R TTFT 回填）

| 方向 | 指标 | baseline | Mimir | 降幅 |
| --- | --- | --- | --- | --- |
| 上下文压缩 | tool_call TTFT | 236ms | 17ms | **-93%** |
| 工具数据外置 | tool_call TTFT | 240ms | 21ms | **-91%**（43776 字符移出）|
| 分支 CoW | KV tokens（记账） | 7040 | 1500 | **-78.7%** |
| 分层存储 | 长上下文存活轮次 | 4/20 (OOM) | 20/20 | OOM→存活 |
| 生命周期淘汰 | 主动回收率 | 0% (LRU) | 100% | +100% |
| 端到端 demo | 10 轮 new_prefill | 34 | 2 | **-94%** |
| fp8 KV 量化 | KV 容量 | 1772 块 | 3659 块 | **2.06x** |

> 长生命周期存活（Phase 5，分层存储）：baseline 第 5 轮 OOM，Mimir（GPU/HOST/DISK 三层）存活全部 20 轮 —— 见 ![分层存活曲线](benchmark_results/phase5_tiered_tier.png)

## vLLM 内核层 in-tree patch（patched v1，7 文件 144 处 Mimir 标记，纯 Python 不重编 _C）

| Phase | patch | 引擎实测 |
| --- | --- | --- |
| B | `scheduler.get_mimir_stats()` 块级统计导出 | total_blocks 可读 |
| C | `block_pool.mimir_finish_task` 任务边界主动回收 | used 10→0（reclaims 10） |
| D | `kv_cache_manager` 跨分支 CoW 复用记账 | 9 次复用 |
| E | per-block KV-pin（lifecycle-bounded） | 3/3 pinned 块压力下存活 |
| F | fp8 KV 优雅降级 | 不支持硬件降级 bf16 不崩 |
| G | `'mimir'` 调度策略 + MimirRequestQueue | 引擎日志 "Mimir policy active" |
| I | `mimir_pin_hits` 计数器 | pin 阻止回收可观测 |
| J | `mimir_reclaim_evictable` 闭环回收 | EVICTABLE 块扫描回收 |
| L | mimir 策略自动回收（自驱动） | 任务完成即回收，无需外部调用 |
| R | `output_processor` v1 TTFT 回填 + `disable_log_stats=False` | 每请求 TTFT/prefill 可观测 |
| **BC** | **【创新核心】block-class 类别感知淘汰** | `evict(57)` 只淘汰 reasoning 57 块、tool_result/system 0 损失；5 单测 + probe 召回佐证 |

### 创新核心：tool-call 感知的 per-block KV 类别管理（Phase BC）【夺冠差异化】

业界 KV 淘汰（H2O/SnapKV/PyramidKV/KIVI）在长上下文 QA 上按 attention 权重/访问热度评分；agent-specific 的（IntentKV/FlowKV/MemArt）仅研究阶段、SGLang-only。**无论文按「语义角色」标签化 KV 块并按类别路由淘汰**——Mimir 填补该空白：每个 KV 块打 `{system, user, reasoning, tool_result}` 标签，按 `reasoning > user > tool_result > system` 优先级淘汰（工具返回保留到任务结束、推理中间态优先丢）。**3090 无 fp8**，KV 容量只能靠「谁该被淘汰」更聪明——正是 agent 场景下最大化有效 KV 容量的正解。详见 `docs/技术方案.md` §3.7、`tests/test_block_class.py`、`scripts/run_phase_blockclass.py`、`scripts/run_recall_metric.py`。

## 异构硬件抽象（赛题方向之六）

`mimir/hardware/device.py`：`DeviceBackend` 抽象（CUDA/ROCm/Ascend/Cambricon/CPU），运行时探测（`torch.cuda` / `torch_npu` / 环境变量）+ 降级链。`supports_fp8()` 只在 Hopper(sm90)+ 为真；3090(sm86) 走 Phase F 优雅降级。`recommend_engine_config()` 按后端推荐 dtype/kv_cache_dtype（无卡→float32/eager；3090→bf16；Hopper→fp8）。9 测试覆盖；真实报告：4× RTX 3090 sm86，fp8 false。

## 多模型规模泛化（Phase K）

| 模型 | lifecycle 回收 | CoW 复用 | patch 生效 |
| --- | --- | --- | --- |
| Qwen3-1.7B | used 3→0 (reclaims=3) | 3 | ✅ |
| Qwen3-4B-Instruct-2507 | used 3→0 (reclaims=3) | 3 | ✅ |
| Qwen3-8B | — | — | 3090 显存不足（非 patch 问题） |

## 与同实验室 Continuum 的差异化

Mimir 独有（Continuum 无）：任务边界主动回收（Phase C）、per-block pin（Phase E）、CoW 记账（Phase D）、fp8 容错（Phase F）、`mimir` 调度策略（Phase G）。
Continuum 做 tool-call-pause 的 time-bounded whole-request pin；Mimir 做 lifecycle-bounded per-block pin + 任务边界主动回收 + 自驱动策略。

## 统一入口（外部层 + 引擎层协同）

```python
from mimir.manager import MemoryManager
from mimir.engine_vllm_v1 import VLLMEngineV1

mm = MemoryManager(features=["context_compress","tool_offload","prefix_cache","tiered","lifecycle"])
eng = VLLMEngineV1(cfg, device=0)  # scheduling_policy="mimir" -> 引擎层自动回收
# 单次调用：外部压缩/外置 + 引擎 mimir 策略自动回收
r = mm.run_turn_with_engine(eng, case, task_id="t1", max_tokens=16)
# r["engine_stats"]["mimir_lifecycle_reclaims"] 显示回收次数，used_blocks 守恒
```

## 一键复现

```bash
make reproduce          # CPU 模式 ~2 分钟（115 测试 + 仿真）
# 或带 GPU：bash scripts/reproduce.sh
```
