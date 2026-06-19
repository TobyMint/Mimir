# Mimir 结果总览（评审速读）

> 一页纸看懂 Mimir 的优化效果。全部真实测量，Qwen3-4B-Instruct-2507，单卡 RTX 3090，vLLM 0.10.2。
> 详见 `docs/测试报告.md`、`docs/VLLM_PATCH_INVENTORY.md`、`benchmark_results/`。

## 头图：两个引擎级决定性 A/B（used_blocks，越低越好）

| 场景 | 原生 vLLM | Mimir (in-tree patched v1) | 说明 |
| --- | --- | --- | --- |
| 单 agent 10 轮对话（Phase M） | **69**（累积） | **0**（reclaims=213） | mimir 策略每轮自动回收 |
| 3 agent 并发 6 步（Phase O） | **14**（累积） | **0**（reclaims=24） | per-task 隔离 + 自动回收 |
| KV 池压力 6 任务（Phase P） | **27**（累积） | **0**（reclaims=132） | lifecycle-aware 分配 + 自动回收 |
| 工具调用并发 3agent×2轮（Phase Q） | **262**（大返回进KV） | **0**（offload+回收，reclaims=42） | tool_offload + 逐任务回收 |

原生 vLLM KV 持续累积，Mimir 在任务边界主动回收，显存稳态为 0。

## 外部优化层（Mimir 模块，v0 引擎）

| 方向 | 指标 | baseline | Mimir | 降幅 |
| --- | --- | --- | --- | --- |
| 上下文压缩 | tool_call TTFT | 307ms | 27ms | **-91%** |
| 工具数据外置 | tool_call TTFT | 304ms | 29ms | **-90%** |
| 分支 CoW | KV tokens（记账） | 7040 | 1500 | **-78.7%** |
| 分层存储 | 长上下文存活轮次 | 4/20 (OOM) | 20/20 | OOM→存活 |
| 生命周期淘汰 | 主动回收率 | 0% (LRU) | 100% | +100% |
| 端到端 demo | 10 轮 new_prefill | 34 | 2 | **-94%** |
| fp8 KV 量化 | KV 容量 | 1772 块 | 3659 块 | **2.06x** |

## vLLM 内核层 in-tree patch（patched v1，10 文件，纯 Python 不重编 _C）

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
make reproduce          # CPU 模式 ~2 分钟（92 测试 + 仿真）
# 或带 GPU：bash scripts/reproduce.sh
```
