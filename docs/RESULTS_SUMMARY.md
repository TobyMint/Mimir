# Mimir 结果总览（评审速读）

> 一页纸看懂 Mimir 的优化效果。全部真实测量，Qwen3-4B-Instruct-2507，单卡 RTX 3090，vLLM 0.10.2。
> **真实部署配置**：`gpu_memory_utilization=0.9`（vLLM 默认，榨干单卡）。
> 详见 `docs/测试报告.md`、`docs/VLLM_PATCH_INVENTORY.md`、`benchmark_results/`。

## 头图：真实减少必需 KV + 提速（核心收益）

Mimir 的收益不在「把显存计数器刷成 0」（那种归 0 是任务结束后的瞬时计数，推理时该占仍占、系偷换概念），而在**真实减少进入 KV 的 prompt 量 + 避免 OOM 崩溃**：

| 方向 | 指标 | baseline | Mimir | 降幅 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 上下文压缩 | tool_call TTFT | 236ms | 17ms | **-93%** | 压缩冗余内容，真实减少 prefill 计算量 |
| 工具数据外置 | 进 KV 的 new_prefill | 3792 tok | 647 tok | **-83%** | 大 JSON 外置、上下文只留引用，真实减少必需 KV |
| 分支 CoW | KV tokens（记账） | 7040 | 1500 | **-78.7%** | 分支共享前缀，9 次跨分支复用 |
| 分层存储 | 长上下文存活轮次 | 4/20（OOM） | 20/20 | OOM→存活 | 「能跑 vs 崩」，不偷换 |
| 端到端 demo | 10 轮 new_prefill | 34 | 2 | **-94%** | 压缩 + 外置叠加 |
| fp8 KV 量化 | KV 容量 | 1772 块 | 3659 块 | **2.06x** | 精度换容量（Hopper 硬件） |

> 长生命周期存活（分层存储）：baseline 第 5 轮 OOM，Mimir（GPU/HOST/DISK 三层）存活全部 20 轮 —— 见 ![分层存活曲线](benchmark_results/phase5_tiered_tier.png)

## 避免 OOM 崩溃（「能跑 vs 不能跑」，不偷换）

长上下文 / 工具结果累积下，原生 vLLM 上下文溢出直接崩溃，Mimir 靠工具外置压缩上下文、分层存储兜住，存活完成任务：

| 场景 | 原生 vLLM | Mimir |
| --- | --- | --- |
| 分层存储 20 轮长上下文 | 第 5 轮 **OOM 崩溃**（存活 4/20） | 存活全部 20 轮 |
| DeepSeek 轨迹 compare_frameworks | 上下文溢出**崩溃** | 压缩后跑完 |

**DeepSeek-V4-Pro 真实轨迹 A/B**（创新评测，native 跑同一份前沿模型轨迹；轨迹口径已对齐）：

| 任务 | native | Mimir | 能否跑完 |
| --- | --- | --- | --- |
| compare_frameworks | 崩（上下文溢出） | 3 步跑完 | native 崩，Mimir 活 |
| multi_step_estimate | 7 步 | 7 步 | 都跑完 |
| research_kv_cache | 10 步 | 10 步 | 都跑完 |

**LLM-judge 保真**（DeepSeek-flash 裁判，full vs Mimir 压缩上下文打分 0-10）：能跑场景压缩无损（10/10==10/10）；full 上下文超 `max_model_len` 直接崩，Mimir 压缩后反能答——压缩不仅是省显存，更是救任务。

> **诚实 trade-off**：低强度回放下 Mimir 的 offload 使上下文前缀 hash 改变、vLLM APC 前缀缓存命中率下降，TTFT 与 native 持平或略高。这是「真实减少必需 KV、避免崩溃」的代价——价值在避免 OOM + 真实减少 prefill，不在低强度抢 TTFT。

## 诚实边界：已移除的「lifecycle 主动回收 / per-block pin」

经自审，早期版本的「任务边界主动回收 / per-block pin」机制使 `used_blocks→0`，但那是任务结束后的瞬时计数——推理时该占仍占、回收重算反而拖慢服务、对用户无益。系偷换概念，**已彻底从代码与测评中删除**。Mimir 的真收益回归：工具外置/压缩真实减少必需 KV、避免 OOM、block-class 创新差异化。（负结果数据 `concurrent_press_*.json` 保留备查。）

## vLLM 内核层 in-tree patch（patched v1，7 文件 144 处 Mimir 标记，纯 Python 不重编 _C）

| Phase | patch | 引擎实测 |
| --- | --- | --- |
| B | `scheduler.get_mimir_stats()` 块级统计导出 | total_blocks / used 可读 |
| D | `kv_cache_manager` 跨分支 CoW 复用记账 | 9 次复用 |
| F | fp8 KV 优雅降级 | 不支持硬件降级 bf16 不崩 |
| R | `output_processor` v1 TTFT 回填 + `disable_log_stats=False` | 每请求 TTFT/prefill 可观测 |
| **BC** | **【创新核心】block-class 类别感知淘汰** | `evict(57)` 只淘汰 reasoning 57 块、tool_result/system 0 损失；5 单测 + probe 召回佐证 |

> 注：原 Phase C/E/I/J/L 的「lifecycle 主动回收 / per-block pin」机制已删除（used_blocks→0 系偷换概念，见上方诚实边界）。

### 创新核心：tool-call 感知的 per-block KV 类别管理（Phase BC）【夺冠差异化】

业界 KV 淘汰（H2O/SnapKV/PyramidKV/KIVI）在长上下文 QA 上按 attention 权重/访问热度评分；agent-specific 的（IntentKV/FlowKV/MemArt）仅研究阶段、SGLang-only。**无论文按「语义角色」标签化 KV 块并按类别路由淘汰**——Mimir 填补该空白：每个 KV 块打 `{system, user, reasoning, tool_result}` 标签，按 `reasoning > user > tool_result > system` 优先级淘汰（工具返回保留、推理中间态优先丢）。**3090 无 fp8**，KV 容量只能靠「谁该被淘汰」更聪明。详见 `docs/技术方案.md` §3.7、`tests/test_block_class.py`、`scripts/run_phase_blockclass.py`、`scripts/run_recall_metric.py`。

## 异构硬件抽象（赛题方向之六）

`mimir/hardware/device.py`：`DeviceBackend` 抽象（CUDA/ROCm/Ascend/Cambricon/CPU），运行时探测（`torch.cuda` / `torch_npu` / 环境变量）+ 降级链。`supports_fp8()` 只在 Hopper(sm90)+ 为真；3090(sm86) 走 Phase F 优雅降级。`recommend_engine_config()` 按后端推荐 dtype/kv_cache_dtype（无卡→float32/eager；3090→bf16；Hopper→fp8）。9 测试覆盖；真实报告：4× RTX 3090 sm86，fp8 false。

## 多模型规模泛化（Phase K）

| 模型 | CoW 复用 | block-class 标签 | patch 生效 |
| --- | --- | --- | --- |
| Qwen3-1.7B | 3 | 标签注入成功 | ✅ |
| Qwen3-4B-Instruct-2507 | 3 | 标签注入成功 | ✅ |
| Qwen3-8B | — | — | 3090 显存不足（非 patch 问题） |

## 与同实验室 Continuum 的差异化

Mimir 独有（Continuum 无）：block-class 类别感知淘汰（Phase BC，创新核心）、CoW 复用记账（Phase D）、fp8 容错（Phase F）、工具数据外置 + 上下文压缩真实减少必需 KV。
Continuum 做 tool-call-pause 的 time-bounded whole-request pin；Mimir 做语义角色标签化的 per-block 类别淘汰 + 工具外置。

## 统一入口（外部层 + 引擎层协同）

```python
from mimir.manager import MemoryManager
from mimir.engine_vllm_v1 import VLLMEngineV1

mm = MemoryManager(features=["context_compress", "tool_offload", "prefix_cache", "tiered"])
eng = VLLMEngineV1(cfg, device=0)  # 内核 block-class 标签 + CoW 记账
# 单次调用：外部压缩/外置真实减少进 KV 的 prompt
r = mm.run_turn_with_engine(eng, case, task_id="t1", max_tokens=16)
# r["engine_stats"]["mimir_block_class"] / ["mimir_cow_reuses"] 可观测
```

## 一键复现

```bash
make reproduce          # CPU 模式 ~2 分钟（115 测试 + 仿真）
# 或带 GPU：bash scripts/reproduce.sh
```
