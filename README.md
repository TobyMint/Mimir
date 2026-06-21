# Mimir — 面向智能体的内存管理系统

> Mimir 取名自北欧神话中以智慧与记忆闻名者，寓意本项目聚焦于智能体推理过程中**记忆（KV Cache / 上下文）的管理与复用**。
>
> 📊 **结果总览（评审速读）**：[docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) — tool_call TTFT **-93%**（上下文压缩+工具外置，真减 KV）、分支 CoW **-78.7%**、block-class 创新核心、DeepSeek 真实轨迹 + LLM-judge 保真、多模型泛化。
>
> 🎯 **创新核心（Phase BC）**：tool-call 感知的 per-block KV 类别管理——给 KV 块打语义类别标签（system/user/reasoning/tool_result），按类别优先级淘汰。无论文做过，是 Mimir 的夺冠差异化。详见 [技术方案 §3.7](docs/技术方案.md)。
>
> 🔥 **核心收益（工具外置，真减少必需 KV）**：工具返回的大 JSON 外置、上下文只留引用，进 KV 的 prompt token 从 **3792 → 647**（-83%），tool_call TTFT **236ms → 17ms（-93%）**——真实减少 prefill 计算量，用户可感知提速。
> ![Agent-Loop](benchmark_results/agent_benchmark_heavy12_Qwen3-4B-Instruct-2507_curves.png)
>
> 🧯 **避免 OOM 崩溃**：长上下文累积下原生 vLLM 上下文溢出崩溃，Mimir 靠工具外置 + 分层存储存活（baseline 第 5 轮 OOM vs Mimir 存活 20 轮）——「能跑 vs 不能跑」，不偷换。
>
> 🎞️ **动态演示**：[Agent-Loop GIF](benchmark_results/agent_loop_demo.gif)（native 第 2-5 步崩溃 vs Mimir 全程跑完）

<p align="center">
  <b>研究创新赛道 · 面向智能体的内存管理系统设计与实现（高校赛题）</b>
</p>

## 项目简介

随着基于大语言模型（LLM）的智能体系统发展，推理范式已由传统的单轮生成扩展为涵盖
**规划 → 执行 → 反思**的长生命周期复杂过程。在这一过程中，智能体需进行多轮动态交互并频繁调用
工具与外部环境，导致推理上下文持续增长与反复重构，并带来 **KV Cache 持续累积、上下文高度冗余、
推理路径分支、工具调用产生大规模中间数据**等典型问题。

**Mimir** 是一个面向智能体推理过程的内存管理系统，在保证推理效果的前提下，通过对
KV Cache、上下文结构及显存分配机制的系统性优化，有效降低内存占用并提升整体推理效率。

## vLLM 0.10.2 In-Tree Patch（内核级优化）

Mimir 不仅在 vLLM 之上做外部封装，更**直接 patch 了 vLLM v0.10.2 源码**（拍平为普通目录
`third_party/vllm`（源自 v0.10.2 fork `mimir-patches-v0.10.2`），纯 Python、不重编 `_C`）：

| 引擎层 patch | 真实验证 |
| --- | --- |
| **分支 CoW 复用记账**（`kv_cache_manager`） | 4 分支测得 **9 次跨分支 KV 复用**（与 BranchTree 预测一致） |
| **fp8 KV 优雅降级**（arg_utils oracle） | 不支持的硬件上**降级 bf16**而非崩溃 |
| **【创新】block-class 类别感知淘汰**（`block_pool.mimir_class_aware_evict`） | `evict(57)` 只淘汰 reasoning 57 块、tool_result/system **0 损失**；5 单测 + probe 召回佐证 |
| **v1 TTFT 可观测**（Phase R 回填 `RequestMetrics`） | 每请求 TTFT/prefill 可读（v1 原本恒 None） |

详见 [`docs/VLLM_PATCH_INVENTORY.md`](docs/VLLM_PATCH_INVENTORY.md) 与 [`docs/VLLM_EDITABLE_SETUP.md`](docs/VLLM_EDITABLE_SETUP.md)。
区别于同实验室 Continuum（pin 工具调用暂停）—— Mimir 在 block-class 类别感知淘汰 + CoW 记账 + 工具外置上差异化。

> 注：早期版本的「lifecycle 主动回收 / per-block pin」（used_blocks→0）已移除——其归 0 是任务结束后瞬时计数，推理时该占仍占、回收重算反而拖慢服务，系偷换概念。Mimir 的真收益在工具外置/压缩减少必需 KV、避免 OOM、block-class 创新差异化。

## 核心优化方向

本赛题推荐的优化方向，本项目均规划支持（详见 [`docs/技术方案.md`](docs/技术方案.md)）：

| 模块 | 方向 | 说明 |
| --- | --- | --- |
| `mimir.kv_cache` | KV Cache 生命周期管理 | KV 的复用、淘汰与分层存储，支持动态资源回收与重分配 |
| `mimir.branch` | 分支推理内存共享 | 多路径决策下的 KV Cache 共享与 Copy-on-Write 机制 |
| `mimir.context` | Prompt 与上下文压缩 | 对 system prompt / 工具描述去重与精简，消除冗余 |
| `mimir.tools` | 工具调用数据优化 | 中间数据结构化存储或按需加载，避免完全进入 KV Cache |
| `mimir.tiered` | 分层内存与异构存储 | GPU 显存 / 主存 / 外部存储间的冷热分离与动态迁移 |
| `mimir.hardware` | 异构 AI 加速硬件支持 | 适配 CUDA / DTK / CANN 等异构平台与国产化算力 |

## 技术栈

- **基础推理框架**：vLLM / llama.cpp（在其之上扩展）
- **开源模型**：Qwen、MiniCPM 等
- **运行系统**：openEuler / openKylin / OpenHarmony 等（鼓励在更多 Linux 发行版上编译运行）
- **开发语言**：Python（核心），关键性能路径可选 C++ 扩展

## 目录结构

```
Mimir/
├── README.md              # 项目说明
├── CLAUDE.md              # Claude Code 协作约定
├── LICENSE                # Apache-2.0
├── pyproject.toml         # Python 项目配置
├── requirements.txt       # 依赖清单
├── Makefile               # 常用开发命令
├── docs/                  # 文档（赛题、技术方案、设计、部署、测试报告）
├── src/mimir/             # 源代码主包
├── tests/                 # 单元测试
└── benchmarks/            # Benchmark 测试方案
```

## 快速开始

> **重要**：vLLM v0.10.2 已**拍平为普通目录** `third_party/vllm`（不再是 submodule），
> 通过 `.pth` + dist-info 接入（详见 [`docs/VLLM_EDITABLE_SETUP.md`](docs/VLLM_EDITABLE_SETUP.md)）。
> 因此新 clone / 新环境**必须先 `source scripts/activate_env.sh`** 才能 `import vllm`。
> 该脚本幂等地：`conda activate mimir` → 写 `.pth`/dist-info → `LD_LIBRARY_PATH += torch/lib` → v1 单进程 env。

```bash
# 0. 首次：重建 vLLM 预编译二进制（fresh clone 需此步，~1 分钟）
bash scripts/setup_vllm_binaries.sh   # 从 vllm==0.10.2 wheel 提取 .so + flash_attn，symlink 进 vllm
#    （vllm_prebuilt_bin/ 被 gitignore，不入库；详见 docs/VLLM_EDITABLE_SETUP.md）

# 1. conda 环境 mimir（python 3.11 + torch 2.8.0+cu128 + vllm 0.10.2 依赖）
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mimir
pip install -e ".[dev]"          # Mimir 自身

# 2. 激活 vLLM flat 接入（每次会话）
source scripts/activate_env.sh

# 3. 一键复现验证（CPU 模式 ~2 分钟）
make reproduce    # 或 bash scripts/reproduce.sh --quick

# 4. 全量 Benchmark（需空闲单卡）—— 真实引擎实测
python scripts/run_phase_blockclass.py        # 创新：block-class 类别感知淘汰
python scripts/run_recall_metric.py           # probe 召回率（block-class 佐证）
python scripts/run_phase2_context.py          # 上下文压缩 TTFT -93%
python scripts/run_phase3_offload.py          # 工具外置 new_prefill -83%
python scripts/run_trace_benchmark.py         # DeepSeek 真实轨迹 A/B
python scripts/run_llm_judge.py               # DeepSeek-judge 压缩保真
```

## 复现性说明

- 优化前后使用**相同的硬件配置**进行评估
- 基于**开源大模型**（Qwen / MiniCPM 等）构建可复现的 Benchmark 测试方案
- 所有评测脚本位于 [`benchmarks/`](benchmarks)，结果与报告位于 [`docs/测试报告.md`](docs/测试报告.md)

## 赛题信息

完整赛题说明见 [`docs/赛题说明.md`](docs/赛题说明.md)。

- 赛道：研究创新赛道
- 赛题：面向智能体的内存管理系统设计与实现（高校赛题）
- 赛题联系人：张老师 jfzhang@nudt.edu.cn

## License

[Apache License 2.0](LICENSE)
