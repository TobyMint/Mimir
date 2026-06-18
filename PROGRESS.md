# PROGRESS — 实时进度

> **续上时先读本文件。** 记录当前阶段、已完成、进行中、阻塞、下一步、环境与 GPU 选择。

**当前阶段**：Phase 0 — 基座（环境重建中）+ Phase 1 预备（可推进部分已完成）

**最后更新**：2026-06-18

## 已完成 ✅

- git 仓库初始化（`main`），远程 `origin` = `git@github.com:TobyMint/Mimir.git`（SSH）
- 项目骨架 + 文档骨架 + 源码骨架 + benchmark 脚手架（16 测试通过，ruff clean）
- **conda 环境 `mimir`**（py3.11）创建
- 邮件通道测试通过（163，`x2406862525@163.com`）
- **charter 文件**：GOAL / ROADMAP / PROGRESS / ENV / DECISIONS 已提交推送
- **Phase 1 预备（vLLM 无关）**：`mimir/metrics.py`（MetricsCollector）、`benchmarks/workloads.py`（3 类工作流生成器）、`tests/test_metrics.py`（4 测试）已提交

## 进行中 🔄

- **环境重建**：首次 `pip install -U vllm` 装到 vLLM 0.23.0 → 拉入 **torch 2.11.0 + CUDA 13**，与本机驱动 12.8 不兼容（`cuda_available=False`，报 "driver too old (12080)"）。
  → 正在重建：**torch 2.6.0+cu126 + vLLM 0.8.x**（CUDA 12，驱动 12.8 兼容）。后台任务 `bspnorqln`。

## 下一步 ➡️

1. 环境就绪后校验 `torch.cuda.is_available()==True` + vLLM 版本 → 回填 `ENV.md`
2. 单卡跑通 vLLM 服务 Qwen3-4B-Instruct-2507（smoke 生成）
3. 写 vLLM 引擎适配器（`mimir/engine_vllm.py`）+ 把 workloads 喂进去
4. Phase 1：记录未优化基线指标 → `benchmark_results/`

## 阻塞 🚧

（暂无。GPU 被占满时回退做非 GPU 工作：写码 / 测试 / 文档 / 画图。）

## 环境与运行备忘

- 激活：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mimir`
- GPU 选择：跑前 `nvidia-smi` 选最空闲单卡 → `export CUDA_VISIBLE_DEVICES=<idx>`；全占则退避
- 模型：`/data/models/Qwen3-{1.7B,4B-Instruct-2507,8B}`
- 调度规则：忙→轻量正确性；空闲→重 benchmark
- 发邮件：`python3 ~/.claude/hooks/notify_email.py "<标题>"`

## 关键节点

- 2026-06-18 charter 提交推送（commit 92940b7）
- 2026-06-18 Phase 1 预备（metrics/workloads/tests）提交推送
- 2026-06-18 发现 vLLM 0.23 / torch2.11 不兼容驱动 12.8 → 重建为 torch2.6+cu126 / vLLM 0.8.x（进行中）
- [待] Phase 0 完成：vLLM 跑通 Qwen3-4B
