# PROGRESS — 实时进度

> **续上时先读本文件。** 记录当前阶段、已完成、进行中、阻塞、下一步、环境与 GPU 选择。
> 每完成一个里程碑更新本文件并 commit/push。

**当前阶段**：Phase 0 — 基座（进行中）

**最后更新**：2026-06-18（开跑首轮）

## 已完成 ✅

- git 仓库初始化（`main`），远程 `origin` = `git@github.com:TobyMint/Mimir.git`（SSH）
- 项目骨架：README / LICENSE(Apache-2.0) / pyproject / Makefile / .gitignore / CLAUDE.md
- 文档骨架：赛题说明 / 技术方案 / 系统设计 / 部署指南 / 测试报告（模板）
- 源码骨架：`src/mimir/`（core.py + 6 个优化方向子模块），12 项测试通过，ruff 通过
- benchmark 脚手架：`benchmarks/run.py`（stub，口径已固化）
- **conda 环境 `mimir`**（Python 3.11，`/data/xbow/conda_envs/mimir`）
- **torch 2.6.0+cu126 安装并验证**：`torch.cuda.is_available()==True`，可见 4× RTX 3090
- **邮件通道测试**：已发测试邮件到 `x2406862525@163.com`（`~/.claude/hooks/notify_email.py`）

## 进行中 🔄

- **vLLM 安装**：`pip install -U vllm numpy`（后台进行），装完校验 CUDA 与 Qwen3-4B 加载

## 下一步 ➡️

1. 校验 vLLM 版本 + CUDA 仍可用 → 写入 `ENV.md`
2. 单卡跑通 vLLM 服务 Qwen3-4B-Instruct-2507，smoke 生成
3. 进入 Phase 1：benchmark 接真实 vLLM 引擎，记录基线

## 阻塞 🚧

（暂无。GPU 被占满时，回退做非 GPU 工作：写码/测试/文档/调研/画图。）

## 环境与运行备忘

- **激活环境**：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mimir`
- **GPU 选择**：跑前 `nvidia-smi`，选最空闲单卡 → `export CUDA_VISIBLE_DEVICES=<idx>`；全占则退避
- **模型**：`/data/models/Qwen3-1.7B`、`/data/models/Qwen3-4B-Instruct-2507`、`/data/models/Qwen3-8B`
- **GPU 调度规则**：忙→轻量正确性；空闲→重 benchmark
- **发邮件**：`python3 ~/.claude/hooks/notify_email.py "<标题>"`

## 关键节点（每次完成后在此追加一行）

- [时间] Phase 0 完成：环境就绪、vLLM 跑通 Qwen3-4B（待填）
