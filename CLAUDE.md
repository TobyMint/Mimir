# CLAUDE.md

本文件为 Claude Code 在本仓库工作时提供的指引。每次会话开始时自动加载。

## 项目概览

**Mimir** 是参赛「面向智能体的内存管理系统设计与实现（高校赛题）」的项目 ——
在 vLLM / llama.cpp 等推理框架之上，提供面向智能体长生命周期推理的内存管理优化
（KV Cache 生命周期管理、分支 CoW、上下文压缩、工具数据外置、分层存储、异构硬件）。

- 完整赛题与评分细则：`docs/赛题说明.md`
- 技术方案：`docs/技术方案.md`
- 系统设计：`docs/系统设计.md`

## 仓库结构

```
src/mimir/        主包；core.py 为编排层，其余子模块各对应一个赛题优化方向
  core.py           KVBlock / MemoryManager / 枚举
  kv_cache/         KV 生命周期管理（前缀复用、淘汰）
  branch/           分支推理 CoW
  context/          Prompt / 上下文压缩
  tools/            工具中间数据外置
  tiered/           分层存储与冷热迁移
  hardware/         CUDA / DTK / CANN 设备抽象
tests/            单元测试（pytest）
benchmarks/       评测脚手架（python -m benchmarks.run）
docs/             赛题说明 / 技术方案 / 系统设计 / 部署指南 / 测试报告
```

## 常用命令

```bash
make dev-install    # 安装开发依赖（vllm, agent, dev）
make test           # 运行测试
make test-fast      # 跳过 slow / gpu 标记的测试
make lint           # ruff 代码检查
make format         # 自动格式化
make typecheck      # mypy
make benchmark      # 运行 Benchmark
```

直接命令：`pytest`、`ruff check src tests benchmarks`、`python -m benchmarks.run`。

## 工作约定（重要）

### Git：频繁提交并推送

- **每个逻辑里程碑都应提交**（不要攒成一个大 commit，也不要无意义拆分）。
- **每次 commit 后都 `git push`** 到 `origin`（`main` 分支）。
- 提交信息使用 Conventional Commits 风格：`feat:` / `fix:` / `docs:` / `test:`
  / `refactor:` / `chore:` / `perf:`，标题用简洁中文或英文，正文说明动机与范围。
- 仅在用户明确要求时才 push；本仓库用户已授权每次 commit 后即推送。

### GitHub 访问：仅用 SSH

- 本环境内 **GitHub 的 HTTPS 访问被屏蔽**。所有 clone / fetch / pull / push 必须用
  SSH（`git@github.com:` 前缀），包括引用的子模块 / fork。
- 本仓库 `origin` 已配置为 `git@github.com:TobyMint/Mimir.git`。

### 代码风格

- Python 3.9+（`from __future__ import annotations`，类型注解可用新语法 `X | None`）。
- `ruff` 已配置（line-length 100）；提交前 `make format` 保持风格一致。
- 中文注释 / docstring 受欢迎（与赛题文档一致），代码标识符用英文。

## 验证习惯

修改 `src/` 后：`make test`（或至少相关测试）应通过；新增功能补对应测试。
`benchmarks/` 改动应能 `python -m benchmarks.run` 正常跑通（脚手架阶段允许 stub）。

## 优化方向 → 模块映射（评审对照）

| 赛题优化方向 | 模块 |
| --- | --- |
| KV Cache 生命周期管理 | `mimir.kv_cache` |
| 分支推理内存共享 | `mimir.branch` |
| Prompt 与上下文压缩 | `mimir.context` |
| 工具调用数据优化 | `mimir.tools` |
| 分层内存与异构存储 | `mimir.tiered` |
| 异构 AI 加速硬件支持 | `mimir.hardware` |

新增优化方向应作为 `src/mimir/` 下的新子模块，并经 `MemoryManager` 编排启用。
