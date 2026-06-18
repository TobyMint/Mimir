# GOAL — Mimir 冠军级交付目标

> 这是整个项目的「北极星」。任何新会话先读本文件 + `PROGRESS.md`，即可对齐当前状态。

## 一句话目标

把 **Mimir**（面向智能体推理的内存管理系统）做到 **可交付、且具备夺冠竞争力** 的状态。
推理引擎用 **vLLM**，单卡（RTX 3090）跑，模型用 **Qwen3 系列**。

## 锁定的关键决策（讨论已确认）

| 决策点 | 结论 |
| --- | --- |
| vLLM 路线 | **先包裹 vLLM 公共 API（去风险、快速出真实数字）→ 后定点 patch 分支 CoW 等（拔高 + 报告「我们修改了 vLLM 内核」）** |
| GPU 调度 | **忙 → 轻量正确性验证；空闲 → 重 benchmark**；每次跑前 `nvidia-smi` 选最空闲单卡，设 `CUDA_VISIBLE_DEVICES` |
| 进度通知 | GitHub 提交 + `PROGRESS.md`（实时）+ 关键节点邮件（`~/.claude/hooks/notify_email.py`，163 邮箱） |
| 博眼球范围 | **全做**：实时内存仪表盘 + 报告 matplotlib 图表 + 端到端 agent demo + 演示动图/短视频 |
| 环境隔离 | 独立 conda 环境 `mimir`（Python 3.11），**不动共享 base** |
| 主力模型 | 开发 Qwen3-4B-Instruct-2507；泛化验证 Qwen3-1.7B + Qwen3-8B |
| 资源 | token / 时间不限，做到最好为止 |

## 范围（六大优化方向全做）

1. KV Cache 生命周期管理（前缀复用 / 淘汰 / 重分配）
2. 分支推理内存共享（CoW）
3. Prompt 与上下文压缩
4. 工具调用数据外置
5. 分层存储（GPU / CPU / Disk）
6. 异构硬件抽象（CUDA 优先，国产化为扩展）

每个方向都要有 **baseline vs optimized 的可测数字**（显存峰值↓、TTFT/E2E↓、任务成功率≈持平）。

## 执行原则（自治长任务）

- **持续小步前进**：每个阶段满足「测试通过 + benchmark 落盘 + commit/push + 更新 PROGRESS.md」才算完成。
- **GPU 被占就做非 GPU 工作**：写代码、写测试、写文档、调研、画图——绝不空转。
- **每阶段独立可交付**：无论停在哪都是完整状态。
- **遇到阻塞写进 PROGRESS.md 的「阻塞」段**，并切去做可推进的事。
- **关键节点发邮件**：`python3 ~/.claude/hooks/notify_email.py "<阶段> 完成"`。

## 如何续上（给未来的我）

1. 读 `PROGRESS.md`（当前阶段 / 已完成 / 进行中 / 阻塞 / 下一步）。
2. 读 `ENV.md`（conda 环境路径、可用的 torch/vLLM 版本、运行命令、模型路径）。
3. 读 `docs/ROADMAP.md`（完整阶段路线与退出标准）。
4. 读 `docs/DECISIONS.md`（架构决策记录，避免重复争论）。
5. 继续。
