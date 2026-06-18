# ROADMAP — 执行路线图

> 每个阶段的退出标准 = 「测试通过 + benchmark 落盘并 commit + 更新 PROGRESS.md」。
> 阶段独立可交付；GPU 忙时做非 GPU 阶段（写码/测试/文档/画图）。

- [ ] **Phase 0 — 基座**：conda 环境 `mimir` + torch + vLLM；charter 文件；vLLM 跑通 Qwen3-4B-Instruct 单卡生成。
- [ ] **Phase 1 — Baseline**：benchmark 接真实 vLLM；3 类工作流（多轮/工具/分支）；记录**未优化基线**指标（显存/TTFT/E2E/吞吐/成功率）。
- [ ] **Phase 2 — 上下文压缩**：轮次摘要 + 静态去重，保真度可调。测：显存峰值↓、成功率≈。
- [ ] **Phase 3 — 工具数据外置**：大返回外存 + 惰性加载，KV 仅留引用。测：工具密集场景显存↓。
- [ ] **Phase 4 — 分支 CoW**：前缀树 + 写时复制 + agent 生命周期。测：ToT 显存↓∝分支数。
- [ ] **Phase 5 — 分层存储**：GPU/CPU(vLLM offload)/Disk 三层 + 迁移策略。测：超长上下文存活（基线 OOM）。
- [ ] **Phase 6 — 生命周期感知淘汰**：step/任务边界感知 vs LRU。测：命中率↑、重算↓。
- [ ] **Phase 7 — 多模型/多任务**：单卡跨任务 KV 协调。测：直击「多模型/多任务」40 分子项。
- [ ] **Phase 8 — 定点 patch vLLM**：选最高收益特性改内核、重编译、测增量。报告「我们修改了 vLLM 内核」。
- [ ] **Phase 9 — 博眼球层**：实时内存仪表盘（Web/TUI）+ matplotlib 图表 + 端到端 demo（基线 OOM vs Mimir）+ 动图/短视频。
- [ ] **Phase 10 — 收尾**：一键复现 + 全测试覆盖 + 四份文档定稿（真实数字）+ 仓库整洁。

## 指标口径（固定，对应 docs/测试报告.md §4）

`peak_gpu_memory_gb`（核心）、`ttft_ms`、`e2e_latency_s`、`throughput_tok_per_s`、`task_success_rate`。
**优化前后同一硬件配置、同一模型、同一 seed。**

## 消融实验（每个特性都要进）

baseline → 逐特性叠加 → 全开，记录显存/延迟/成功率曲线，支撑「应用效果」40 分。
