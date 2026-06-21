# ROADMAP — 执行路线图

> 每个阶段的退出标准 = 「测试通过 + benchmark 落盘并 commit + 更新 PROGRESS.md」。
> 阶段独立可交付；GPU 忙时做非 GPU 阶段（写码/测试/文档/画图）。

- [x] **Phase 0 — 基座**：conda 环境 `mimir` + torch + vLLM；charter 文件；vLLM 跑通 Qwen3-4B-Instruct 单卡生成。
- [x] **Phase 1 — Baseline**：benchmark 接真实 vLLM；3 类工作流（多轮/工具/分支）；记录**未优化基线**指标（显存/TTFT/E2E/吞吐/成功率）。
- [x] **Phase 2 — 上下文压缩**：轮次摘要 + 静态去重，保真度可调。测：显存峰值↓、成功率≈。
- [x] **Phase 3 — 工具数据外置**：大返回外存 + 惰性加载，KV 仅留引用。测：工具密集场景显存↓。
- [x] **Phase 4 — 分支 CoW**：前缀树 + 写时复制 + agent 生命周期。测：ToT 显存↓∝分支数。
- [x] **Phase 5 — 分层存储**：GPU/CPU(vLLM offload)/Disk 三层 + 迁移策略。测：超长上下文存活（基线 OOM）。
- [x] **Phase 6 — 生命周期感知淘汰**：step/任务边界感知 vs LRU。测：命中率↑、重算↓。
- [x] **Phase 7 — 多模型/多任务**：单卡跨任务 KV 协调。测：直击「多模型/多任务」40 分子项。
- [x] **Phase 8 — 定点 patch vLLM**：选最高收益特性改内核、重编译、测增量。报告「我们修改了 vLLM 内核」。
- [x] **Phase 9 — 博眼球层**：实时内存仪表盘（Web/TUI）+ matplotlib 图表 + 端到端 demo（基线 OOM vs Mimir）+ 动图/短视频。
- [x] **Phase 10 — 收尾**：一键复现 + 全测试覆盖 + 四份文档定稿（真实数字）+ 仓库整洁。
- [x] **Phase R — TTFT 可观测**：v1 `RequestOutput.metrics` 恒 None，in-tree patch 用 `RequestState.stats` 回填 TTFT；`disable_log_stats=False` 保活 stat pipeline。
- [x] **Phase DeepSeek — 真实轨迹 + LLM-judge**：DeepSeek V4 Pro 产真实 agent 轨迹作 benchmark 工作负载；DeepSeek-flash 作裁判量化压缩保真度（替代「≈持平」手 wave）。
- [x] **Phase BC — 创新核心：block-class 类别感知 KV 管理**【夺冠差异化】：给 KV 块打语义类别标签（system/user/reasoning/tool_result），按类别优先级淘汰——tool_result/system 0 损失、reasoning 优先回收。5 单测 + 真实引擎演示 + probe 召回佐证。

## 指标口径（固定，对应 docs/测试报告.md §4）

`new_prefill_tokens`（核心，真实减少必需 KV）、`ttft_ms`、`used_blocks`（观测）、`e2e_latency_s`、`throughput_tok_per_s`、`task_success_rate`；
`peak_gpu_mem_alloc_gib` 仅反映 vLLM 预分配固定 KV 池，不随优化变，不作核心证据。
**优化前后同一硬件配置、同一模型、同一 seed。**

> 注：早期 used_blocks→0（lifecycle 主动回收）已删除——系任务结束瞬时计数偷换。真信号用 new_prefill 真减 + 避免 OOM。

## 消融实验（每个特性都要进）

baseline → 逐特性叠加 → 全开，记录显存/延迟/成功率曲线，支撑「应用效果」40 分。
