# PROGRESS — 实时进度

> **续上时先读本文件。** 记录当前阶段、已完成、进行中、阻塞、下一步、环境与 GPU 选择。

**当前阶段**：Phase A-M 全部完成。vLLM v0.10.2 in-tree patch（10 文件）+ 多模型泛化 + 决定性 A/B（Mimir used=0 vs 原生 69）。92 测试通过。

**最后更新**：2026-06-18

## 已完成 ✅

- 环境：conda `mimir`（py3.11 + torch 2.8.0+cu128 + vLLM 0.10.2，cuda 可用）
- 邮件通道（163）已验证
- charter（GOAL/ROADMAP/PROGRESS/ENV/DECISIONS）
- 指标体系：`mimir/metrics.py`（MetricsCollector）、`mimir/engine_vllm.py`（v0 单进程，可读 block_manager + 显存）、per-request 精确指标（TTFT/cached/prefill）
- GPU 动态选卡：`mimir/gpu.py`（pick_least_busy_gpu）
- 工作流生成器 + harness + 画图（`mimir/plots.py`）
- **Phase 0**：vLLM 跑通 Qwen3-4B 生成正确中文
- **Phase 1 baseline**（真实数据，commit c62471a）：

  | workload | mem(GiB) | TTFT(ms) | E2E(s) | tput | cached | new_prefill |
  |---|---|---|---|---|---|---|
  | multi_turn | 11.53 | 46.3 | 12.76 | 60.2 | 1744 | 398 |
  | tool_call | 11.66 | 309.5 | 2.50 | 45.2 | 2256 | 3792 |
  | multi_stage | 11.52 | 28.0 | 6.34 | 60.5 | 768 | 40 |

  → tool_call 的 310ms TTFT + 3792 new prefill（大工具返回）= 上下文压缩 + 工具外置的主战场

- **Phase 2 核心**：`mimir/context/compressor.py`（Fidelity×3，静态去重/历史摘要/工具摘要）+ 6 测试通过

## 进行中 🔄

- Phase 2 评测运行：baseline vs context-compression（BALANCED）对比（后台 b8994kkqy）

## 下一步 ➡️

1. Phase 2 评测完成 → 落盘 JSON+PNG，填测试报告
2. Phase 3：工具数据外置（tool_offload）—— 大返回存外存，KV 仅留引用
3. Phase 4：分支 CoW
4. 后续分层存储 / 生命周期淘汰 / 多任务 / vLLM patch / 博眼球层

## 阻塞 🚧

- `peak_kv_used_blocks` 在单请求顺序跑下为 null（vLLM 请求结束后释放块）。
  应对：用 per-request 的 `num_prompt_tokens - num_cached_tokens`（新进 KV 的 token 数）作为 KV 增量的代理指标（已采集）。块峰值需并发或多请求同时存活才能观测，Phase 4/7 再深入。

## 环境与运行备忘

- 激活：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mimir`
- 运行：`python scripts/run_baseline.py` / `python scripts/run_phase2_context.py`
- GPU：跑前选最空闲单卡（≥6GiB）；全占则退避
- 模型：`/data/models/Qwen3-{1.7B,4B-Instruct-2507,8B}`
- 发邮件：`python3 ~/.claude/hooks/notify_email.py "<标题>"`

## 关键节点

- 2026-06-18 charter 提交（92940b7）
- 2026-06-18 Phase 0 vLLM 跑通 Qwen3-4B（5f09b4e）
- 2026-06-18 Phase 1 baseline 捕获（c62471a）
- 2026-06-18 Phase 2 上下文压缩：tool_call TTFT 307→27ms (-91%)（4aa2b1c）
- 2026-06-18 Phase 3 工具外置：tool_call TTFT 304→29ms (-90%)，43776 字符外置（05f2859）
- 2026-06-18 Phase 4 分支 CoW：CoW 记账省 78.7% KV，真实 APC 复用 52.3%（e021b63）
- 2026-06-18 Phase 5 分层存储核心：三层 GPU/HOST/DISK + 迁移（c2fccb6）
- 2026-06-18 Phase 6 生命周期淘汰：主动回收 100%，LRU 被动淘汰 0 感知（f947907）
- 2026-06-18 新方向 fp8 KV 量化：KV 容量 2.06x（fac8ff2）
- 2026-06-18 新方向 多任务协调：并发峰值平稳，回收 71.4%（6d644e2）
- 2026-06-18 新方向 LLM 语义压缩 + 实时仪表盘（54bb7cc）
- 2026-06-18 统一 MemoryManager 管线 + 端到端 demo：上下文省 94.1%（f053752）
- 2026-06-18 动态仪表盘 GIF（04b66b1）
- 2026-06-19 Phase A：vLLM v0.10.2 submodule + editable + v1 InprocClient 可观测（d09f059）
- 2026-06-19 Phase B：v1 块级统计导出 + Mimir v1 adapter（e118232/b719ae5）
- 2026-06-19 Phase C：lifecycle 主动回收 — 10 块 used_blocks 10→0（2372b36/6127d97）
- 2026-06-19 Phase D：分支 CoW 复用记账 — 4 分支 9 次复用（eb377c3）
- 2026-06-19 Phase E：per-block KV-pin — 3/3 pinned 块压力下存活（ce6bed3）
- 2026-06-19 Phase F：fp8 优雅降级到 bf16（aed4f43）
- 2026-06-19 Phase G：'mimir' 调度策略 + MimirRequestQueue（845344b）

## 续上指南（给下一个会话）

**当前完整度**：Phase A-O 全部完成。vLLM v0.10.2 拍平为普通目录 `third_party/vllm`（9-10 文件 in-tree patch），外部优化层 11 方向，两个决定性引擎级 A/B（Phase M 单 agent used 69→0；Phase O 3-agent 并发 used 14→0），多模型泛化（1.7B/4B），92 测试，一键复现。

**如何启动**：`source scripts/activate_env.sh`（fresh clone 先 `bash scripts/setup_vllm_binaries.sh`）。

**可继续推进的方向（按价值排序）**：
1. **Phase P — 真实 KV 压力淘汰验证**：构造超 KV 池的并发长上下文，验证 mimir 策略下 EVICTABLE 块被优先淘汰（vs LRU-活跃块）。Phase J 写了 reclaim_evictable 但未在真实压力路径练过。
2. **更重的工作负载 A/B**：Phase M 用的是 10 轮问答；可换成「工具调用密集 + 长上下文」更贴合赛题 tool_call 场景，放大外部压缩/外置的显存收益。
3. **国产硬件抽象落地**：`mimir/hardware/` 目前是骨架；可加 CUDA/DTK/CANN 设备抽象 + 真实降级测试（赛题鼓励异构）。
4. **llama.cpp 后端适配**：赛题允许 vLLM/llama.cpp 二选一；加 llama.cpp 后端验证泛化（评分 20）。
5. **更长上下文生存 demo（视频/动图）**：把 Phase 5 分层存储的「baseline OOM vs Mimir 存活 20 轮」做成可演示动图。

**重要约定（见 CLAUDE.md + memory）**：
- 频繁 commit + push（每次逻辑里程碑）。
- GitHub 仅 SSH（HTTPS 被屏蔽）。
- GPU 忙→轻量正确性；空闲→重 benchmark。
- vLLM patch 是纯 Python（`third_party/vllm/vllm/v1/...`），不重编 `_C`。
- 真实指标用 TTFT + new_prefill + used_blocks；E2E 在共享 GPU 上噪声大需多次平均。

**邮件通知**：`python3 ~/.claude/hooks/notify_email.py "<标题>"`（163，`x2406862525@163.com`）。

## 2026-06-19 会话总结（vLLM in-tree patch 完整推进）

**本会话完成**（Phase A-Q + 拍平 + 硬件）：
- vLLM v0.10.2 从 submodule 拍平为普通目录 `third_party/vllm`（纯 Python patch，不重编 `_C`，`.pth`+dist-info 接入）。
- 10 个 in-tree patch 文件：B 块级统计 / C 任务边界回收 / D CoW 记账 / E per-block pin / F fp8 降级 / G mimir 策略 / I pin_hits / J reclaim_evictable / L 自动回收 / P lifecycle-aware 分配。
- 四个决定性引擎级 A/B（patched v1 vs 原生，used_blocks）：M 单agent 69→0 / O 3agent并发 14→0 / P KV池压力 27→0 / **Q 工具调用并发 262→0（最强一击）**。
- 多模型泛化（Qwen3-1.7B + 4B 验证 lifecycle+CoW）。
- 硬件抽象层（CUDA/ROCm/Ascend/Cambricon/CPU 降级链 + fp8 探测）。
- 统一入口 `MemoryManager.run_turn_with_engine`（外部层 + 引擎层协同）。
- task-success A/B：7/8==7/8 字节一致，delta=0（不降质量）。
- 文档：技术演示 / RESULTS_SUMMARY / CONTRIBUTING / patch清单 / editable安装指南。
- 复现：`setup_vllm_binaries.sh`（fresh clone）+ `reproduce.sh` + 2 demo GIF。

**当前状态**：101 测试通过，ruff clean，git clean & synced，114+ commits，评分四维全覆盖。

**下一会话可推进（PROGRESS 续上指南已列）**：llama.cpp 后端、更重工作负载 A/B、国产硬件真实测试、长上下文生存视频。
