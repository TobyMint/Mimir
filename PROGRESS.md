# PROGRESS — 实时进度

> **续上时先读本文件。** 记录当前阶段、已完成、进行中、阻塞、下一步、环境与 GPU 选择。

**当前阶段**：Phase 5（分层存储）核心完成；Phase 0-4 已完成有真实数据。下一步：分层存储接入工具外置 + Phase 6 生命周期淘汰

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
- 2026-06-18 进度邮件已发
- 2026-06-18 Phase 4 分支 CoW：CoW 记账省 78.7% KV，真实 APC 复用 52.3%（e021b63）
- 2026-06-18 Phase 5 分层存储核心：三层 GPU/HOST/DISK + 迁移（c2fccb6）
