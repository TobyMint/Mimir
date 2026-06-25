# PROGRESS — 实时进度

> **续上时先读本文件。** 记录当前阶段、已完成、进行中、阻塞、下一步、环境与 GPU 选择。

**当前阶段**：全部完成。vLLM v0.10.2 in-tree patch（block-class 创新 + CoW + fp8 降级 + TTFT 回填）。**lifecycle 主动回收机制已删除**（used_blocks→0 系偷换概念，经自审移除）。真卖点：工具外置/压缩真实减少必需 KV（new_prefill -83%/-80%、TTFT -93%）、避免 OOM 崩溃（baseline 第5轮 OOM vs Mimir 存活20轮）、**block-class 创新核心（Phase BC）**、DeepSeek V4 Pro 真实轨迹 + LLM-judge 保真。108 测试通过。

**最后更新**：2026-06-20

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

**当前完整度**：Phase A-D/F/R/BC + DeepSeek。vLLM v0.10.2 拍平为普通目录 `third_party/vllm`（纯 Python in-tree patch），外部优化层（压缩/外置/分层/CoW），**创新核心 block-class 类别感知淘汰（Phase BC）**，**DeepSeek V4 Pro 真实轨迹 + LLM-judge 保真 A/B**。lifecycle 主动回收机制（Phase C/E/I/J/L/M/O/P）已删除（used_blocks→0 偷换概念）。108 测试，一键复现。真卖点：new_prefill -83%/-80%、TTFT -93%、避免 OOM、block-class 创新。

**如何启动**：`source scripts/activate_env.sh`（fresh clone 先 `bash scripts/setup_vllm_binaries.sh`）。

**可继续推进的方向（按价值排序）**：
1. **block-class 在真实 agent 框架验证**：当前 block-class 优势（probe 召回 +48）偏小，可在 BFCL/τ-bench 等真实 agent 基准放大「类别淘汰保住工具结果」的收益。
2. **国产硬件抽象落地**：`mimir/hardware/` 目前是骨架；可加 CUDA/DTK/CANN 设备抽象 + 真实降级测试（赛题鼓励异构）。
3. **llama.cpp 后端适配**：赛题允许 vLLM/llama.cpp 二选一；加 llama.cpp 后端验证泛化（评分 20）。
4. **更长上下文生存 demo（视频/动图）**：把分层存储的「baseline OOM vs Mimir 存活 20 轮」做成可演示动图。
5. **避免 OOM 场景强化**：构造显存真紧张到 native 崩溃的场景，正面证明 Mimir 靠外置+分层存活（补「显存极限」维度正面证据）。

> 注：原 Phase P/M「并发压测/重工作负载 A/B」方向已废弃——回收机制删除后，并发吞吐场景 Mimir 不占优（见 2026-06-21 会话总结）。

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
- 7 个 in-tree patch 文件（144 处 Mimir 标记）：B 块级统计 / C 任务边界回收 / D CoW 记账 / E per-block pin / F fp8 降级 / G mimir 策略 / I pin_hits / J reclaim_evictable / L 自动回收 / P lifecycle-aware 分配 / R v1 TTFT 回填 / **BC block-class 类别感知淘汰（创新核心）**。
- 四个决定性引擎级 A/B（patched v1 vs 原生，used_blocks）：M 单agent 74→0 / O 3agent并发 14→0 / P KV池压力 27→0 / **Q 工具调用并发 262→0（最强一击）**。
- 多模型泛化（Qwen3-1.7B + 4B 验证 lifecycle+CoW）。
- 硬件抽象层（CUDA/ROCm/Ascend/Cambricon/CPU 降级链 + fp8 探测）。
- 统一入口 `MemoryManager.run_turn_with_engine`（外部层 + 引擎层协同）。
- task-success A/B：7/8==7/8 字节一致，delta=0（不降质量）。
- 文档：技术演示 / RESULTS_SUMMARY / CONTRIBUTING / patch清单 / editable安装指南。
- 复现：`setup_vllm_binaries.sh`（fresh clone）+ `reproduce.sh` + 2 demo GIF。

**2026-06-20 新增（夺冠冲刺）**：
- **创新核心 Phase BC**：tool-call 感知 per-block KV 类别管理（block_id→{system,user,reasoning,tool_result}，按类别优先级淘汰）。5 单测 + 真实引擎 `evict(57) 只淘汰 reasoning` 验证。研究确认无论文做过，夺冠差异化。
- **DeepSeek V4 Pro 真实轨迹 + LLM-judge**：前沿模型产真实 agent 轨迹作 benchmark 工作负载（native 崩 vs Mimir used=0，匹配步 TTFT −49%~−84%）；DeepSeek-flash 裁判量化压缩保真（能跑场景 10/10==10/10，full 超长崩而 Mimir 可跑）。
- **Phase R v1 TTFT 可观测**：output_processor 回填 RequestMetrics + disable_log_stats=False。
- **ngram speculative decoding A/B**：训练无关 decode 加速路径。
- 全部 benchmark 在空闲卡 0 重测，数据一致可信。

**2026-06-21 回收机制删除（诚实自审）**：
- **lifecycle 主动回收 / per-block pin 全套删除**：经用户审视，used_blocks→0 系任务结束后瞬时计数偷换——推理时该占仍占、回收重算反而拖慢服务、对用户无益。并发压测三种模式（一次性批量/多轮/请求潮）作证：0.9 大池子下 native 靠 preemption 默默扛住，Mimir 回收在吞吐反因重算少完成请求——省显存计数器未换服务收益。
- 代码删除：block_pool 的 finish_task/reclaim_evictable/pin 全套、scheduler 自驱动回收、engine_v1 回收接口、Phase C/E/I/J/L/M/O/P/Q 脚本与 test_vllm_lifecycle。
- **保留真东西**：block-class 创新（BC）、CoW（D）、fp8 降级（F）、TTFT 回填（R）、工具外置/压缩/分层、DeepSeek trace+judge。
- 叙事转向：从「used_blocks →0」转向「真实减少必需 KV（new_prefill -83%/-80%、TTFT -93%）+ 避免 OOM 崩溃（能跑 vs 崩）+ block-class 创新」。
- 文档全量清扫：README/RESULTS_SUMMARY/测试报告/技术方案/技术演示/PATCH_INVENTORY/系统设计/部署指南 均更新，回收叙事降级为「诚实边界」说明。

**当前状态**：108 测试通过，ruff clean，git clean & synced，评分四维全覆盖 + 创新核心（block-class）+ 诚实评测 + 诚实自审。

**下一会话可推进**：llama.cpp 后端、国产硬件真实测试、长上下文生存视频、block-class 在更大模型/真实 agent 框架（如 BFCL/τ-bench）验证。

## 2026-06-24/25 三篇融合（最新方向，supersede block-class 定位）

> 本节为最新方向。早期 block-class/lifecycle 叙事（06-18~06-21）已被三篇融合 supersede，以本节 + `docs/三篇融合技术报告.md` / `评测与边界.md` 为准。定调见 memory `mimir-core-idea-tiered-kv`。

**方向定调**：不自造机制，做 Continuum（何时留/放）+ LMCache（搬去哪）+ CacheGen（怎么压）三篇融合，agent 工具调用场景统一 KV 放置管线。block-class 不再作为夺冠核心（降为三机制之一），对外仍叫 Mimir。

**落地**（commit d9070f3 / 97f882f / 0fe2dc7 / e0b6f73 / de6c5e8 / 3e3ccaf / 985e6b6）：
- Continuum TTL port 进 `"mimir"` 策略：pin 用 vLLM 原生 `block_pool.touch` 保活（增 ref_cnt + 移出 free queue，不动 ref_cnt 手动增减），13 单测 + 冒烟。
- LMCache 0.4.7 接入：`lmcache_compat.py` 修 otel LoggerProvider + connector 自注册。
- CacheGen serde：LMCache 0.4.7 自带编解码，2.88× 压缩验证（`test_cachegen_serde.py`）。
- vLLM 原生 SharedStorageConnector + 前缀匹配补丁：绕开 LMCache hash 黑盒（vLLM 0.10.2 缺 builtin hash），支持 agent 多轮 prompt 增长 + 容错 + 非互斥 store/load。
- 突破 InprocClient 同步限制：in-process `add_request`+`step` 交错造争用，保父进程可观测（异步 server 下统计在子进程读不到）。

**收益（干扰强度扫描，2026-06-26，`benchmark_results/interference_sweep.{json,png}`）**：
- native 命中率随压力退化 90.1%→90.1%→37.5%→0%（none/weak/medium/strong）——反驳"打地板"：无压 native 自命中 90%，0% 是真实压力退化。
- pin 增量随压力 0→+41%→+78%（weak 零增量→medium +41.1%→strong +78.3%），TTFT 降 70%（KV 留 GPU 免 reload）。pin 不创造命中，只在 native 会丢 KV 时保住它。
- pin+SSC 兜底：重压 pin TTL 到期释放 KV，SSC reload 兜底，命中率 +9.7%（pin 78.3% → 88%）。弱压 pin 不到期，SSC 负收益（88.1% < native 90.1%）。
- pin+SSC 初版退化为 pin（SSC 空转，Inject=0），已修：取消 pin `unpin_requests_regular` 的 waiting 绕过（让 TTL 真到期）+ SSC 每轮 store（不只第一轮）。修复后 Inject KV=1728，pin+SSC 78.3% → 88%。
- **三步走（strong 档）**：native 0%/1470ms/224s → +pin 78.3%/442ms/269s → +pin+SSC 88%/1733ms/984s。命中率/重算单调提升（0→78→88%）；TTFT/total 非单调（pin 最低，pin+SSC 被 reload+store 拖累——memory 兜底慢于 GPU 保活）。**pin（快）+ pin+SSC（全）互补**。

**文档**：`docs/三篇融合技术报告.md`（方法+结果）、`docs/三篇融合评测与边界.md`（评测口径+诚实边界）。

**待办**：PROGRESS/RESULTS_SUMMARY 早期 block-class 叙事的全面对齐（当前仅末尾本节 + 三篇融合文档反映最新方向，主体仍为 06-21 状态）；跨进程 SSC 验证独占价值；CacheGen serde 接进 SSC 数据流。
