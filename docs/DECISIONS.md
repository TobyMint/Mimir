# DECISIONS — 架构决策记录（ADR）

> 记录已拍板的关键决策与理由，避免重复争论。新决策按时间倒序追加。

## ADR-001 vLLM 集成路线：先包裹，后定点 patch（2026-06-18）

**决策**：先用 vLLM 公共 API 构建 Mimir 智能体内存层（Phase 1–7），跑通端到端并产出真实对比数字；再对最高收益特性（分支 CoW / tree attention / KV 记账 hook）做**针对性** vLLM 内核改动（Phase 8）。

**理由**：
- vLLM 体量大、改内核风险高（C++/CUDA + Python），改坏可能导致整个引擎跑不通。
- 「包裹优先」保证：**无论后续 patch 成否，都有一套可交付、可测的系统**。
- 报告里仍可写「我们修改了 vLLM 内核」（Phase 8 patch），保留加分点。
- 用户已确认「放开手脚做到最好」，但不牺牲可交付性。

## ADR-002 单卡运行，动态选最空闲卡（2026-06-18）

**决策**：只用单卡；每次 GPU 任务前 `nvidia-smi` 选最空闲的一张，设 `CUDA_VISIBLE_DEVICES`。

**理由**：用户说明多卡互联性能差，多卡收益有限甚至负向；3090 单卡 24GB 对 Qwen3-4B 足够。

## ADR-003 GPU 调度：忙→轻量正确性，空闲→重 benchmark（2026-06-18）

**决策**：GPU 被别人占用时，只做轻量正确性验证（不打扰他人）；完全空闲时才跑完整 benchmark（复现性更稳）。用户负责协调空闲时段。

**理由**：机器多人共用；benchmark 需要稳定显存/算力，抢跑结果不可信。

## ADR-004 主力模型 Qwen3-4B-Instruct-2507（2026-06-18）

**决策**：开发主力用 Qwen3-4B-Instruct-2507（本地已有），泛化验证用 Qwen3-1.7B + Qwen3-8B。

**理由**：4B 对 3090 单卡友好（权重 ~8GB，留足 KV 空间）；Instruct 版贴合 agent 场景；三档规模直击「适配不同模型规模」20 分。

## ADR-005 进度通知：GitHub + PROGRESS.md + 关键节点邮件（2026-06-18）

**决策**：进度以 GitHub 提交 + 实时 `PROGRESS.md` 为主；关键节点用 `~/.claude/hooks/notify_email.py` 发邮件（163，`x2406862525@163.com`）。不强制配 Stop hook（避免与 goal hook 冲突），改用手动在里程碑发信。

**理由**：零额外配置；邮件脚本已验证可用；hook 叠加 goal 的阻塞语义有不确定性，手动更可控。

## ADR-006 环境隔离：独立 conda 环境 `mimir`（2026-06-18）

**决策**：所有工作在 `conda activate mimir`（Python 3.11）内进行，不动共享 base。

**理由**：base 装了坏的 torch 2.11.0+cu130（cuda=False）；隔离环境避免影响他人、便于复现与回滚。

## 待决（开放问题）

- **vLLM 具体版本**：待安装后确认与 torch 的兼容组合，回填 `ENV.md`。
- **分支 CoW 是否能完全靠 vLLM APC 实现，还是必须内核 patch**：Phase 4 探明后定。
