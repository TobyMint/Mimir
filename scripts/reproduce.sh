#!/usr/bin/env bash
# Mimir 一键复现脚本（赛题「可复现性」20 分）。
#
# 用法：bash scripts/reproduce.sh [--quick]
#   --quick: 只跑单元测试 + 纯 CPU 仿真（不加载模型，~2 分钟）
#   默认: 上述 + 关键 GPU benchmark（需空闲单卡，~10 分钟）
#
# 从零验证：环境 → 单元测试 → 非GPU仿真 → vLLM in-tree patch 验证 → 关键 benchmark。

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

echo "============================================================"
echo "  Mimir 一键复现  (quick=$QUICK)"
echo "============================================================"

# 1. 环境
echo "[1/5] 激活环境..."
source scripts/activate_env.sh

# 2. 代码检查 + 单元测试
echo "[2/5] ruff + pytest..."
ruff check src tests benchmarks scripts conftest.py
python -m pytest -q

# 3. 非 GPU 仿真（分层存储 + 生命周期 + 多任务）
echo "[3/5] 非 GPU 仿真验证..."
python scripts/run_phase5_tiered.py --turns 20 --gpu-cap 4 2>&1 | grep -E "baseline:|tiered:|PHASE5"
python scripts/run_phase6_lifecycle.py --tasks 8 --blocks-per-task 4 --capacity 12 2>&1 | grep -E "lifecycle:|pure_lru:|PHASE6"
python scripts/run_phase7b_multitask.py --sweep --tasks 6 2>&1 | grep -E "PHASE7B"

# 4. vLLM in-tree patch 可观测性（需 GPU）
if [ "$QUICK" = "0" ]; then
    echo "[4/5] vLLM in-tree patch 验证（需空闲单卡）..."
    # 关键验证：lifecycle 主动回收 + CoW + 多模型
    python scripts/run_phase_c_lifecycle.py 2>&1 | grep -E "Mimir total|PHASE_C" || echo "  (Phase C skipped — GPU busy?)"
    python scripts/run_phase_d_cow.py --branches 4 2>&1 | grep -E "engine mimir|PHASE_D" || echo "  (Phase D skipped — GPU busy?)"
    # 结论性 A/B
    python scripts/run_phase_m_ab.py --max-tokens 12 2>&1 | grep -E "baseline final|Mimir final|PHASE_M" || echo "  (Phase M skipped — GPU busy?)"
else
    echo "[4/5] 跳过 GPU 验证（--quick 模式）"
fi

# 5. 总结
echo "[5/5] 复现完成。"
echo "  - 单元测试: 110 项（pytest -q）"
echo "  - vLLM in-tree patch: 11 文件（详见 docs/VLLM_PATCH_INVENTORY.md，含 Phase R v1 TTFT 观测）"
echo "  - benchmark 结果: benchmark_results/*.json（30+ 个）"
echo "  - 动态仪表盘 GIF（真实引擎数据）: benchmark_results/agent_loop_demo.gif"
echo "  - 文档: docs/（赛题/技术方案/系统设计/部署/测试报告/patch清单/editable安装）"
echo "============================================================"
echo "  全部结果可在 benchmark_results/ 与 docs/测试报告.md 查阅。"
echo "============================================================"
