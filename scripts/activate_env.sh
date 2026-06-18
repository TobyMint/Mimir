#!/usr/bin/env bash
# 激活脚本：进入 vLLM editable-install + Mimir 工作环境。
#
# 用法：source scripts/activate_env.sh
#
# 做三件事：
# 1. conda activate mimir
# 2. 把 torch/lib 加入 LD_LIBRARY_PATH（editable-install 的 _C.abi3.so 需要找 libtorch.so）
# 3. 切到仓库根目录

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# conda 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mimir

# vLLM editable-install 的 _C 需要 torch 的动态库
TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# 可选：v1 单进程模式（Mimir 默认用 v1 InprocClient 以获得 block_pool 可观测性）
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

cd "$REPO_ROOT"
echo "[env] conda=mimir  vllm=$(python -c 'import vllm;print(vllm.__version__)')  "
echo "[env] VLLM_USE_V1=$VLLM_USE_V1  VLLM_ENABLE_V1_MULTIPROCESSING=$VLLM_ENABLE_V1_MULTIPROCESSING"
echo "[env] LD_LIBRARY_PATH includes torch/lib"
