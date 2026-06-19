#!/usr/bin/env bash
# 激活脚本：进入 vLLM (已拍平为 third_party/vllm_flat 普通目录) + Mimir 工作环境。
#
# 用法：source scripts/activate_env.sh
#
# 做四件事：
# 1. conda activate mimir
# 2. 确保 third_party/vllm_flat 通过 .pth 注册到 site-packages（import vllm 解析到拍平目录）
# 3. 把 torch/lib 加入 LD_LIBRARY_PATH（_C.abi3.so 需找 libtorch.so）
# 4. 设置 v1 单进程（VLLM_USE_V1=1, VLLM_ENABLE_V1_MULTIPROCESSING=0）

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# conda 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mimir

VLLM_FLAT="$REPO_ROOT/third_party/vllm_flat"
# 把 vllm_flat 注册到 site-packages（幂等）。.pth 第一行是绝对路径，site.py 会加到 sys.path。
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
PTH="$SITE_PKGS/mimir_vllm_flat.pth"
if [ -d "$VLLM_FLAT/vllm" ]; then
  echo "$VLLM_FLAT" > "$PTH"
else
  echo "[env] 警告: $VLLM_FLAT/vllm 不存在（未拍平？）" >&2
fi

# 造一个最小 dist-info，让 importlib.metadata.version("vllm") 返回 0.10.2
# （vLLM 平台检测依赖包元数据；.pth 不装元数据，需手动补，否则 UnspecifiedPlatform/device 为空）
DI="$SITE_PKGS/vllm-0.10.2.dist-info"
if [ ! -f "$DI/METADATA" ]; then
  mkdir -p "$DI"
  printf 'Metadata-Version: 2.1\nName: vllm\nVersion: 0.10.2\nSummary: vLLM (Mimir-patched flat copy, 0.10.2)\n' > "$DI/METADATA"
  printf 'mimir-flat\n' > "$DI/INSTALLER"
  printf 'vllm/__init__.py,,\n' > "$DI/RECORD"
fi

# 顺带清理可能残留的旧 editable 产物（namespace 污染）
rm -rf "$SITE_PKGS/__editable___vllm"*.py "$SITE_PKGS/__editable__.vllm"*.pth 2>/dev/null || true

# vLLM editable 的 _C 需要 torch 的动态库
TORCH_LIB=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

# v1 单进程模式（Mimir 默认用 v1 InprocClient 以获得 block_pool 可观测性）
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

cd "$REPO_ROOT"
VLLM_VER=$(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo "?")
VLLM_PATH=$(python -c 'import vllm;print(vllm.__path__[0])' 2>/dev/null || echo "?")
echo "[env] conda=mimir  vllm=$VLLM_VER"
echo "[env] vllm_path=$VLLM_PATH"
echo "[env] VLLM_USE_V1=$VLLM_USE_V1  VLLM_ENABLE_V1_MULTIPROCESSING=$VLLM_ENABLE_V1_MULTIPROCESSING"
echo "[env] LD_LIBRARY_PATH includes torch/lib"
