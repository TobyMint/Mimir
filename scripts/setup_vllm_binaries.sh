#!/usr/bin/env bash
# 一次性脚本：为 fresh clone 重建 vLLM 预编译二进制（third_party/vllm_prebuilt_bin）。
#
# 背景：vllm_prebuilt_bin/ 被 gitignore（二进制不入库）。新 clone 后需跑此脚本从
# vllm==0.10.2 wheel 提取 .so + flash_attn 包，并 symlink 进 third_party/vllm_flat/vllm/。
# 之后 source scripts/activate_env.sh 即可 import vllm。
#
# 用法：bash scripts/setup_vllm_binaries.sh

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mimir 2>/dev/null || true

echo "[1/4] 下载 vllm==0.10.2 wheel（缓存命中则秒下）..."
TMP=$(mktemp -d)
pip download vllm==0.10.2 --no-deps -d "$TMP" -q
WHL=$(ls "$TMP"/vllm-0.10.2-*.whl | head -1)
echo "  wheel: $WHL"

echo "[2/4] 提取 .so + vllm_flash_attn 包..."
mkdir -p "$TMP/extract"
unzip -o -q "$WHL" 'vllm/*.so' 'vllm/vllm_flash_attn/*' -d "$TMP/extract"

echo "[3/4] 复制到 third_party/vllm_prebuilt_bin/..."
PBIN="$REPO/third_party/vllm_prebuilt_bin"
mkdir -p "$PBIN/vllm_flash_attn"
cp -a "$TMP/extract/vllm/"*.so "$PBIN/"
cp -a "$TMP/extract/vllm/vllm_flash_attn/"* "$PBIN/vllm_flash_attn/"

echo "[4/4] symlink 进 third_party/vllm_flat/vllm/..."
VF="$REPO/third_party/vllm_flat/vllm"
for so in _C.abi3.so cumem_allocator.abi3.so _flashmla_C.abi3.so _moe_C.abi3.so; do
  ln -sf "$PBIN/$so" "$VF/$so"
done
rm -rf "$VF/vllm_flash_attn"
ln -sfn "$PBIN/vllm_flash_attn" "$VF/vllm_flash_attn"

rm -rf "$TMP"
echo ""
echo "完成。现在 source scripts/activate_env.sh 即可 import vllm。"
