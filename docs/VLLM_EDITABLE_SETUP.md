# vLLM 0.10.2 editable-install setup（Mimir in-tree patch 基座）

> 本文件记录如何把 vLLM v0.10.2 源码以 **editable + 纯 Python（不重编 `_C`）** 的方式接入 Mimir，
> 使 `import vllm` 指向 `third_party/vllm/` 源码树，便于 in-tree patch。
> 这是 Phase A 的产物，后续所有 vLLM 内核 patch 都基于此。

## 一键激活

```bash
source scripts/activate_env.sh
```

该脚本做三件事：
1. `conda activate mimir`
2. 把 `torch/lib` 加入 `LD_LIBRARY_PATH`（editable 的 `_C.abi3.so` 需找 `libtorch.so`）
3. 设置 `VLLM_USE_V1=1` + `VLLM_ENABLE_V1_MULTIPROCESSING=0`（v1 单进程，父进程可观测 `block_pool`）

## 安装原理（从零复现）

### 1. clone vLLM v0.10.2 为 submodule
```bash
git submodule add git@github.com:vllm-project/vllm.git third_party/vllm   # SSH（HTTPS 被屏蔽）
cd third_party/vllm && git checkout v0.10.2
```

### 2. editable 安装（跳过 `_C` 重编）
```bash
pip install setuptools_scm                       # 构建依赖（--no-build-isolation 需自带）
VLLM_TARGET_DEVICE=empty pip install -e third_party/vllm --no-build-isolation --no-deps
```
- `VLLM_TARGET_DEVICE=empty` → `setup.py:_is_empty()` 跳过 C++/CUDA 编译（`_C.abi3.so` 不重编）。
- `--no-deps` → 不动已装的 torch 2.8.0+cu128。
- 产物：`__editable__.vllm-0.10.2+empty.pth` → `import vllm` 解析到 `third_party/vllm/vllm/`。

### 3. 接入预编译二进制（`_C` 等）
editable 不会带 wheel 里的 `.abi3.so`。从缓存 wheel 提取并 symlink：
```bash
pip download vllm==0.10.2 --no-deps -d /tmp/vllm_wheel_dl
unzip /tmp/vllm_wheel_dl/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl 'vllm/*.so' 'vllm/vllm_flash_attn/*' -d /tmp/extract

mkdir -p third_party/vllm_prebuilt_bin
cp -a /tmp/extract/vllm/*.so third_party/vllm_prebuilt_bin/                      # _C / _moe_C / _flashmla_C / cumem_allocator
cp -a /tmp/extract/vllm/vllm_flash_attn third_party/vllm_prebuilt_bin/           # flash_attn 包（Python + _vllm_fa*_C）

# symlink 到 clone 的 vllm/
for so in third_party/vllm_prebuilt_bin/*.so; do
  ln -sf "$PWD/$so" "third_party/vllm/vllm/$(basename $so)"
done
rm -rf third_party/vllm/vllm/vllm_flash_attn   # 删 clone 的空占位目录
ln -sfn "$PWD/third_party/vllm_prebuilt_bin/vllm_flash_attn" third_party/vllm/vllm/vllm_flash_attn
```
- 二进制与 torch 2.8.0+cu128 ABI 匹配（同一 wheel 构建），**无需 nvcc 重编**。
- `third_party/vllm_prebuilt_bin/` 与 symlink 的 `*.so` 已在 `.gitignore`（不入库）。

### 4. 清理 namespace 污染
原 wheel 卸载后会残留空 `site-packages/vllm/` 目录，作为 namespace 包遮蔽 editable finder：
```bash
rm -rf /data/xbw/conda_envs/mimir/lib/python3.11/site-packages/vllm
rm -rf /data/xbw/conda_envs/mimir/lib/python3.11/site-packages/vllm-0.10.2.dist-info
```

## 验证（Phase A gate）

```bash
source scripts/activate_env.sh
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="/data/models/Qwen3-4B-Instruct-2507", dtype="bfloat16",
          gpu_memory_utilization=0.55, enable_prefix_caching=True, max_model_len=2048)
# v1 InprocClient 遍历（父进程可观测 block_pool）
bp = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
print("num_gpu_blocks:", bp.num_gpu_blocks, "free:", bp.get_num_free_blocks())
print(llm.chat([{"role":"user","content":"hi"}], SamplingParams(max_tokens=8), use_tqdm=False)[0].outputs[0].text)
PY
```
预期：`engine_core` 为 `InprocClient`，`block_pool` 可读，`num_gpu_blocks≈1780`，生成正确。

## 关键路径

| 项 | 值 |
| --- | --- |
| vLLM 源码树 | `third_party/vllm/vllm/`（editable，改即生效） |
| 预编译二进制 | `third_party/vllm_prebuilt_bin/`（gitignored） |
| 激活脚本 | `scripts/activate_env.sh` |
| v1 单进程开关 | `VLLM_USE_V1=1` `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| block_pool 遍历 | `llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool` |

## 限制
- 仅纯 Python patch（不重编 `_C`）。涉及 C++/CUDA 算子签名的改动不在此模式支持范围。
- `vllm_flash_attn` 包整体来自 wheel（上游源码不含），symlink 接入。
