# vLLM 0.10.2 接入方式（拍平为普通目录）—— Mimir in-tree patch 基座

> 本文件记录如何把 vLLM v0.10.2 源码以 **普通目录 + 纯 Python（不重编 `_C`）** 的方式接入 Mimir，
> 使 `import vllm` 指向 `third_party/vllm_flat/` 源码树，便于 in-tree patch。
> vLLM 已**拍平为普通 tracked 目录**（不再是 git submodule），只含可导入的 `vllm/` 包 + 构建文件。
> 这是 Phase A 的产物，后续所有 vLLM 内核 patch 都基于此。

## 一键激活

```bash
source scripts/activate_env.sh
```

该脚本做四件事：
1. `conda activate mimir`
2. 写 `.pth`（`site-packages/mimir_vllm_flat.pth` 内容为 `third_party/vllm_flat` 绝对路径）→ `import vllm` 解析到拍平目录
3. 写最小 dist-info（`vllm-0.10.2.dist-info`）→ `importlib.metadata.version("vllm")` 返回 0.10.2（vLLM 平台检测依赖包元数据；.pth 不装元数据，需手动补，否则 `UnspecifiedPlatform` → device 为空）
4. `torch/lib` 加入 `LD_LIBRARY_PATH`（symlinked `_C.abi3.so` 需找 `libtorch.so`）+ 设 `VLLM_USE_V1=1` `VLLM_ENABLE_V1_MULTIPROCESSING=0`（v1 单进程，父进程可观测 `block_pool`）

## 为什么是普通目录（而非 submodule / editable install）

- **非 submodule**：避免 submodule 复杂度（`clone --recursive`、detached HEAD、pointer bump），patch 后的 vLLM 源码直接在仓库里可审计。
- **非 `pip install -e`**：`setup.py` 用 `setuptools_scm` 从 git 推断版本，拍平目录不是 git repo → 无法 editable 安装。改用 `.pth` 直接挂到 sys.path（editable 的本质），再补 dist-info 解决元数据。
- **不重编 `_C`**：预编译二进制（`.abi3.so` + `vllm_flash_attn` 包）来自缓存的 0.10.2 wheel，symlink 进 `vllm_flat/vllm/`，与 torch 2.8.0+cu128 ABI 匹配。

## 从零复现（一次性，已做完，记录在此）

### 1. （历史步骤）clone + checkout + 拍平
```bash
git submodule add git@github.com:vllm-project/vllm.git third_party/vllm     # SSH
cd third_party/vllm && git checkout v0.10.2 && cd -
mkdir -p third_party/vllm_flat
rsync -a --exclude='__pycache__' --exclude='*.pyc' third_party/vllm/vllm/ third_party/vllm_flat/vllm/
cp third_party/vllm/{setup.py,pyproject.toml,.gitignore} third_party/vllm_flat/
git rm third_party/vllm && rm -rf .git/modules/third_party/vllm && rm .gitmodules
```

### 2. 接入预编译二进制（`.so` + flash_attn 包）
```bash
pip download vllm==0.10.2 --no-deps -d /tmp/vllm_wheel_dl
unzip /tmp/vllm_wheel_dl/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl 'vllm/*.so' 'vllm/vllm_flash_attn/*' -d /tmp/extract

mkdir -p third_party/vllm_prebuilt_bin
cp -a /tmp/extract/vllm/*.so third_party/vllm_prebuilt_bin/                      # _C / _moe_C / _flashmla_C / cumem_allocator
cp -a /tmp/extract/vllm/vllm_flash_attn third_party/vllm_prebuilt_bin/           # flash_attn 包（Python + _vllm_fa*_C）

# symlink 到拍平目录的 vllm/
TBIN=$(pwd)/third_party/vllm_prebuilt_bin
for so in _C.abi3.so cumem_allocator.abi3.so _flashmla_C.abi3.so _moe_C.abi3.so; do
  ln -sf "$TBIN/$so" third_party/vllm_flat/vllm/$so
done
rm -rf third_party/vllm_flat/vllm/vllm_flash_attn   # 删占位目录
ln -sfn "$TBIN/vllm_flash_attn" third_party/vllm_flat/vllm/vllm_flash_attn
```
- 二进制与 torch 2.8.0+cu128 ABI 匹配（同一 wheel 构建），**无需 nvcc 重编**。
- `third_party/vllm_prebuilt_bin/` 与 symlink 的 `*.so`/`vllm_flash_attn` 已在 `.gitignore`（不入库）。

### 3. 激活（每次会话）
```bash
source scripts/activate_env.sh   # 自动写 .pth + dist-info + LD_LIBRARY_PATH + v1 单进程 env
```

## 验证

```bash
source scripts/activate_env.sh
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="/data/models/Qwen3-4B-Instruct-2507", dtype="bfloat16",
          gpu_memory_utilization=0.55, enable_prefix_caching=True, max_model_len=2048,
          scheduling_policy="mimir")
bp = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
print("num_gpu_blocks:", bp.num_gpu_blocks, "free:", bp.get_num_free_blocks())
print(llm.chat([{"role":"user","content":"hi"}], SamplingParams(max_tokens=8), use_tqdm=False)[0].outputs[0].text)
PY
```
预期：`engine_core` 为 `InprocClient`，`block_pool` 可读，`num_gpu_blocks≈1780`，"Mimir scheduling policy active"，生成正确。

## 关键路径

| 项 | 值 |
| --- | --- |
| vLLM 源码树（拍平） | `third_party/vllm_flat/vllm/`（普通 tracked 目录，改即生效） |
| 预编译二进制 | `third_party/vllm_prebuilt_bin/`（gitignored，symlink 进 vllm_flat） |
| 激活脚本 | `scripts/activate_env.sh`（幂等写 .pth + dist-info） |
| v1 单进程开关 | `VLLM_USE_V1=1` `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| block_pool 遍历 | `llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool` |

## 限制
- 仅纯 Python patch（不重编 `_C`）。涉及 C++/CUDA 算子签名的改动不在此模式支持范围。
- `vllm_flash_attn` 包整体来自 wheel（上游源码不含），symlink 接入。
- `.pth` + dist-info 写在 site-packages（不在仓库），由 `activate_env.sh` 每次幂等写入 —— 因此**新环境/新 clone 后必须先 `source scripts/activate_env.sh`** 才能正确 `import vllm`。
