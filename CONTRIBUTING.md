# 贡献指南（Contributing）

感谢关注 Mimir —— 面向智能体推理的内存管理系统。本文件说明开发约定与新增优化方向的方式。

## 开发环境

```bash
bash scripts/setup_vllm_binaries.sh   # 首次：重建 vLLM 预编译二进制
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mimir
pip install -e ".[dev]"
source scripts/activate_env.sh        # 每次会话：vLLM flat 接入 + v1 单进程 env
```

详见 [`docs/VLLM_EDITABLE_SETUP.md`](VLLM_EDITABLE_SETUP.md)。

## 代码规范

- **Python 3.9+**（`from __future__ import annotations`），类型注解用新语法（`X | None`）。
- **ruff**（line-length 100）：`make lint` / `make format`。提交前须 `ruff check` 全清。
- 中文注释/docstring 受欢迎（与赛题文档一致），标识符用英文。
- 单元测试（`pytest`）：新增功能补对应测试，无 GPU 跑通逻辑测试。
- 需 GPU 的测试标 `@pytest.mark.gpu` / `@pytest.mark.slow`，`make test-fast` 跳过。

## vLLM in-tree patch（third_party/vllm_flat/）

- vLLM v0.10.2 **拍平为普通目录**（非 submodule），patch 直接改 `third_party/vllm_flat/vllm/v1/...`。
- **只改纯 Python**（不重编 `_C`）；涉及 C++/CUDA 算子签名的改动不支持。
- patch 清单见 [`docs/VLLM_PATCH_INVENTORY.md`](VLLM_PATCH_INVENTORY.md)。
- 新 patch 模式：加 `mimir_*` 方法 + 在 `get_mimir_stats()` 导出计数器 + 在 `docs/VLLM_PATCH_INVENTORY.md` 登记。

## 新增优化方向

1. 外部层（请求侧变换）：在 `src/mimir/<方向>/` 加模块 + `MemoryManager.apply()` 编排 + `tests/`。
2. 引擎层（vLLM in-tree）：在 `third_party/vllm_flat/vllm/v1/` 加 patch + `engine_vllm_v1.py` adapter 方法 + 验证脚本 `scripts/run_phase_<x>.py`。
3. 带 GPU 验证脚本用 subprocess-per-side（避免双引擎叠加 OOM）；指标用 TTFT + new_prefill + used_blocks（E2E 在共享 GPU 噪声大需多次平均，见 `docs/` 与 memory）。

## 提交约定

- Conventional Commits：`feat:` / `fix:` / `docs:` / `test:` / `refactor:` / `chore:` / `perf:`。
- 频繁提交 + `git push`（GitHub 仅 SSH，HTTPS 被屏蔽）。
- benchmark 结果落盘 `benchmark_results/` 并纳入版本控制。

## 硬件与 GPU

- 单卡优先（多卡互联性能差）；跑前 `pick_least_busy_gpu()` 选最空闲单卡。
- GPU 忙→轻量正确性验证；空闲→重 benchmark。
