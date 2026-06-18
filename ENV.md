# ENV — 环境与复现

> 一处记录「怎么把环境跑起来」。装/换依赖后更新本文件。

## 机器

- OS：Ubuntu 20.04（Linux 5.15）
- GPU：**4× NVIDIA RTX 3090（24GB）**，驱动 570.133.07（CUDA 12.8 运行时），nvcc 12.6
  - ⚠️ 多卡互联性能差 → **只用单卡**，动态选最空闲的一张
- CPU：48 核；内存：**251GB**（适合 CPU offload 层 + 多任务并发）
- 磁盘：`/data` 2.0TB 空闲，`/home` 954GB 空闲

## conda 环境 `mimir`

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mimir
```

- 路径：`/data/xbw/conda_envs/mimir`
- Python：3.11.15
- torch：2.6.0+cu126（cuda_avail=True）
- vLLM：_（安装中，完成后填入实际版本）_
- numpy：_（随 vLLM 安装补齐）_

> ⚠️ 共享的 miniconda base 装了**坏的 torch 2.11.0+cu130**（driver 太旧，cuda=False）。
> 我们**只在新 `mimir` 环境里工作**，绝不碰 base。

## 模型（本地已有，无需下载）

| 模型 | 路径 | 用途 |
| --- | --- | --- |
| Qwen3-1.7B | `/data/models/Qwen3-1.7B` | 泛化验证（小） |
| Qwen3-4B-Instruct-2507 | `/data/models/Qwen3-4B-Instruct-2507` | **主力开发** |
| Qwen3-8B | `/data/models/Qwen3-8B` | 泛化验证（大） |

## 运行命令

```bash
conda activate mimir
# 选单卡（示例 GPU 1；实际跑前 nvidia-smi 选最空闲）
export CUDA_VISIBLE_DEVICES=1
# 安装项目（开发模式）
pip install -e ".[dev]"
# 测试
make test            # 或 make test-fast
# benchmark
python -m benchmarks.run
```

## 网络

- pypi ✅、GitHub SSH（`git@github.com:`）✅；GitHub HTTPS 被屏蔽 → 一律用 SSH。
- SMTP 出站 ✅：smtp.163.com:465（用于邮件通知）。
