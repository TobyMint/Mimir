"""Phase 1 baseline runner：用真实 vLLM 引擎跑三类工作流，记录未优化基线指标。

用法（mimir 环境）：
    python scripts/run_baseline.py [--model PATH] [--gpu-memory-util 0.55]

输出：
- benchmark_results/baseline_<model_name>.json  （结构化指标）
- benchmark_results/baseline_<model_name>_kv.png  （KV 显存柱状图，单条）

注意：baseline 即「不启用 Mimir 任何特性」，仅依赖 vLLM 自带能力（APC 默认开，
作为公平起点——优化前后都跑在同一 vLLM 配置上）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.harness import run_workload
from benchmarks.workloads import all_workloads

from mimir.engine_vllm import EngineConfig, VLLMEngine
from mimir.gpu import as_env, pick_least_busy_gpu, snapshot_env
from mimir.metrics import save_results

DEFAULT_MODEL = "/data/models/Qwen3-4B-Instruct-2507"


def model_tag(path: str) -> str:
    return Path(path).name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU：无单卡空闲 >=6GiB。请协调 GPU 后重试。")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index} ({g.name}), free {g.mem_free_gib:.1f}GiB", flush=True)

    cfg = EngineConfig(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_util,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        use_v1=False,
    )
    eng = VLLMEngine(cfg, device=0)
    _ = eng.llm  # force init
    print(f"engine_init_seconds={eng.engine_init_seconds:.1f}", flush=True)

    results = []
    tag = model_tag(args.model)
    for name, case in all_workloads().items():
        print(f"\n--- baseline workload: {name} ({case.description}) ---", flush=True)
        m = run_workload(eng, case, max_tokens=args.max_tokens, label=f"baseline_{tag}")
        print(json.dumps(m.to_dict(), ensure_ascii=False, indent=2), flush=True)
        results.append(m)

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"baseline_{tag}.json"
    save_results(results, json_path)
    meta = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_memory_util,
        "max_tokens": args.max_tokens,
        "engine_init_s": eng.engine_init_seconds,
        "vllm": _vllm_version(),
        "torch": _torch_version(),
        "gpu_snapshot": snapshot_env(),
    }
    (out_dir / f"baseline_{tag}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n保存结果: {json_path}")
    print(f"保存元数据: {out_dir / f'baseline_{tag}_meta.json'}")
    print("BASELINE_OK")
    return 0


def _vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
