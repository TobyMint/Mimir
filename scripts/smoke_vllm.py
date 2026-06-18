"""vLLM smoke test：加载 Qwen3-4B-Instruct-2507 并生成（Phase 0）。

需在 mimir 环境运行：
    source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mimir
    python scripts/smoke_vllm.py

自动选最空闲单卡（ADR-002/003）。退出码 0 表示通过。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.engine_vllm import EngineConfig, VLLMEngine
from mimir.gpu import as_env, pick_least_busy_gpu, snapshot_env, to_json
from mimir.metrics import MetricsCollector


def main() -> int:
    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU：当前无单卡空闲 >=6GiB，回退非 GPU 工作。")
        return 2
    os.environ.update(as_env(g))
    print(f"Using GPU {g.index} ({g.name}), free {g.mem_free_gib:.1f}GiB", flush=True)

    cfg = EngineConfig(
        model="/data/models/Qwen3-4B-Instruct-2507",
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        enable_prefix_caching=True,
        max_model_len=4096,
    )
    eng = VLLMEngine(cfg, device=0)
    col = MetricsCollector(device=0)
    with col.track("smoke_qwen3_4b") as c:
        msgs = [
            {"role": "system", "content": "你是中文助手。"},
            {"role": "user", "content": "用一句话解释什么是 KV Cache 复用。"},
        ]
        txt, n = eng.chat(msgs, max_tokens=64, temperature=0.0)
        c.mark_first_token()
        c.add_output_tokens(n)
        c.success = True
        kv = eng.kv_usage()
    m = col.metrics()
    print("=== 生成结果 ===")
    print(repr(txt))
    print("=== 指标 ===")
    print(
        json.dumps(
            {**m.to_dict(), "kv": kv, "gpu": to_json(snapshot_env())},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
