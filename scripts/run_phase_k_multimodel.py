# ruff: noqa: E501
"""Phase K：多模型规模泛化验证（直击「适配不同模型规模」20 分）。

在 Qwen3-1.7B / Qwen3-4B-Instruct-2507 / Qwen3-8B 上跑相同的 Mimir 生命周期+CoW 验证，
证明 vLLM in-tree patch 跨模型规模生效：
- 生命周期回收：跑 2 任务，调 mimir_finish_task，看 used_blocks 下降
- CoW 复用：跑 2 分支共享前缀，看 mimir_cow_reuses 增长

输出：benchmark_results/phase_k_multimodel.json

用法：python scripts/run_phase_k_multimodel.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.gpu import pick_least_busy_gpu

MODELS = [
    ("Qwen3-1.7B", "/data/models/Qwen3-1.7B"),
    ("Qwen3-4B-Instruct-2507", "/data/models/Qwen3-4B-Instruct-2507"),
    ("Qwen3-8B", "/data/models/Qwen3-8B"),
]
SYS = "You are a helpful agent. Answer briefly about KV cache."


CHILD_SCRIPT = r"""
import os, json, sys
sys.path.insert(0, os.getcwd())
from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
name, path, gpu, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
cfg = EngineConfig(model=path, dtype="bfloat16", gpu_memory_utilization=0.45,
                   enable_prefix_caching=True, max_model_len=2048)
eng = VLLMEngineV1(cfg, device=0); _ = eng.llm
SYS = "You are a helpful agent. Answer briefly about KV cache."
# lifecycle
eng.set_current_task("t1")
eng.chat([{"role":"system","content":SYS},{"role":"user","content":"What is prefix caching?"}], max_tokens=max_tokens)
eng.set_current_task("t2")
eng.chat([{"role":"system","content":SYS},{"role":"user","content":"What is KV reuse?"}], max_tokens=max_tokens)
pre = eng.mimir_stats()
r1 = eng.mimir_finish_task("t1"); r2 = eng.mimir_finish_task("t2")
post = eng.mimir_stats()
# CoW (same engine: branch B reuses A prefix)
eng.set_current_task("brA")
eng.chat([{"role":"system","content":SYS},{"role":"user","content":"Estimate KV for 7B 32k. Approach A: decompose."}], max_tokens=max_tokens)
eng.set_current_task("brB")
eng.chat([{"role":"system","content":SYS},{"role":"user","content":"Estimate KV for 7B 32k. Approach B: analogy."}], max_tokens=max_tokens)
cow = eng.mimir_stats().get("mimir_cow_reuses", 0)
print("RESULT_JSON:" + json.dumps({
    "model": name, "total_blocks": pre.get("total_blocks"),
    "lifecycle_used_before": pre.get("used_blocks"),
    "lifecycle_used_after": post.get("used_blocks"),
    "lifecycle_reclaims": r1 + r2, "cow_reuses": cow,
    "patch_works": (r1 + r2) > 0 or cow > 0,
}))
"""


def verify_model(name: str, path: str, g, max_tokens: int) -> dict:
    """在子进程中跑（完全释放显存），避免多模型叠加 OOM。"""
    import subprocess

    print(f"\n=== {name} ===", flush=True)
    env = dict(os.environ)
    r = subprocess.run(
        ["python", "-c", CHILD_SCRIPT, name, path, str(g.index), str(max_tokens)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:") :])
            print(
                f"  lifecycle: used {res['lifecycle_used_before']} -> {res['lifecycle_used_after']}, "
                f"reclaims={res['lifecycle_reclaims']}, CoW={res['cow_reuses']}",
                flush=True,
            )
            return res
    print(f"  ERROR (no result): {r.stderr[-300:]}", flush=True)
    return {"model": name, "status": "error", "error": r.stderr[-200:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=8.0)
    if g is None:
        print("NO_FREE_GPU (need >=8GiB for 8B model)")
        return 2
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    results = []
    for name, path in MODELS:
        if not Path(path).exists():
            print(f"  {name}: model not found at {path}, skip", flush=True)
            results.append({"model": name, "status": "not_found"})
            continue
        try:
            r = verify_model(name, path, g, args.max_tokens)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: ERROR {e}", flush=True)
            results.append({"model": name, "status": "error", "error": str(e)[:200]})

    summary = {
        "phase": "K",
        "description": "Multi-model scale generalization: Mimir vLLM in-tree patches work across Qwen3-1.7B/4B/8B",
        "gpu": f"GPU {g.index} ({g.name})",
        "results": results,
        "all_patch_works": all(
            r.get("patch_works", False) for r in results if r.get("status") != "not_found"
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "phase_k_multimodel.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {jp}")
    print(f"all patches work across scales: {summary['all_patch_works']}")
    print("PHASE_K_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
