"""Phase E 评测：生命周期 bounded 的 per-block KV-pin（区别于 Continuum）。

Mimir pin 语义（区别于 Continuum）：
- 触发：agent 轮边界（Mimir 知道何时暂停），非 Continuum 的「解析工具调用文本 + 估时」
- 边界：lifecycle-bounded（pin 到同 agent 下一轮开始，无时间猜测），非 time-bounded
- 粒度：per-block（仅 system+history 前缀；中间 scratch 仍可淘汰），非 whole-request
- 组合：pin 与 Phase C lifecycle evictor 协同（PINNED 块 finish_task 不Reclaim）

验证：agent A 跑完 pin 其前缀块；agent B 跑（占Memory）；再调 A 的下一轮，
确认 A 的 pinned 前缀块未被Reclaim（mimir_block_lifecycle 中仍 pinned / 前缀块仍在）。
Comparison「不 pin」时 A 前缀可能被压力淘汰（下轮需重算）。

输出：benchmark_results/phase_e_pin_<model>.json

用法：python scripts/run_phase_e_pin.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimir.engine_vllm import EngineConfig
from mimir.engine_vllm_v1 import VLLMEngineV1
from mimir.gpu import as_env, pick_least_busy_gpu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU")
        return 2
    os.environ.update(as_env(g))
    print(f"GPU {g.index}", flush=True)

    eng = VLLMEngineV1(
        EngineConfig(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_util,
            enable_prefix_caching=True,
            max_model_len=args.max_model_len,
        ),
        device=0,
    )
    _ = eng.llm
    print(f"engine_init={eng.engine_init_seconds:.1f}s", flush=True)

    SYS = "You are a research agent answering about KV cache memory management concisely."

    # agent A 第一轮
    eng.set_current_task("agent_A")
    eng.chat(
        [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "What is prefix caching?"},
        ],
        max_tokens=args.max_tokens,
    )
    a_blocks = eng.mimir_block_pool().mimir_get_task_block_ids("agent_A")
    a_pre = eng.mimir_stats()
    print(
        f"agent_A round1: owns {len(a_blocks)} blocks, used={a_pre.get('used_blocks')}", flush=True
    )

    # Mimir pin A 的前缀块
    pinned = eng.mimir_pin_task_blocks("agent_A")
    lc = eng.mimir_block_pool().mimir_block_lifecycle
    pinned_count = sum(1 for v in lc.values() if v == "pinned")
    print(f"Mimir pinned {pinned} of agent_A's blocks (pinned_count={pinned_count})", flush=True)

    # agent B 跑（不同前缀，占Memory）
    eng.set_current_task("agent_B")
    eng.chat(
        [
            {
                "role": "system",
                "content": "You are a coding agent. Answer Python questions briefly.",
            },
            {"role": "user", "content": "Write a Python lambda to reverse a list."},
        ],
        max_tokens=args.max_tokens,
    )
    b_stats = eng.mimir_stats()
    print(f"agent_B ran: used={b_stats.get('used_blocks')} (pressure)", flush=True)

    # A 的 pinned 块应仍在（未被 B 压力淘汰）
    a_blocks_after = eng.mimir_block_pool().mimir_get_task_block_ids("agent_A")
    still_pinned = sum(
        1
        for bid in a_blocks_after
        if eng.mimir_block_pool().mimir_block_lifecycle.get(bid) == "pinned"
    )
    print(
        f"agent_A pinned blocks surviving B's pressure: {still_pinned}/{len(a_blocks)}", flush=True
    )

    # A 下一轮（同前缀）— 前缀命中应免重算
    eng.set_current_task("agent_A")
    eng.chat(
        [
            {"role": "system", "content": SYS},
            {"role": "user", "content": "How does prefix caching save memory?"},
        ],
        max_tokens=args.max_tokens,
    )
    a_post = eng.mimir_stats()
    cr = a_post.get("mimir_cow_reuses")
    print(
        f"agent_A round2: used={a_post.get('used_blocks')}, cow_reuses={cr}",
        flush=True,
    )

    summary = {
        "model": Path(args.model).name,
        "agent_a_blocks_round1": len(a_blocks),
        "pinned_blocks": pinned,
        "agent_b_used_blocks": b_stats.get("used_blocks"),
        "pinned_surviving_pressure": still_pinned,
        "agent_a_round2_used": a_post.get("used_blocks"),
        "agent_a_round2_cow_reuses": a_post.get("mimir_cow_reuses"),
        "differentiation_vs_continuum": {
            "trigger": "agent-turn-boundary (not tool-call-text-parse)",
            "bound": "lifecycle (until next turn, not time-estimated)",
            "granularity": "per-block (prefix only, not whole-request)",
            "composes_with": "Phase C lifecycle evictor (PINNED skipped on finish_task)",
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"phase_e_pin_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {jp}")
    print("PHASE_E_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
