"""用 DeepSeek V4 Pro 生成真实 agent 多步轨迹。

DeepSeek 充当 agent：system + user → 生成 → 解析 [TOOL: ...] → 执行 mock 工具 →
结果喂回 → 再生成 → ... → [FINAL: ...]。每一步记录完整消息序列，落盘为 trace。

这些 trace 作为 benchmark 工作负载，相比弱模型（Qwen3-4B）自驱动的轨迹更真实：
- 工具调用更合理、推理链更长（前沿模型的 agent 行为）
- replay 时 native / Mimir 跑**同一份 trace**，A/B 干净（同上下文、同 token）

输出：benchmark_results/traces/<task>.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.agent_loop import (  # noqa: E402
    MOCK_TOOLS,
    TASKS,
    TOOL_DESCRIPTIONS,
    parse_tool_call,
)
from benchmarks.deepseek_client import chat as ds_chat  # noqa: E402


def gen_trace(task: dict, *, model: str = "deepseek-v4-pro", tool_result_kb: int = 5,
              max_steps: int = 12) -> dict:
    """跑一个 DeepSeek-driven agent loop，记录完整轨迹。

    返回 dict：{task, system, user, steps: [{step, role_seq, assistant_text,
    tool_called, tool_result_bytes, tool_result_raw}], messages (完整终态)}.
    """
    system = task["system"] + "\n\nAvailable tools:\n" + "\n".join(TOOL_DESCRIPTIONS)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["user"]},
    ]
    steps: list[dict] = []
    final_answer: str | None = None
    total_tool_bytes = 0

    for step_i in range(max_steps):
        t0 = time.perf_counter()
        text = ds_chat(
            messages,
            model=model,
            max_tokens=512,
            temperature=0.0,
        )
        dt = time.perf_counter() - t0

        tool_name, tool_arg, final = parse_tool_call(text)
        step_rec = {
            "step": step_i,
            "assistant_text": text,
            "tool_called": tool_name,
            "tool_arg": tool_arg,
            "tool_result_bytes": 0,
            "deepseek_latency_s": round(dt, 3),
            "context_msgs": len(messages),
        }

        if final is not None:
            final_answer = final
            steps.append(step_rec)
            break

        if tool_name is not None and tool_name in MOCK_TOOLS:
            if tool_name == "search":
                tool_result = MOCK_TOOLS[tool_name](tool_arg or "", result_size_kb=tool_result_kb)
            else:
                tool_result = MOCK_TOOLS[tool_name](tool_arg or "")
            step_rec["tool_result_bytes"] = len(tool_result)
            step_rec["tool_result_raw"] = tool_result
            total_tool_bytes += len(tool_result)
            messages.append({"role": "assistant", "content": text})
            # 用 role=user 承载工具结果（前缀标注），避免 OpenAI 兼容 API 对 role=tool
            # 强制要求 tool_call_id 的校验；同时本地 Qwen3-4B replay 也能直接复用。
            messages.append(
                {"role": "user", "content": f"[TOOL_RESULT {tool_name}]\n{tool_result}"}
            )
        else:
            messages.append({"role": "assistant", "content": text})
            if step_i >= 1 and steps and steps[-1]["tool_called"] is None:
                messages.append(
                    {"role": "user", "content": "Please use a tool or provide [FINAL: answer]."}
                )

        steps.append(step_rec)
    else:
        final_answer = "(max steps reached without final answer)"

    return {
        "task": task["name"],
        "system": system,
        "user": task["user"],
        "model": model,
        "steps": steps,
        "final_answer": final_answer,
        "num_steps": len(steps),
        "total_tool_data_bytes": total_tool_bytes,
        "messages": messages,  # 完整终态消息序列（replay 用）
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--tool-kb", type=int, default=5, help="mock search 结果大小 KB")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out-dir", default="benchmark_results/traces")
    ap.add_argument("--tasks", default="", help="逗号分隔任务名，空=全部")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = set(args.tasks.split(",")) if args.tasks else None
    tasks = [t for t in TASKS if not sel or t["name"] in sel]
    print(f"Generating traces with {args.model} for {[t['name'] for t in tasks]}", flush=True)

    for task in tasks:
        print(f"\n=== {task['name']} ===", flush=True)
        try:
            trace = gen_trace(
                task, model=args.model, tool_result_kb=args.tool_kb, max_steps=args.max_steps
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}", flush=True)
            continue
        p = out_dir / f"{task['name']}.json"
        p.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"  {trace['num_steps']} steps, tool_data={trace['total_tool_data_bytes']}B, "
            f"final={trace['final_answer'][:80]}",
            flush=True,
        )
        print(f"  saved: {p}", flush=True)
    print("GEN_TRACES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
