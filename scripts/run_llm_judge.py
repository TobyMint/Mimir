# ruff: noqa: E501, E701, E702
"""LLM-judge fidelity A/B: 证明 Mimir 上下文压缩不牺牲答案正确性。

设计：
对每个 DeepSeek trace，构造 WorkloadCase（system + 多轮对话 + 大工具结果）。
对同一 case 跑两种上下文，让本地 Qwen3-4B 回答任务问题：
  - full：原始未压缩上下文（全量进 KV）
  - mimir：经 ContextCompressor(BALANCED) 压缩后的上下文（少 token 进 KV）
再用 DeepSeek V4 Pro 作为裁判，对两个答案分别评分（相对参考答案，0-10）。

结论读取：Mimir 压缩后分数 ≈ full 分数 => 压缩是无损的，显存/延迟优化未牺牲正确性。
（若 Mimir 分数明显更低，则提示压缩过头，需调档位——也是诚实的工程信号。）

输出：benchmark_results/llm_judge_<model>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.deepseek_client import chat as ds_chat  # noqa: E402
from benchmarks.harness import build_requests  # noqa: E402
from benchmarks.llm_judge import judge_answer  # noqa: E402
from benchmarks.workloads import ConversationTurn, ToolResult, WorkloadCase  # noqa: E402


def trace_to_workload(trace: dict) -> WorkloadCase:
    """把 DeepSeek trace 转成 WorkloadCase（turns=对话，tool_results=工具返回）。"""
    turns: list[ConversationTurn] = []
    tool_results: list[ToolResult] = []
    for m in trace["messages"]:
        role = m["role"]
        content = m["content"]
        if role == "user" and content.startswith("[TOOL_RESULT "):
            name = content.split("]", 1)[0].replace("[TOOL_RESULT ", "").strip()
            raw = content.split("]\n", 1)[1] if "]\n" in content else content
            tool_results.append(ToolResult(name=name, content=raw, tokens_approx=len(raw) // 4))
        elif role in ("system", "user", "assistant"):
            # 真实对话轮（把 user/assistant 交替作为 turns）
            turns.append(ConversationTurn(role=role, content=content))
    return WorkloadCase(
        name=trace["task"],
        description=f"DeepSeek trace for {trace['task']}",
        system=trace.get("system", ""),
        tool_schemas=[],
        turns=turns,
        tool_results=tool_results,
    )


def answer_with_context(eng, case: WorkloadCase, *, max_tokens: int = 300,
                        question: str = "") -> str:
    """用给定 case 的完整累积上下文，让引擎回答任务问题。

    取 build_requests 的最后一条（累积全上下文），并在末尾追加一条「请基于上述上下文
    回答最初问题」的 user turn，让 Qwen3-4B 产出实际答案。
    """
    reqs = build_requests(case, max_tokens=max_tokens)
    if not reqs:
        return ""
    msgs = reqs[-1].messages  # 累积全上下文
    if question:
        msgs = list(msgs) + [
            {"role": "user",
             "content": f"Based on the above conversation and tool results, "
                        f"answer concisely: {question}"}
        ]
    text, _ = eng.chat(msgs, max_tokens=max_tokens, temperature=0.0)
    return text


def main() -> int:
    from mimir.context.compressor import ContextCompressor, Fidelity
    from mimir.engine_vllm import EngineConfig, VLLMEngine
    from mimir.gpu import as_env, pick_least_busy_gpu

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--gpu-memory-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--fidelity", default="balanced", choices=[f.value for f in Fidelity])
    ap.add_argument("--judge-model", default="deepseek-v4-flash")
    ap.add_argument("--trace-dir", default="benchmark_results/traces")
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    g = pick_least_busy_gpu(min_free_gib=6.0)
    if g is None:
        print("NO_FREE_GPU"); return 2
    os.environ.update(as_env(g))
    print(f"GPU {g.index}, free {g.mem_free_gib:.1f}GiB", flush=True)

    trace_dir = Path(args.trace_dir)
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(trace_dir.glob("*.json"))]
    print(f"Traces: {[t['task'] for t in traces]}", flush=True)

    cfg = EngineConfig(model=args.model, dtype="bfloat16",
                       gpu_memory_utilization=args.gpu_memory_util,
                       enable_prefix_caching=True, max_model_len=args.max_model_len, use_v1=True)
    eng = VLLMEngine(cfg, device=0)
    _ = eng.llm
    comp = ContextCompressor(fidelity=Fidelity(args.fidelity), keep_recent_turns=2)

    print("\n=== LLM-judge fidelity A/B (DeepSeek V4 Pro 裁判) ===", flush=True)
    results = []
    for trace in traces:
        case_full = trace_to_workload(trace)
        case_mimir = comp.compress(case_full)
        cs = comp.stats
        ref = trace.get("final_answer", "")
        if not ref or ref.startswith("("):
            # trace 没拿到 final，用 DeepSeek 现场产一个参考答案
            print(f"  {trace['task']}: generating reference answer via DeepSeek...", flush=True)
            ref = ds_chat(
                [{"role": "system", "content": "Answer the user's question precisely."},
                 {"role": "user", "content": trace["user"]}],
                model=args.judge_model, max_tokens=400, temperature=0.0,
            )
        question = trace["user"]
        print(f"  {trace['task']}: full={cs.original_chars}c -> mimir={cs.compressed_chars}c "
              f"(-{cs.char_reduction_pct:.0f}%), asking Qwen3-4B on both...", flush=True)
        try:
            ans_full = answer_with_context(eng, case_full, max_tokens=args.max_tokens, question=question)
        except Exception as e:  # noqa: BLE001
            ans_full = f"(full crashed: {e})"
        try:
            ans_mimir = answer_with_context(eng, case_mimir, max_tokens=args.max_tokens, question=question)
        except Exception as e:  # noqa: BLE001
            ans_mimir = f"(mimir crashed: {e})"
        jf = judge_answer(question, ref, ans_full, model=args.judge_model)
        jm = judge_answer(question, ref, ans_mimir, model=args.judge_model)
        print(f"    full score={jf['score']} correct={jf['is_correct']} | "
              f"mimir score={jm['score']} correct={jm['is_correct']}", flush=True)
        results.append({
            "task": trace["task"], "question": question[:200],
            "reference": ref[:400],
            "compression": {"original_chars": cs.original_chars,
                            "compressed_chars": cs.compressed_chars,
                            "reduction_pct": round(cs.char_reduction_pct, 1),
                            "tool_results_summarized": cs.tool_results_summarized},
            "full_answer": ans_full[:400], "full_score": jf["score"],
            "full_correct": jf["is_correct"], "full_reason": jf["reason"],
            "mimir_answer": ans_mimir[:400], "mimir_score": jm["score"],
            "mimir_correct": jm["is_correct"], "mimir_reason": jm["reason"],
        })

    summary = {"model": Path(args.model).name, "judge": args.judge_model,
               "fidelity": args.fidelity, "results": results}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    jp = out / f"llm_judge_{Path(args.model).name}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fs = [r["full_score"] for r in results if r["full_score"] is not None]
    ms = [r["mimir_score"] for r in results if r["mimir_score"] is not None]
    favg = round(sum(fs) / len(fs), 2) if fs else None
    mavg = round(sum(ms) / len(ms), 2) if ms else None
    verdict = ("无损" if favg and mavg and mavg >= favg - 1.0
               else ("近似无损" if favg and mavg and mavg >= favg - 2.5 else "有损需调档"))
    print(f"\n平均分: full={favg} / Mimir(压缩)={mavg}  => 压缩{verdict}")
    print(f"JSON: {jp}")
    print("LLM_JUDGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
