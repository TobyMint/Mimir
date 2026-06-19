"""DeepSeek V4 Pro 作为 LLM-judge，评判 Mimir 压缩上下文后答案的正确性。

替代测试报告中「任务成功率 ≈持平」的手 wave——给 native 与 Mimir 的最终答案
各打 0-10 分 + 正确性判定，量化「Mimir 优化是否以质量为代价」。

评判口径：给定任务问题 + 参考答案，对候选答案评分：
correctness (0-10)、is_correct (bool)、reasoning。
模型可能先输出推理再给 JSON；我们取最后出现的 JSON 对象。
"""

from __future__ import annotations

import json
import re

from benchmarks.deepseek_client import chat as ds_chat

JUDGE_SYSTEM = (
    "You are an answer-grading judge. Score the CANDIDATE answer's correctness vs the "
    "REFERENCE on 0-10 (10=fully correct, 5=partial, 0=wrong/empty/crash-message). "
    "Respond with ONLY a JSON object — the very first character must be '{'. "
    "No reasoning, no markdown. Format: "
    '{"score": <number 0-10>, "is_correct": <bool>, "reason": "<one short sentence>"}'
)


def _extract_last_json(raw: str) -> dict | None:
    """取 raw 中最后出现的 JSON 对象（跳过模型可能先输出的推理文本）。"""
    # 贪心匹配最后一个 {...}（允许内部嵌套大括号：用栈式扫描）
    last: dict | None = None
    for m in re.finditer(r"\{", raw):
        depth = 0
        start = m.start()
        for i in range(start, len(raw)):
            c = raw[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cand = raw[start : i + 1]
                    try:
                        last = json.loads(cand)
                    except Exception:
                        pass
                    break
    return last


def judge_answer(task_question: str, reference: str, candidate: str,
                 *, model: str = "deepseek-v4-pro") -> dict:
    """对单个候选答案评分。返回 {score, is_correct, reason, raw}。"""
    user = (
        f"TASK:\n{task_question}\n\n"
        f"REFERENCE ANSWER:\n{reference[:1500]}\n\n"
        f"CANDIDATE ANSWER:\n{candidate[:1500]}\n\n"
        "Score the CANDIDATE's correctness vs the REFERENCE. Output one JSON object only."
    )
    raw = ds_chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        model=model, max_tokens=400, temperature=0.0,
    )
    out = {"score": None, "is_correct": None, "reason": "", "raw": raw[:300]}
    j = _extract_last_json(raw)
    if j and j.get("score") is not None:
        try:
            out["score"] = float(j["score"])
        except (TypeError, ValueError):
            pass
        out["is_correct"] = j.get("is_correct")
        out["reason"] = j.get("reason", "")
    return out
