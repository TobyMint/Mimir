"""DeepSeek V4 Pro 客户端（OpenAI 兼容）。

用途：
1. 让 DeepSeek V4 Pro 充当 agent，在 mock 工具上跑出**真实的多步轨迹**，
   作为 benchmark 工作负载（替代合成 / 弱模型自驱动的轨迹）。
2. 作为 **LLM-judge**，裁判 Mimir 压缩上下文后答案的正确性（替代「≈持平」手 wave）。

密钥从环境变量 ``DEEPSEEK_API_KEY`` 或仓库根 ``.deepseek_key``（gitignored）读取，
**绝不入库**。默认模型 ``deepseek-v4-pro``。

设计为「无卡可跑」：trace 生成与 judge 都只调 API，不需要本地 GPU。
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from openai import OpenAI  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def _load_api_key() -> str:
    """读取 API key：env 优先，其次仓库根 ``.deepseek_key``。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    # 仓库根的 gitignored 文件
    candidates = [
        Path.cwd() / ".deepseek_key",
        Path(__file__).resolve().parents[1] / ".deepseek_key",
    ]
    for cand in candidates:
        if cand.exists():
            k = cand.read_text(encoding="utf-8").strip()
            if k:
                return k
    raise RuntimeError(
        "DeepSeek API key 未找到：请设置 DEEPSEEK_API_KEY 环境变量，"
        "或在仓库根创建 .deepseek_key（已 gitignore）。"
    )


def make_client(*, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai 库未安装：pip install openai")
    return OpenAI(api_key=_load_api_key(), base_url=base_url)


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    base_url: str = DEFAULT_BASE_URL,
    retries: int = 3,
) -> str:
    """单轮 chat（非流式），返回 assistant 文本。便于 trace/judge 复用。

    带重试：DeepSeek V4 Pro 偶发空回复（temperature=0 + 短输出时），重试并取首个
    非空结果。同时回退读取 ``reasoning_content``（部分模型把内容放该字段）。
    """
    import time as _time

    client = make_client(base_url=base_url)
    last = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            msg = resp.choices[0].message
            text = msg.content or ""
            if not text:
                text = getattr(msg, "reasoning_content", "") or ""
            if text.strip():
                return text
            last = text
        except Exception:
            last = ""
        _time.sleep(0.6 * (attempt + 1))
    return last


def list_models(*, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """列出可用模型（连通性自检）。"""
    client = make_client(base_url=base_url)
    resp = client.models.list()
    return [m.id for m in resp.data]
