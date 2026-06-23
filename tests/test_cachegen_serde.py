"""三篇融合:CacheGen(arXiv 2310.07240,SIGCOMM'24)编解码集成验证。

CacheGen 的编解码器(delta + 分层量化 + 算术编码,把 KV 压成 bitstream)已作为
LMCache 的 storage serde 后端随 LMCache 0.4.7 一起发布
(``lmcache.v1.storage_backend.naive_serde.cachegen_encoder.CacheGenSerializer``)。
本测试验证:对**我们主力模型 Qwen3-4B 的真实 KV 形状**,
CacheGenSerializer/Deserializer 能正确 round-trip 且产生可观压缩比。

不依赖 GPU 跑模型:直接构造真实形状的合成 KV 张量(2, 36, 256, 1024 bf16,
= 2[K/V] × 36 层 × 256 token × 1024 hidden),测 encode→byte 大小、decode→还原。
模型名走 CacheGenConfig.from_model_name 的 AutoConfig 默认回退(Qwen3-4B: 36 层 ≥10
→ 前 10 层 K 用 32 bins、后续 16 bins;V 前 2 层 32、后续 16)。

诚实边界:
- 合成张量是随机的,真实 KV 因 token-wise locality 压缩比通常更高(CacheGen 论文
  实测 3.5–4.3×)。这里测的是「编码器能跑通 + 压缩比 > 1.5× + 解码形状还原」,
  非真实 KV 的精度。
- 真实 KV 精度由 CacheGen 论文保证(delta+分层量化在长上下文 QA 上 <2% 精度损失);
  本测试不重测精度,只验证集成正确性。
"""
from __future__ import annotations

import pytest

# 必须在 import lmcache 之前修 otel provider(lmcache 0.4.7 import 时挂
# LoggingHandler,otel-sdk 未 set provider 会崩 ProxyLogger no resource)。
from mimir.lmcache_compat import _fix_otel_logger_provider

_fix_otel_logger_provider()

lmcache = pytest.importorskip("lmcache")  # 跳过无 LMCache 环境

MODEL_PATH = "/data/models/Qwen3-4B-Instruct-2507"
NLAYERS = 36
KV_HIDDEN = 1024  # 8 KV heads × 128 head_dim
CHUNK = 256


class _DummyTensorMemoryObj:
    """最小 MemoryObj 替身:只暴露 CacheGenSerializer.serialize 读取的 .tensor。"""

    def __init__(self, tensor):
        self.tensor = tensor


def _build_serde(serde_type: str = "cachegen"):
    """构造给定 serde 类型的 serializer/deserializer(Qwen3-4B 配置)。"""
    import torch

    try:
        from transformers import AutoConfig  # noqa: F401
    except Exception as exc:
        pytest.skip(f"transformors unavailable: {exc}")

    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.naive_serde import CreateSerde

    cfg = LMCacheEngineConfig.from_defaults(chunk_size=CHUNK)
    meta = LMCacheMetadata(
        model_name=MODEL_PATH,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=[2, NLAYERS, 8, 128],
        use_mla=False,
        role=None,
    )
    return CreateSerde(serde_type, meta, cfg)


def _raw_chunk_bytes() -> int:
    """原始(未压缩)KV chunk 字节数:2 × 36 × 256 × 1024 × bf16(2B)。"""
    return 2 * NLAYERS * CHUNK * KV_HIDDEN * 2


def test_cachegen_roundtrip_and_compression():
    """合成真实形状 KV → encode → decode,验证压缩比 > 1.5× 且形状还原。"""
    import torch

    serializer, deserializer = _build_serde("cachegen")

    # KV_2LTD 布局:[2(K/V), L, T, hidden]。bf16 合成张量。
    kv = torch.randn(2, NLAYERS, CHUNK, KV_HIDDEN, dtype=torch.bfloat16)
    mem = _DummyTensorMemoryObj(kv)

    encoded = serializer.serialize(mem)
    enc_bytes = len(encoded.raw_data)

    raw = _raw_chunk_bytes()
    ratio = raw / enc_bytes
    assert enc_bytes < raw, f"encoded ({enc_bytes}B) 不应大于原始 ({raw}B)"
    assert ratio > 1.5, f"压缩比 {ratio:.2f}× 过低,期望 >1.5×"

    decoded = deserializer.deserialize(encoded)
    assert decoded.tensor is not None
    assert decoded.tensor.shape[0] == 2
    assert decoded.tensor.shape[1] == NLAYERS
    assert decoded.tensor.shape[2] == CHUNK
    assert decoded.tensor.shape[-1] == KV_HIDDEN


def test_cachegen_smaller_than_naive():
    """CacheGen 编码字节数应显著小于 naive(naive = 直接拷贝原始张量字节)。"""
    import torch

    serializer_cg, _ = _build_serde("cachegen")

    kv = torch.randn(2, NLAYERS, CHUNK, KV_HIDDEN, dtype=torch.bfloat16)
    mem = _DummyTensorMemoryObj(kv)
    cg_bytes = len(serializer_cg.serialize(mem).raw_data)
    # naive serde = 直接拷贝原始张量(无压缩),故字节数 = raw
    naive_bytes = _raw_chunk_bytes()
    assert cg_bytes < naive_bytes, (
        f"CacheGen ({cg_bytes}B) 应小于 naive ({naive_bytes}B)")
