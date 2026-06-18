"""``mimir.gpu`` 单元测试（不依赖真实 GPU，用 monkeypatch）。"""

from __future__ import annotations

from mimir.gpu import GPUInfo, as_env, pick_least_busy_gpu, query_gpus


def _fake_gpus(monkeypatch) -> None:
    fake = [
        GPUInfo(0, "RTX 3090", 24576, 14000, 10576, 56),  # 较忙
        GPUInfo(1, "RTX 3090", 24576, 4000, 20576, 22),  # 最空闲
        GPUInfo(2, "RTX 3090", 24576, 4500, 20076, 22),
    ]
    monkeypatch.setattr("mimir.gpu.query_gpus", lambda: list(fake))


def test_pick_least_busy_prefers_most_free(monkeypatch) -> None:
    _fake_gpus(monkeypatch)
    g = pick_least_busy_gpu(min_free_gib=2.0)
    assert g is not None
    assert g.index == 1  # 20.5 GiB 空闲，最大


def test_pick_returns_none_when_all_below_threshold(monkeypatch) -> None:
    fake = [GPUInfo(0, "X", 24576, 24000, 576, 90), GPUInfo(1, "X", 24576, 23500, 1076, 80)]
    monkeypatch.setattr("mimir.gpu.query_gpus", lambda: list(fake))
    assert pick_least_busy_gpu(min_free_gib=2.0) is None


def test_prefer_index_used_when_viable(monkeypatch) -> None:
    _fake_gpus(monkeypatch)
    g = pick_least_busy_gpu(min_free_gib=2.0, prefer_index=2)
    assert g is not None and g.index == 2


def test_as_env(monkeypatch) -> None:
    _fake_gpus(monkeypatch)
    g = pick_least_busy_gpu()
    assert as_env(g)["CUDA_VISIBLE_DEVICES"] == "1"
    assert as_env(None) == {}


def test_query_gpus_empty_without_smi(monkeypatch) -> None:
    monkeypatch.setattr("mimir.gpu._nvidia_smi_available", lambda: False)
    assert query_gpus() == []
