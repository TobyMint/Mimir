"""``mimir.hardware.device`` 单元测试（纯逻辑，monkeypatch 后端探测）。"""

from __future__ import annotations

from mimir.hardware.device import (
    BackendKind,
    DeviceBackend,
    detect_all_backends,
    hardware_report,
    pick_backend,
    recommend_engine_config,
)


def _mk(kind, available, **kw):
    return DeviceBackend(kind, available, **kw)


def test_supports_fp8_hopper_only() -> None:
    assert _mk(BackendKind.CUDA, True, capability=(9, 0)).supports_fp8() is True
    assert _mk(BackendKind.CUDA, True, capability=(8, 6)).supports_fp8() is False  # 3090
    assert _mk(BackendKind.CUDA, True, capability=None).supports_fp8() is False
    assert _mk(BackendKind.CPU, True).supports_fp8() is False


def test_kv_dtype_options() -> None:
    assert _mk(BackendKind.CUDA, True, capability=(9, 0)).kv_cache_dtype_options() == [
        "auto",
        "fp8",
        "fp8_e4m3",
    ]
    assert _mk(BackendKind.CUDA, True, capability=(8, 6)).kv_cache_dtype_options() == ["auto"]


def test_pick_backend_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        "mimir.hardware.device.detect_all_backends",
        lambda: [_mk(BackendKind.CUDA, False), _mk(BackendKind.ROCM, False)],
    )
    b = pick_backend()
    assert b.kind is BackendKind.CPU
    assert b.available is True


def test_pick_backend_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        "mimir.hardware.device.detect_all_backends",
        lambda: [
            _mk(BackendKind.CUDA, True, device_count=2, capability=(8, 6)),
            _mk(BackendKind.ROCM, False),
        ],
    )
    b = pick_backend()
    assert b.kind is BackendKind.CUDA
    assert b.device_count == 2


def test_hardware_report_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "mimir.hardware.device.detect_all_backends",
        lambda: [_mk(BackendKind.CUDA, True, device_count=1, name="RTX 3090", capability=(8, 6))],
    )
    r = hardware_report()
    assert r["chosen_backend"] == "cuda"
    assert r["fp8_kv_supported"] is False
    assert len(r["backends"]) == 1
    assert r["backends"][0]["name"] == "RTX 3090"


def test_recommend_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "mimir.hardware.device.detect_all_backends",
        lambda: [_mk(BackendKind.CUDA, False)],
    )
    cfg = recommend_engine_config()
    assert cfg["dtype"] == "float32"
    assert cfg["enforce_eager"] is True
    assert cfg["kv_cache_dtype"] == "auto"


def test_recommend_cuda_3090_no_fp8(monkeypatch) -> None:
    monkeypatch.setattr(
        "mimir.hardware.device.detect_all_backends",
        lambda: [_mk(BackendKind.CUDA, True, capability=(8, 6))],
    )
    cfg = recommend_engine_config()
    assert cfg["dtype"] == "bfloat16"
    assert cfg["kv_cache_dtype"] == "auto"
    assert "gracefully fall back" in cfg.get("_mimir_note", "")


def test_recommend_cuda_hopper_fp8(monkeypatch) -> None:
    monkeypatch.setattr(
        "mimir.hardware.device.detect_all_backends",
        lambda: [_mk(BackendKind.CUDA, True, capability=(9, 0))],
    )
    cfg = recommend_engine_config()
    assert cfg["kv_cache_dtype"] == "fp8"


def test_detect_all_backends_runs(monkeypatch) -> None:
    # 真实探测（无卡环境应返回 CUDA.available=False 或 True）
    backs = detect_all_backends()
    assert any(b.kind is BackendKind.CUDA for b in backs)
    assert any(b.kind is BackendKind.ASCEND for b in backs)
