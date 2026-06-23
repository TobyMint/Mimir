"""三篇融合:LMCache 集成兼容层(CacheGen 之"搬去哪"轴的底座)。

LMCache(arXiv 2510.09665)是 vLLM 0.10.2 兼容的 KV cache 分层 offload 底座,
作为 Continuum TTL 到期/显存紧时"offload 到 CPU、下一步 reload"的搬运后端,
配合 CacheGen 编解码压缩(后续集成)。

本模块集中处理三件让 LMCache 在我们的 in-tree vLLM 上可用的事:
1. **otel LoggerProvider 兜底**:lmcache 0.4.7 的 logging.py 无条件挂
   `LoggingHandler()`,而 otel-sdk 1.26 在未 `set_logger_provider` 时
   `emit` 会崩(`ProxyLogger has no attribute 'resource'`)。所以 import
   lmcache 前必须先 `set_logger_provider(LoggerProvider())`。
2. **connector 自注册**:lmcache 0.4.7 不自带 entry point,需手动
   `KVConnectorFactory.register_connector("LMCacheConnectorV1Dynamic", ...)`。
3. 装载状态探测:返回 connector 是否可用的报告(装了哪些 dep、版本)。

用法:
    from mimir.lmcache_compat import ensure_lmcache
    report = ensure_lmcache()  # 幂等:首次调用注册 connector 并返回报告
    # 之后构造 VLLMEngineV1 时传 extra={"kv_connector":"LMCacheConnectorV1Dynamic", ...}
"""
from __future__ import annotations

from typing import Any


def _fix_otel_logger_provider() -> None:
    """lmcache 0.4.7 + otel-sdk 1.26 的 import 崩溃兜底:先设一个真 LoggerProvider。

    幂等:只在当前全局 provider 仍是 otel 的 ProxyLoggerProvider(默认惰性代理)时设。
    """
    try:
        from opentelemetry._logs import get_logger_provider, set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider
    except Exception:
        return  # otel 不在,lmcache 的 OTel 分支会自然 no-op,无需处理
    try:
        provider = get_logger_provider()
    except Exception:
        provider = None
    # ProxyLoggerProvider 是 otel API 在未 set 时的惰性代理;set_sdk_provider 替之
    if provider is None or type(provider).__name__ == "ProxyLoggerProvider":
        try:
            set_logger_provider(LoggerProvider())
        except Exception:
            pass  # 已被别处 set 会抛,忽略


_REGISTRY_LOCK = False


def ensure_lmcache() -> dict[str, Any]:
    """幂等确保 LMCache 可用:修 otel + 注册 connector,返回可用性报告。"""
    global _REGISTRY_LOCK
    report: dict[str, Any] = {"available": False, "lmcache_version": None,
                              "vllm_version": None, "connector_registered": False,
                              "error": None}

    _fix_otel_logger_provider()

    try:
        import lmcache  # noqa: F401
        report["lmcache_version"] = getattr(lmcache, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        report["error"] = f"lmcache import failed: {e!r}"
        return report

    try:
        import vllm  # noqa: F401
        report["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        report["error"] = f"vllm import failed: {e!r}"
        return report

    # 注册 connector(幂等)
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
        name = "LMCacheConnectorV1Dynamic"
        if name not in KVConnectorFactory._registry or _REGISTRY_LOCK is False:  # noqa: SLF001
            try:
                KVConnectorFactory.register_connector(
                    name,
                    "lmcache.integration.vllm.lmcache_connector_v1",
                    name,
                )
            except ValueError:
                pass  # 已注册
            _REGISTRY_LOCK = True
        report["connector_registered"] = name in KVConnectorFactory._registry  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        report["error"] = f"connector register failed: {e!r}"
        return report

    report["available"] = True
    return report


def kv_transfer_config(role: str = "kv_both") -> dict[str, Any]:
    """构造给 LLM(kv_transfer_config=...) 的最小 LMCache 配置(CPU offload)。

    role: 'kv_both'(默认,single-node 自存自取)/ 'kv_producer' / 'kv_consumer'。
    CPU offload 大小由 LMCACHE_MAX_LOCAL_CPU_SIZE 环境变量控制(默认 5 GB)。
    """
    return {
        "kv_connector": "LMCacheConnectorV1Dynamic",
        "kv_role": role,
        "kv_transfer_config": {},
    }
