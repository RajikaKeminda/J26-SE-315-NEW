"""
Persistent history of what the autoscaler saw and did (PostgreSQL), one set of rows per cluster. See docs/PERSISTENCE.md.

    sink = persistence.make_sink(mode="live", shadow=True, controller=..., policy=..., k8s=k8s_client, entry=...)
    sink.record_step(payload, extras)       # never raises, never blocks
"""

from src import config
from src.persistence.sink import NullSink, PostgresSink


def settings_snapshot():
    """What the run was configured with - so a stored result can be reproduced."""
    keys = ["METRICS_SOURCE", "STEP_SECONDS", "POD_STARTUP_SECONDS", "METRIC_WINDOW", "MIN_REPLICAS", "MAX_REPLICAS", "CPU_MIN", "CPU_MAX",
            "MEM_MIN_MB", "MEM_MAX_MB", "SLO_LATENCY_MS", "SCALE_DOWN_COOLDOWN_STEPS", "RESIZE_COOLDOWN_STEPS",
            "SCALE_DOWN_AFTER_UP_STEPS", "CIRCUIT_BREAKER_VIOLATIONS", "NODE_MEMORY_RESERVE_MB", "ALLOW_RESIZE",
            "CRITICAL_PATH_SOURCE", "OBSERVE_UNTRACED", "CONTROLLED_SERVICES", "NAMESPACE"]
    return {k: getattr(config, k) for k in keys if hasattr(config, k)}


def make_sink(mode, shadow, controller, policy, slo_ms, k8s=None, entry=None):
    """The sink for this engine: NullSink unless a database URL is set and this mode is meant to be stored."""
    if not config.DATABASE_URL:
        return NullSink()
    if mode not in config.PERSIST_MODES:
        return NullSink(f"mode {mode!r} is not stored (AUTOSCALER_PERSIST_MODES={','.join(config.PERSIST_MODES)})")
    from src.persistence import identity
    from src.persistence.writer import Writer

    def resolve():
        return identity.resolve(k8s, entry)

    writer = Writer(config.DATABASE_URL, resolve,
                    {"mode": mode, "shadow": shadow, "controller": controller, "policy": policy, "slo_ms": slo_ms,
                     "settings": settings_snapshot()},
                    (config.RETENTION_METRICS_HOURS, config.RETENTION_HISTORY_DAYS))
    return PostgresSink(writer)
