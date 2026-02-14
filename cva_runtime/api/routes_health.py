from __future__ import annotations

from flask import Blueprint

from cva_runtime.api.responses import ok
from cva_runtime.api.runtime import get_runtime


health_bp = Blueprint("health_bp", __name__)


@health_bp.get("/api/health")
def api_health():
    runtime = get_runtime()
    metrics = runtime.runtime_metrics()
    runtime_state = "online" if metrics["loop_alive"] else "starting"
    return ok(metrics, runtime_state=runtime_state)


@health_bp.get("/api/status")
def api_status():
    runtime = get_runtime()
    metrics = runtime.runtime_metrics()
    runtime_state = "online" if metrics["loop_alive"] else "starting"
    return ok({"service": "cva-runtime", **metrics}, runtime_state=runtime_state)
