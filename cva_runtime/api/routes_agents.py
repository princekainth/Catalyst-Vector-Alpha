from __future__ import annotations

from flask import Blueprint, request

from cva_runtime.api.responses import err, ok
from cva_runtime.api.runtime import get_runtime


agents_bp = Blueprint("agents_bp", __name__)


@agents_bp.post("/api/agents/spawn")
def api_agents_spawn():
    runtime = get_runtime()
    data = request.get_json(silent=True) or {}

    purpose = data.get("purpose", "").strip()
    if not purpose:
        return err("invalid_request", "'purpose' is required.", http_status=400)

    context = data.get("context", {})
    if not isinstance(context, dict):
        return err("invalid_request", "'context' must be an object.", http_status=400)

    if runtime.system_instance is None or not hasattr(runtime.system_instance, "handle_spawn_request"):
        return err("service_unavailable", "System is not running.", http_status=503)

    try:
        with runtime.agent_instances_lock:
            result = runtime.system_instance.handle_spawn_request(
                purpose=purpose,
                context=context,
                parent_agent="api_v1",
            )

        if isinstance(result, dict):
            return err(
                "spawn_validation_failed",
                result.get("error", "Spawn validation failed."),
                details={
                    "suggestions": result.get("suggestions", []),
                    "hint": result.get("hint", ""),
                },
                http_status=400,
            )

        if result:
            return ok({"agent_id": result})

        return err("spawn_failed", "Spawn failed.", http_status=500)

    except Exception as e:
        return err("internal_error", str(e), http_status=500)
