from __future__ import annotations

import time
from flask import Blueprint, request

from cva_runtime.api.responses import err, ok
from cva_runtime.api.runtime import get_runtime
from cva_runtime.control_plane.approvals import issue_approval_token
from cva_runtime.control_plane.audit_log import hash_args


ops_bp = Blueprint("ops_bp", __name__)


def _require_runtime():
    runtime = get_runtime()
    if runtime.system_instance is None:
        return None, err("service_unavailable", "System is not running.", http_status=503)
    return runtime, None


@ops_bp.post("/api/incidents/detect")
def api_incidents_detect():
    runtime, error_response = _require_runtime()
    if error_response:
        return error_response

    data = request.get_json(silent=True) or {}
    incident = (data.get("incident") or data.get("command") or "").strip()
    if not incident:
        return err("invalid_request", "'incident' is required.", http_status=400)

    task_id = runtime.create_task("Incident dispatched to planner")
    directive = {
        "type": "AGENT_PERFORM_TASK",
        "agent_name": data.get("agent_name") or "ProtoAgent_Planner_instance_1",
        "task_description": incident,
        "task_type": data.get("task_type") or "UserCommand",
        "task_id": task_id,
        "mission_type": data.get("mission_type") or "general_planning",
    }

    try:
        runtime.enqueue_directive(directive)
    except Exception as e:
        runtime.update_task_status(task_id, "failed", f"Failed to inject directive: {e}")
        return err("inject_failed", str(e), http_status=500)

    return ok({"task_id": task_id, "queue_status": "processing"}, runtime_state="online")


@ops_bp.post("/api/plan")
def api_plan():
    runtime, error_response = _require_runtime()
    if error_response:
        return error_response

    data = request.get_json(silent=True) or {}
    goal = (data.get("goal") or data.get("high_level_goal") or "").strip()
    if not goal:
        return err("invalid_request", "'goal' is required.", http_status=400)

    task_id = runtime.create_task("Planning cycle queued")
    directive = {
        "type": "INITIATE_PLANNING_CYCLE",
        "planner_agent_name": data.get("planner_agent_name") or "ProtoAgent_Planner_instance_1",
        "high_level_goal": goal,
        "mission_type": data.get("mission_type") or "general_planning",
        "task_id": task_id,
        "cycle_id": f"api_plan_{int(time.time())}",
    }

    try:
        runtime.enqueue_directive(directive)
    except Exception as e:
        runtime.update_task_status(task_id, "failed", f"Failed to queue planning cycle: {e}")
        return err("inject_failed", str(e), http_status=500)

    return ok({"task_id": task_id, "queue_status": "processing"}, runtime_state="online")


@ops_bp.post("/api/act")
def api_act():
    runtime, error_response = _require_runtime()
    if error_response:
        return error_response

    data = request.get_json(silent=True) or {}
    task_description = (data.get("task") or data.get("task_description") or "").strip()
    if not task_description:
        return err("invalid_request", "'task' is required.", http_status=400)

    task_id = runtime.create_task("Action queued")
    directive = {
        "type": "AGENT_PERFORM_TASK",
        "agent_name": data.get("agent_name") or "ProtoAgent_Worker_instance_1",
        "task_description": task_description,
        "task_type": data.get("task_type") or "ExecuteAction",
        "mission_type": data.get("mission_type") or "general_planning",
        "tool_name": data.get("tool_name"),
        "tool_args": data.get("tool_args"),
        "task_id": task_id,
    }

    try:
        runtime.enqueue_directive(directive)
    except Exception as e:
        runtime.update_task_status(task_id, "failed", f"Failed to queue action: {e}")
        return err("inject_failed", str(e), http_status=500)

    return ok({"task_id": task_id, "queue_status": "processing"}, runtime_state="online")


@ops_bp.post("/api/verify")
def api_verify():
    runtime = get_runtime()
    data = request.get_json(silent=True) or {}
    task_id = (data.get("task_id") or "").strip()
    if not task_id:
        return err("invalid_request", "'task_id' is required.", http_status=400)

    task = runtime.get_task(task_id)
    if not task:
        return err("not_found", f"Task '{task_id}' not found.", http_status=404)

    return ok({"task_id": task_id, "task": task})


@ops_bp.post("/api/report")
def api_report():
    runtime = get_runtime()
    data = request.get_json(silent=True) or {}

    task_id = (data.get("task_id") or "").strip()
    if task_id:
        task = runtime.get_task(task_id)
        if not task:
            return err("not_found", f"Task '{task_id}' not found.", http_status=404)
        report = {
            "task_id": task_id,
            "status": task.get("status"),
            "summary": (task.get("result") or {}).get("summary"),
            "details": (task.get("result") or {}).get("details", {}),
        }
        return ok({"task_id": task_id, "report": report})

    limit = data.get("limit", 10)
    try:
        limit = max(1, int(limit))
    except Exception:
        limit = 10

    return ok({"report": {"recent_tasks": runtime.get_recent_tasks(limit=limit)}})


@ops_bp.post("/api/approvals/issue")
def api_issue_approval():
    data = request.get_json(silent=True) or {}
    trace_id = (data.get("trace_id") or "").strip()
    tool = (data.get("tool") or "").strip()
    args_hash = (data.get("args_hash") or "").strip()
    agent_id = (data.get("agent_id") or "").strip() or None

    if not args_hash and isinstance(data.get("args"), dict):
        args_hash = hash_args(data.get("args") or {})

    if not trace_id or not tool or not args_hash:
        return err(
            "invalid_request",
            "'trace_id', 'tool', and 'args_hash' are required (or provide raw 'args').",
            http_status=400,
        )

    ttl_seconds = data.get("ttl_seconds")
    try:
        ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
    except Exception:
        return err("invalid_request", "'ttl_seconds' must be an integer.", http_status=400)

    token, expires_in_s = issue_approval_token(
        trace_id=trace_id,
        tool=tool,
        args_hash=args_hash,
        agent_id=agent_id,
        ttl_seconds=ttl_seconds,
    )
    return ok(
        {
            "approval_token": token,
            "expires_in_s": expires_in_s,
            "trace_id": trace_id,
            "tool": tool,
            "args_hash": args_hash,
        }
    )
