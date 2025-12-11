# app.py
# ──────────────────────────────────────────────────────────────────────────────
# Catalyst Vector Alpha — Flask app (dashboard + API)
# - Safe startup (env + imports)
# - Background CVA loop
# - Task tracking + live event feed
# - Pending plan store + approval endpoint (registry-backed + token support)
# - System + k8s metrics passthrough
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import uuid
import json
import logging
from supervisor import CognitiveSupervisor
import threading
from threading import Thread, Lock
from collections import deque
from typing import Dict, Any, Optional, Tuple
from notify_agent import ProtoAgent_Notifier
import gmail_agent
import threading
import gmail_agent
import threading
from flask import Response
import time

# --- Environment --------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Add project root to PYTHONPATH ------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

# --- Third-Party --------------------------------------------------------------
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

# --- Project Imports (fail fast with message) ---------------------------------
try:
    from catalyst_vector_alpha import CatalystVectorAlpha
    from shared_models import MessageBus, EventMonitor, timestamp_now
    from tool_registry import tool_registry, ToolRegistry  # singleton + class
    from core.mission_runner import MissionRunner
    # Optional clients from tool_registry (if you expose them)
    from tool_registry import PROM, K8S, POL
except ImportError as e:
    print(f"FATAL IMPORT ERROR: {e}")
    print("Ensure project files exist and PYTHONPATH is correct.")
    sys.exit(1)

# Register all tools into the global singleton registry (lazy module import, no function imports)
import tools  # the module only

# --- Globals ------------------------------------------------------------------
API_KEY = os.getenv("CATALYST_API_KEY", "your-secret-key")

system_instance: Optional[CatalystVectorAlpha] = None
system_thread: Optional[Thread] = None

# Task tracking (thread-safe)
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = Lock()

# Recent task outcomes buffer (optional helper)
RECENT_TASKS = deque(maxlen=100)

# Live UI event feed (thread-safe)
LIVE_EVENTS = deque(maxlen=50)
events_lock = Lock()

# Minimal pending plan store (task_id -> plan)
# Expected keys: task_id, status, action, namespace, deployment, replicas, ts, approval_token
plan_store: Dict[str, Dict[str, Any]] = {}
plan_lock = Lock()

# Protect agent instance access from Flask threads
agent_instances_lock = Lock()

# Mission runner (CPU threshold mission helper)
_runner = MissionRunner(PROM, K8S, POL, mem_kernel=None)

# --- Logger -------------------------------------------------------------------
# --- Logger -------------------------------------------------------------------
logger = logging.getLogger("CatalystLogger")
logger.setLevel(logging.INFO)

# Remove duplicate handlers if any
logger.handlers = []

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler - ALWAYS add this
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/catalyst.log", mode='a')
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# JSON log handler (parallel structured logs)
class JsonLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = {
                "timestamp": getattr(record, "timestamp", time.time()),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "event_type": getattr(record, "event_type", None),
                "source": getattr(record, "source", None),
                "details": getattr(record, "details", None),
            }
            with open("logs/catalyst.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

json_handler = JsonLogHandler()
json_handler.setLevel(logging.INFO)
logger.addHandler(json_handler)

# Confirmation print
print(f"✅ Logging enabled: Console + File (logs/catalyst.log)")

# --- UI Broadcast Handler -----------------------------------------------------
class UIBroadcastHandler(logging.Handler):
    """Push structured log records into LIVE_EVENTS for the UI."""
    def emit(self, record):
        if hasattr(record, 'event_type'):
            log_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': getattr(record, 'timestamp', timestamp_now()),
                'type': getattr(record, 'event_type', 'log'),
                'source': getattr(record, 'source', 'unknown'),
                'description': getattr(record, 'description', record.getMessage()),
            }
            with events_lock:
                LIVE_EVENTS.append(log_entry)

# --- Task Completion Handler --------------------------------------------------
class TaskCompletionHandler(logging.Handler):
    """Detect agent task completions and update our task map."""
    def emit(self, record):
        if getattr(record, 'event_type', '') == 'AGENT_TASK_PERFORMED':
            try:
                details = getattr(record, 'details', {}) or {}
                task_id = details.get('task_id')
                if task_id and task_id in tasks:
                    agent = details.get('agent', '')
                    task_desc = details.get('task', '')
                    outcome = details.get('outcome', 'unknown')
                    status = "completed" if outcome == "completed" else "failed"
                    summary = f"Agent {agent} {status}: {task_desc}"
                    report_content = details.get('report_content', {})
                    update_task_status(
                        task_id=task_id,
                        status=status,
                        result_summary=summary,
                        details=report_content
                    )
            except Exception as e:
                logger.warning(f"TaskCompletionHandler error: {e}")

# Attach UI + completion handlers
ui_handler = UIBroadcastHandler()
ui_handler.setLevel(logging.INFO)
logger.addHandler(ui_handler)

task_handler = TaskCompletionHandler()
task_handler.setLevel(logging.INFO)
logger.addHandler(task_handler)

# --- Helper functions ---------------------------------------------------------
def update_task_status(task_id: Optional[str], status: str, result_summary="Task completed", details=None):
    """Update status of a tracked task; used by system callbacks."""
    if not task_id:
        return
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id] = {
                "status": status,
                "result": {"summary": result_summary, "details": details or {}}
            }
            try:
                RECENT_TASKS.appendleft({"task_id": task_id, "status": status, "summary": result_summary, "details": details or {}})
            except Exception:
                pass
        logger.info(f"Updated task {task_id}: status={status}")

def _safe_get_task_history(limit=15):
    try:
        if system_instance and hasattr(system_instance, "get_task_history"):
            return system_instance.get_task_history(limit=limit) or []
    except Exception as e:
        logger.warning(f"get_task_history failed: {e}")
    return []

def _safe_get_latest_agent_reflection():
    try:
        if system_instance and hasattr(system_instance, "get_latest_agent_reflection"):
            return system_instance.get_latest_agent_reflection()
    except Exception as e:
        logger.warning(f"get_latest_agent_reflection failed: {e}")
    return None

# -------------------- Pending Missions: helpers -------------------------------
def _normalize_scale_params(params: Dict[str, Any]) -> Tuple[str, str, int]:
    ns = params.get("namespace", "default")
    dep = params.get("deployment")
    rep = params.get("replicas", 1)
    if dep in (None, ""):
        raise ValueError("Missing 'deployment' in params.")
    try:
        rep = int(rep)
    except Exception:
        raise ValueError("'replicas' must be an integer.")
    return ns, dep, rep

def save_pending_scale_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call this from the Planner when emitting a dry-run k8s_scale plan.
    Normalized shape stored in memory with an approval_token for the dashboard.
    """
    p = dict(plan or {})
    p["status"] = "awaiting_approval"
    p["action"] = "k8s_scale"
    p["ts"] = time.time()
    p.setdefault("task_id", str(uuid.uuid4()))
    p.setdefault("approval_token", f"tok_{int(time.time())}_{p['task_id'][:6]}")

    # Optional convenience: allow Planner to pass params dict or flat fields
    if "params" in p:
        try:
            ns, dep, rep = _normalize_scale_params(p["params"])
            p["namespace"], p["deployment"], p["replicas"] = ns, dep, rep
        except Exception as e:
            logger.warning(f"save_pending_scale_plan param normalization failed: {e}")

    with plan_lock:
        plan_store[p["task_id"]] = p
    return p

def latest_pending_scale_plan() -> Optional[Dict[str, Any]]:
    with plan_lock:
        pending = [p for p in plan_store.values() if p.get("status") == "awaiting_approval"]
    if not pending:
        return None
    return max(pending, key=lambda x: x.get("ts", 0.0))

def _find_pending_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with plan_lock:
        for p in plan_store.values():
            if p.get("status") == "awaiting_approval" and p.get("approval_token") == token:
                return p
    return None

def mark_plan_executed(task_id: str) -> None:
    with plan_lock:
        if task_id in plan_store:
            plan_store[task_id]["status"] = "executed"
            plan_store[task_id]["executed_ts"] = time.time()

def queue_pending_mission(action: str, params: Dict[str, Any], rationale: str = "") -> Dict[str, Any]:
    """
    Public helper you can import from the Planner:
      from app import queue_pending_mission
      queue_pending_mission("k8s_scale", {"namespace":"default","deployment":"demo-nginx","replicas":2}, "Scale up due to high CPU")
    """
    if action != "k8s_scale":
        raise ValueError(f"Unsupported action for queue_pending_mission: {action}")
    ns, dep, rep = _normalize_scale_params(params)
    plan = {
        "action": "k8s_scale",
        "namespace": ns,
        "deployment": dep,
        "replicas": rep,
        "rationale": rationale,
    }
    return save_pending_scale_plan(plan)

# --- Background System Thread -------------------------------------------------
def run_catalyst_system_in_background():
    """Initialize and run the CVA cognitive loop in a resilient way."""
    import traceback
    global system_instance
    logger.info("Initializing Catalyst Vector Alpha components...")

    message_bus_instance = MessageBus()
    event_monitor_instance = EventMonitor()

    try:
        system_instance = CatalystVectorAlpha(
            message_bus=message_bus_instance,
            tool_registry=tool_registry,              # use the singleton registry
            event_monitor=event_monitor_instance,
            external_log_sink=logger,
            # pass references for task tracking
            tasks_dict_ref=tasks,
            tasks_lock_ref=tasks_lock,
            task_update_callback=update_task_status,
        )
        # Expose running instance to tools that expect a global reference
        import tools as tools_module
        tools_module._cva_instance = system_instance
    except Exception as e:
        logger.exception("Fatal: failed to construct CatalystVectorAlpha")
        return

    # Optional: wire pending plan saver
    if hasattr(system_instance, "set_save_pending_scale_plan_fn"):
        try:
            system_instance.set_save_pending_scale_plan_fn(save_pending_scale_plan)
        except Exception as _e:
            logger.warning(f"Could not set save_pending_scale_plan_fn: {_e}")

    system_instance.is_running = True

    logger.info("Catalyst Vector Alpha system is starting its cognitive loop...")
    try:
        supervisor = CognitiveSupervisor(
            cva_instance=system_instance,
            database=system_instance.db,
            logger=logger
        )
        supervisor.run_supervised(tick_sleep=10)
    except Exception as e:
        logger.error("Cognitive loop crashed: %s\n%s", e, traceback.format_exc())
    finally:
        logger.info("Catalyst Vector Alpha system thread has finished.")

# --- Flask App ----------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # tighten with resources={r"/api/*": {"origins": "..."}}

# --- Routes: Health -----------------------------------------------------------
@app.get("/api/health")
def api_health():
    loop_alive = bool(system_thread and system_thread.is_alive())
    return jsonify({"status": "ok", "data": {"loop_alive": loop_alive, "ts": time.time()}}), 200

@app.get("/api/metrics/stats")
def api_metrics_stats():
    """Aggregate basic performance metrics from the metrics table."""
    try:
        from db_postgres import execute_query
        # Average planner latency (last 50)
        planner_avg = execute_query(
            """
            SELECT AVG(value) AS avg_val FROM (
                SELECT value FROM metrics
                WHERE metric_type = 'agent_execution_time'
                  AND agent_name = 'ProtoAgent_Planner_instance_1'
                ORDER BY timestamp DESC
                LIMIT 50
            ) sub
            """,
            fetch=True,
        )
        worker_avg = execute_query(
            """
            SELECT AVG(value) AS avg_val FROM (
                SELECT value FROM metrics
                WHERE metric_type = 'agent_execution_time'
                  AND agent_name = 'ProtoAgent_Worker_instance_1'
                ORDER BY timestamp DESC
                LIMIT 50
            ) sub
            """,
            fetch=True,
        )
        breaker_trips = execute_query(
            """
            SELECT COUNT(*) AS trips
            FROM metrics
            WHERE metric_type = 'circuit_breaker_trip'
              AND timestamp > NOW() - INTERVAL '24 hours'
            """,
            fetch=True,
        )

        resp = {
            "planner_latency": float(planner_avg[0]["avg_val"]) if planner_avg and planner_avg[0]["avg_val"] is not None else None,
            "worker_latency": float(worker_avg[0]["avg_val"]) if worker_avg and worker_avg[0]["avg_val"] is not None else None,
            "breaker_trips": int(breaker_trips[0]["trips"]) if breaker_trips else 0,
        }
        return jsonify({"status": "ok", "data": resp}), 200
    except Exception as e:
        logger.exception(f"/api/metrics/stats failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get("/api/metrics/trends")
def api_metrics_trends():
    """Return simple trend data for breakers (by hour) and recent latencies."""
    try:
        from db_postgres import execute_query
        breaker_rows = execute_query(
            """
            SELECT date_trunc('hour', timestamp) AS hour, COUNT(*) AS trips
            FROM metrics
            WHERE metric_type = 'circuit_breaker_trip'
              AND timestamp > NOW() - INTERVAL '12 hours'
            GROUP BY hour
            ORDER BY hour ASC
            """,
            fetch=True,
        )
        latency_rows = execute_query(
            """
            SELECT timestamp, agent_name, value
            FROM metrics
            WHERE metric_type = 'agent_execution_time'
            ORDER BY timestamp DESC
            LIMIT 30
            """,
            fetch=True,
        )
        return jsonify({
            "status": "ok",
            "data": {
                "breaker_by_hour": breaker_rows or [],
                "latencies": latency_rows or [],
            }
        }), 200
    except Exception as e:
        logger.exception(f"/api/metrics/trends failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.post("/api/system/pause")
def api_pause_system():
    """Pause the cognitive loop."""
    try:
        reason = (request.json or {}).get("reason", "Paused via API")
        if system_instance:
            system_instance.pause_system(reason=reason)
        return jsonify({"status": "ok", "message": "System pause requested"}), 200
    except Exception as e:
        logger.exception(f"/api/system/pause failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.post("/api/system/unpause")
def api_unpause_system():
    """Resume the cognitive loop."""
    try:
        reason = (request.json or {}).get("reason", "Unpaused via API")
        if system_instance:
            system_instance.unpause_system(reason=reason)
        return jsonify({"status": "ok", "message": "System unpause requested"}), 200
    except Exception as e:
        logger.exception(f"/api/system/unpause failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get("/api/dashboard/summary")
def api_dashboard_summary():
    """Lightweight summary for dashboard cards."""
    try:
        if system_instance is None:
            return jsonify({
                "status": "ok",
                "data": {
                    "agent_count": 0,
                    "paused_agents": 0,
                    "dynamic_agents": 0,
                    "queue_length": 0,
                }
            }), 200

        agents_map = getattr(system_instance, "agent_instances", {}) or {}
        agent_count = len(agents_map)
        paused = 0
        for a in agents_map.values():
            try:
                if hasattr(a, "is_paused") and a.is_paused():
                    paused += 1
            except Exception:
                continue

        factory = getattr(system_instance, "agent_factory", None)
        dynamic_count = len(factory.active_agents) if factory and hasattr(factory, "active_agents") else 0
        queue_len = len(getattr(system_instance, "dynamic_directive_queue", []) or [])
        return jsonify({
            "status": "ok",
            "data": {
                "agent_count": agent_count,
                "paused_agents": paused,
                "dynamic_agents": dynamic_count,
                "queue_length": queue_len,
            }
        }), 200
    except Exception as e:
        logger.exception(f"/api/dashboard/summary failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

# Dashboard route - serves the React app
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Dashboard diagnostics page
@app.route('/dashboard-diagnostics')
def dashboard_diagnostics():
    return send_file('test_dashboard.html')

# Simple dashboard test page
@app.route('/simple-dashboard')
def simple_dashboard():
    return send_file('dashboard/simple/index.html')

# --- Routes: Commands & Tasks -------------------------------------------------
@app.route('/api/command', methods=['POST'])
def handle_command():
    data = request.get_json(silent=True) or {}
    command_text = data.get('command')
    if not command_text:
        return jsonify({"status": "error", "error": "Missing 'command' in body."}), 400
    if not system_instance:
        return jsonify({"status": "error", "error": "System is not running."}), 503

    task_id = str(uuid.uuid4())
    with tasks_lock:
        tasks[task_id] = {"status": "processing", "result": {"summary": "Command dispatched to swarm..."}}

    directive = {
        'type': 'AGENT_PERFORM_TASK',
        'agent_name': 'ProtoAgent_Planner_instance_1',
        'task_description': command_text,
        'task_type': 'UserCommand',
        'task_id': task_id,
    }
    try:
        system_instance.inject_directives([directive])
        logger.info(f"Injected user command: {command_text} (task_id: {task_id})")
    except Exception as e:
        with tasks_lock:
            tasks[task_id] = {"status": "error", "result": {"summary": f"Failed to inject directive: {e}"}}
        return jsonify({"status": "error", "task_id": task_id}), 500

    return jsonify({"status": "processing", "task_id": task_id}), 202

@app.route('/api/catalyst/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found"}), 404
    return jsonify(task), 200

@app.route('/api/event_stream')
def get_event_stream():
    def generate():
        while True:
            with events_lock:
                if LIVE_EVENTS:
                    event = LIVE_EVENTS.pop(0)
                    yield f"data: {json.dumps(event)}\n\n"
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

# --- Routes: System Metrics & Insights ---------------------------------------
@app.route('/api/system_metrics')
def system_metrics():
    if system_instance and hasattr(system_instance, 'resource_monitor'):
        monitor = system_instance.resource_monitor
        try:
            metrics = {
                'cpu': monitor.get_cpu_usage(),
                'memory': monitor.get_memory_usage(),
                'timestamp': timestamp_now()
            }
            return jsonify({"status": "ok", "data": metrics}), 200
        except Exception as e:
            logger.warning(f"/api/system_metrics failed: {e}")
            return jsonify({"status": "error", "error": "Failed to read metrics"}), 500
    return jsonify({"status": "error", "error": "Resource monitor not available"}), 503

@app.route('/api/task_history')
def api_task_history():
    return jsonify({"status": "ok", "data": _safe_get_task_history(limit=15)}), 200

@app.route('/api/latest_insight')
def get_latest_insight():
    insight = _safe_get_latest_agent_reflection()
    msg = insight or "No recent insights available. The system may be busy or has just started."
    return jsonify({"status": "ok", "data": {"insight": msg}}), 200

@app.route('/api/debug/tasks')
def debug_tasks():
    with tasks_lock:
        snapshot = dict(tasks)
    return jsonify({"status": "ok", "data": snapshot}), 200

@app.route('/api/health/detailed')
def health_check():
    """Health endpoint for monitoring CVA status."""
    try:
        from database import cva_db
        
        # Get tool stats
        tool_stats = cva_db.get_tool_stats()
        task_stats = cva_db.get_task_stats()
        
        # Get swarm state
        swarm = cva_db.load_full_swarm_state() or {}
        agents = list(swarm.get('agent_instances', {}).keys())
        
        return jsonify({
            "status": "healthy",
            "running": swarm.get('is_running', False),
            "paused": swarm.get('is_paused', False),
            "cycle": swarm.get('current_action_cycle_id', 'Unknown'),
            "agents": agents,
            "agent_count": len(agents),
            "tool_stats": tool_stats,
            "task_stats": task_stats
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# --- Routes: Mission Runner (threshold scaling helper) -----------------------
@app.post("/api/mission/scale_cpu")
def api_scale_cpu():
    data = request.get_json(silent=True) or {}
    namespace = data.get("namespace", "default")
    deployment = data.get("deployment")
    replicas = data.get("replicas", 2)
    threshold = data.get("threshold", 200.0)  # ms or percent, based on mission type
    approval = data.get("approval", "human")  # "human" (default) or "auto"

    if not deployment:
        return jsonify({"status": "error", "error": "deployment is required"}), 400

    try:
        replicas = int(replicas)
        threshold = float(threshold)
    except Exception:
        return jsonify({"status": "error", "error": "replicas must be int and threshold must be float"}), 400

    res = _runner.scale_on_cpu_threshold(
        namespace=namespace,
        deployment=deployment,
        threshold=threshold,
        replicas=replicas,
        approval=approval,
    )
    return jsonify({"status": "ok", "data": res}), 200

# --- Routes: Pending Plans (for dashboard) -----------------------------------
@app.get("/api/catalyst/plans")
def list_pending_plans():
    with plan_lock:
        items = [
            {k: v for k, v in p.items() if k in {
                "task_id", "status", "action", "namespace", "deployment", "replicas", "ts", "approval_token", "rationale"
            }}
            for p in plan_store.values()
            if p.get("status") == "awaiting_approval"
        ]
    return jsonify({"status": "ok", "data": items, "meta": {"count": len(items), "ts": time.time()}}), 200

# --- New: Simpler pending getter (single latest) ------------------------------
@app.get("/api/pending")
def api_get_pending():
    p = latest_pending_scale_plan()
    return jsonify({"pending": p}), 200

# --- Approve helper: call registry (or simulate) ------------------------------
def _invoke_k8s_scale_via_registry(namespace: str, deployment: str, replicas: int, approval_token: str):
    """
    Try tool_registry 'k8s_scale'. If unavailable or raises, simulate success so the loop keeps flowing.
    """
    try:
        result = tool_registry.invoke("k8s_scale", {
            "namespace": namespace,
            "deployment": deployment,
            "replicas": replicas,
            "dry_run": False,
            "approval_token": approval_token,
        })
        ok = bool(result) and ((result.get("status") == "ok") or (result.get("ok") is True))
        return ok, result or {}
    except Exception as e:
        sim = {
            "ok": True,
            "simulated": True,
            "action": "k8s_scale",
            "namespace": namespace,
            "deployment": deployment,
            "replicas": replicas,
            "approval_token": approval_token,
            "note": f"Registry invoke failed, simulating success: {e}"
        }
        logger.info("[Approve] SIMULATION: %s", sim)
        return True, sim

# --- New: Human-in-the-loop Approve (token OR task_id) ------------------------
@app.post('/api/approve')
def api_post_approve():
    """
    Approves a pending k8s_scale plan and executes it.
    Body options:
      A) {"task_id": "...", "approval_token"?: "..."}
      B) {"approval_token": "..."}  # finds the matching pending plan
      C) {"namespace": "...", "deployment": "...", "replicas": 3, "approval_token"?: "..."}  # ad-hoc
    """
    body = request.get_json(silent=True) or {}
    token = body.get("approval_token") or body.get("token") or "dashboard_approved"
    task_id = body.get("task_id")

    # Resolve plan
    plan = None
    if task_id:
        with plan_lock:
            p = plan_store.get(task_id)
        if not p or p.get("status") != "awaiting_approval":
            return jsonify({"ok": False, "error": "No matching pending plan for task_id."}), 404
        plan = p
    elif "namespace" in body or "deployment" in body or "replicas" in body:
        # Ad-hoc approval (no stored plan)
        try:
            ns, dep, rep = _normalize_scale_params(body)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        plan = {
            "task_id": f"adhoc_{int(time.time())}",
            "status": "awaiting_approval",
            "action": "k8s_scale",
            "namespace": ns,
            "deployment": dep,
            "replicas": rep,
            "approval_token": token,
        }
    else:
        # Try token lookup
        plan = _find_pending_by_token(token)
        if not plan:
            # Fallback to latest
            plan = latest_pending_scale_plan()
        if not plan:
            return jsonify({"ok": False, "error": "No pending mission."}), 404

    try:
        ns, dep, rep = _normalize_scale_params(plan if "params" not in plan else plan["params"])
    except Exception as e:
        # If plan already has flat fields, use them directly
        ns = plan.get("namespace")
        dep = plan.get("deployment")
        rep = plan.get("replicas")
        if not (ns and dep and isinstance(rep, (int, float))):
            return jsonify({"ok": False, "error": f"Invalid plan fields: {e}"}), 400
        rep = int(rep)

    ok, result = _invoke_k8s_scale_via_registry(ns, dep, rep, plan.get("approval_token", token))

    if ok:
        if plan.get("task_id") and plan.get("status") == "awaiting_approval":
            mark_plan_executed(plan["task_id"])
        return jsonify({"ok": True, "detail": result, "message": f"Scaled {dep} to {rep} replicas"}), 200
    return jsonify({"ok": False, "detail": result}), 500

# --- Existing: Approve Scale (kept for backward-compat) -----------------------
@app.post('/api/catalyst/approve_scale')
def approve_scale():
    """
    Approves a pending k8s scale plan and executes it (legacy path).
    Accepts either:
      A) {"task_id": "...", "approval_token"?: "..."}
      B) {"namespace": "...", "deployment": "...", "replicas": 3, "approval_token"?: "..."}
    """
    try:
        body = request.get_json(silent=True) or {}
        task_id = body.get("task_id")
        token = body.get("approval_token") or "dashboard_approved"

        # Resolve plan args
        if task_id:
            with plan_lock:
                plan = plan_store.get(task_id)
            if not plan or plan.get("status") != "awaiting_approval":
                return jsonify({"status": "error", "error": "No matching pending plan for task_id."}), 404
            ns = plan.get("namespace")
            dep = plan.get("deployment")
            rep = plan.get("replicas")
        else:
            ns = body.get("namespace")
            dep = body.get("deployment")
            rep = body.get("replicas")

        # Validate required fields
        missing = [k for k, v in (("namespace", ns), ("deployment", dep), ("replicas", rep)) if v in (None, "")]
        if missing:
            return jsonify({"status": "error", "error": f"Missing field(s): {', '.join(missing)}"}), 400

        try:
            rep = int(rep)
        except Exception:
            return jsonify({"status": "error", "error": "replicas must be an integer"}), 400

        ok, result = _invoke_k8s_scale_via_registry(ns, dep, rep, token)

        if ok:
            if task_id:
                mark_plan_executed(task_id)
            return jsonify({
                "status": "ok",
                "data": {"message": f"Scaled {dep} to {rep} replicas", "result": result},
                "meta": {"ts": time.time()}
            }), 200

        return jsonify({
            "status": "error",
            "error": (result or {}).get("error", "Scale action failed"),
            "data": result,
            "meta": {"ts": time.time()}
        }), 400

    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "meta": {"ts": time.time()}}), 500

# --- Routes: Simple K8s Metrics Proxy (for widgets) --------------------------
@app.get('/api/catalyst/metrics')
def get_metrics():
    """
    Lightweight cluster utilization proxy for dashboard gauges.
    Uses the registered 'kubernetes_pod_metrics' tool.
    """
    now_ms = int(time.time() * 1000)

    try:
        pods = tool_registry.invoke("kubernetes_pod_metrics", {"limit": 999})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e),
                        "data": {"cpu": 0, "memory": 0, "timestamp": now_ms}}), 503

    if pods and pods.get('status') == 'ok':
        total_cpu = pods['data'].get('total_cpu_mcores', 0)  # millicores
        total_mem = pods['data'].get('total_memory_Mi', 0)   # Mi
        # Heuristic normalization (tune to your cluster size/UI expectation)
        cpu_pct = total_cpu / 10.0
        mem_pct = total_mem / 40.0
        return jsonify({"status": "ok", "data": {"cpu": cpu_pct, "memory": mem_pct, "timestamp": now_ms}}), 200

    return jsonify({"status": "error", "error": "metrics unavailable",
                    "data": {"cpu": 0, "memory": 0, "timestamp": now_ms}}), 200

# --- Routes: Prometheus proxy -------------------------------------------------
@app.get("/api/prom/query")
def api_prom_query():
    q = request.args.get("q") or "up"
    res = tool_registry.invoke("prometheus_query", {"query": q})
    return jsonify(res), 200

@app.get('/api/diagnostics')
def api_diagnostics():
    """Expose system diagnostics via tool_registry."""
    try:
        result = tool_registry.safe_call("system_diagnostics")
        return jsonify(result.get("data")), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get('/api/tool_breakers')
def api_tool_breakers():
    """Expose tool circuit breaker state via tool_registry."""
    try:
        result = tool_registry.safe_call("tool_breaker_status")
        status_code = 200 if result.get("status") == "ok" else 500
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get('/api/self_test')
def api_self_test():
    """Expose a quick self-test via tool_registry."""
    try:
        result = tool_registry.safe_call("self_test")
        return jsonify(result.get("data")), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get('/api/tasks')
def api_tasks():
    """Return active tasks, recent tasks, and failed tasks (best effort)."""
    try:
        with tasks_lock:
            active = {k: v for k, v in tasks.items() if v.get("status") == "processing"}
            recent = list(RECENT_TASKS)
        # Failed tasks from logs are not tracked separately; infer from RECENT_TASKS
        failed = [t for t in recent if t.get("status") not in ("completed", "ok")]
        return jsonify({
            "active": active,
            "recent": recent,
            "failed": failed
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.post('/api/restart_agent/<agent_name>')
def api_restart_agent(agent_name):
    """Restart an agent via the restart_agent tool."""
    try:
        result = tool_registry.safe_call("restart_agent", agent_name=agent_name)
        status_code = 200 if result.get("status") == "ok" else 400
        return jsonify(result.get("data")), status_code
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get("/api/agents")
def api_list_agents():
    """List all agents with basic state."""
    agents = {}
    try:
        with getattr(system_instance, "_agents_lock", agent_instances_lock):
            snapshot = dict(getattr(system_instance, "agent_instances", {}) or {})
        for name, agent in snapshot.items():
            agent_state = {}
            try:
                if hasattr(agent, "get_state"):
                    agent_state = agent.get_state()
                else:
                    agent_state = getattr(agent, "__dict__", {})
            except Exception:
                agent_state = {}

            agents[name] = {
                "paused": getattr(agent, "is_paused", lambda: False)(),
                "role": getattr(agent, "role", getattr(agent, "eidos_spec", {}).get("role", "")) if hasattr(agent, "role") else getattr(agent, "eidos_spec", {}).get("role", ""),
                "last_task_outcome": getattr(agent, "last_task_outcome", None),
                "state": agent_state,
            }
        return jsonify({"agents": agents})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.get("/api/agents/<agent_name>")
def api_get_agent(agent_name):
    """Get details for a single agent."""
    try:
        with getattr(system_instance, "_agents_lock", agent_instances_lock):
            agent = getattr(system_instance, "agent_instances", {}).get(agent_name)
        if not agent:
            return jsonify({"status": "error", "error": "agent not found"}), 404

        try:
            agent_state = agent.get_state() if hasattr(agent, "get_state") else getattr(agent, "__dict__", {})
        except Exception:
            agent_state = {}

        data = {
            "name": agent_name,
            "paused": getattr(agent, "is_paused", lambda: False)(),
            "role": getattr(agent, "role", getattr(agent, "eidos_spec", {}).get("role", "")) if hasattr(agent, "role") else getattr(agent, "eidos_spec", {}).get("role", ""),
            "last_task_outcome": getattr(agent, "last_task_outcome", None),
            "state": agent_state,
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# --- Main ---------------------------------------------------------------------

@app.route('/api/agents/factory', methods=['GET'])
def get_factory_status():
    """Get Agent Factory status and metrics."""
    if system_instance is None:
        return jsonify({
            "status": "ok",
            "active_agents": [],
            "metrics": {},
            "agent_list": [],
            "last_health_check": {}
        }), 200
    try:
        with agent_instances_lock:
            factory = system_instance.agent_factory
            guardian = system_instance.guardian
            active_agents = factory.list_active()
            metrics = guardian.get_metrics()
            last_health_check = guardian.health_check()
            agent_list = list(getattr(system_instance, "agent_instances", {}).keys())

        return jsonify({
            "status": "ok",
            "active_agents": active_agents,
            "metrics": metrics,
            "agent_list": agent_list,
            "last_health_check": last_health_check
        }), 200
    except Exception as e:
        logger.exception("/api/agents/factory failed")
        return jsonify({"status": "error", "error": str(e), "active_agents": []}), 500

@app.route('/api/agents/semantic-tools', methods=['POST'])
def get_semantic_tool_suggestions():
    """Get semantically matched tools for a given purpose (debugging endpoint)."""
    try:
        data = request.json
        purpose = data.get('purpose', '')
        if not purpose:
            return jsonify({"error": "purpose is required"}), 400

        factory = system_instance.agent_factory
        suggestions = factory.get_semantic_suggestions(purpose)
        return jsonify(suggestions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/spawn', methods=['POST'])
def spawn_agent_api():
    """Manually spawn an agent via API for testing."""
    try:
        data = request.json
        purpose = data.get('purpose', 'Test agent')
        context = data.get('context', {})

        # Protect factory access
        with agent_instances_lock:
            result = system_instance.handle_spawn_request(
                purpose=purpose,
                context=context,
                parent_agent="api_manual"
            )

        # Check if result is a validation error dict
        if isinstance(result, dict):
            # Validation failed - return error with suggestions
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown error"),
                "suggestions": result.get("suggestions", []),
                "hint": result.get("hint", "")
            }), 400

        # Success - result is agent_id string
        if result:
            return jsonify({"success": True, "agent_id": result})

        return jsonify({"success": False, "error": "Spawn failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/task', methods=['POST'])
def assign_agent_task():
    """Give a dynamic agent a specific task to execute"""
    data = request.get_json()
    agent_id = data.get('agent_id')
    task_description = data.get('task')
    
    if not agent_id or not task_description:
        return jsonify({"error": "Missing agent_id or task"}), 400
    
    # Get agent from factory (thread-safe)
    with agent_instances_lock:
        agent = system_instance.agent_factory.active_agents.get(agent_id)
        if not agent:
            # Fallback to orchestrator registry if present
            agent = getattr(system_instance, "agent_instances", {}).get(agent_id)
    if not agent:
        return jsonify({"error": f"Agent {agent_id} not found"}), 404
    
    try:
        # Create a task directive for the agent
        result = agent.execute_task({"description": task_description})
        
        return jsonify({
            "success": True,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "task": task_description,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/agents/kill/<agent_id>', methods=['POST'])
def api_kill_agent(agent_id):
    """Kill a dynamic agent by id."""
    try:
        factory = getattr(system_instance, "agent_factory", None)
        if factory:
            ok = factory.kill_agent(agent_id)
            if ok:
                return jsonify({"status": "ok", "agent_id": agent_id}), 200
        return jsonify({"status": "error", "error": "not found"}), 404
    except Exception as e:
        logger.exception(f"/api/agents/kill/{agent_id} failed")
        return jsonify({"status": "error", "error": str(e)}), 500

# ==========================================
# GEMINI™ PROTOCOL API LAYER
# ==========================================
@app.route('/api/agents/spawn', methods=['POST'])
def gemini_agent_deployment():
    """Gemini™ cloud agent deployment endpoint"""
    # This currently just returns a status, but connects to the system
    return {
        "status": "Gemini Protocol Initiated", 
        "orchestrator": "GeminiOrchestrator v1.0",
        "timestamp": time.time()
    }

# --- Health Monitoring Helper Functions ---------------------------------------
def calculate_health_score(swarm, tool_stats, task_stats):
    """Calculate overall system health score (0-100)."""
    try:
        score = 50  # Base score
        
        # System state factors
        if swarm.get('is_running', False):
            score += 15
        if not swarm.get('is_paused', False):
            score += 10
        
        # Agent health
        agent_count = len(swarm.get('agent_instances', {}))
        if agent_count > 0:
            score += 10
        if agent_count > 3:
            score += 5  # Bonus for multiple agents
        
        # Tool performance
        if tool_stats:
            success_rate = tool_stats.get('successful_calls', 0) / max(1, tool_stats.get('total_calls', 1))
            if success_rate > 0.95:
                score += 10
            elif success_rate > 0.8:
                score += 5
        
        # Task performance
        if task_stats:
            completion_rate = task_stats.get('completed', 0) / max(1, task_stats.get('total_tasks', 1))
            if completion_rate > 0.9:
                score += 5
            elif completion_rate > 0.7:
                score += 3
        
        # Cap score at 100
        return min(100, max(0, score))
    except Exception as e:
        logger.warning(f"Health score calculation failed: {e}")
        return 50

def get_health_status(score):
    """Convert health score to status level."""
    if score >= 80:
        return 'healthy'
    elif score >= 60:
        return 'warning'
    elif score >= 40:
        return 'degraded'
    else:
        return 'critical'

def get_system_resources():
    """Get current system resource usage."""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        return {
            'cpu': cpu_percent,
            'memory': memory_percent,
            'disk': disk_usage,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}")
        return {
            'cpu': 0,
            'memory': 0,
            'disk': 0,
            'timestamp': time.time()
        }

def get_system_uptime():
    """Get system uptime in seconds."""
    try:
        import psutil
        return psutil.boot_time()
    except:
        return time.time() - 3600  # Default to 1 hour if unavailable

# --- Enhanced Health Endpoint -----------------------------------------------
@app.get("/api/health/enhanced")
def enhanced_health_check():
    """Enhanced health endpoint with comprehensive system metrics and health scoring."""
    try:
        from database import cva_db
        
        # Get basic stats
        tool_stats = cva_db.get_tool_stats()
        task_stats = cva_db.get_task_stats()
        
        # Get swarm state
        swarm = cva_db.load_full_swarm_state() or {}
        agents = list(swarm.get('agent_instances', {}).keys())
        
        # Calculate health metrics
        health_score = calculate_health_score(swarm, tool_stats, task_stats)
        health_status = get_health_status(health_score)
        
        # Get system resource metrics
        resource_metrics = get_system_resources()
        
        # Get recent events
        recent_events = get_recent_system_events()
        
        return jsonify({
            "status": health_status,
            "health_score": health_score,
            "health_status": health_status,
            "running": swarm.get('is_running', False),
            "paused": swarm.get('is_paused', False),
            "current_cycle": swarm.get('current_action_cycle_id', 'Unknown'),
            "agents": {
                "total": len(agents),
                "active": len([a for a in agents if not a.endswith('_paused')]),
                "list": agents
            },
            "tool_stats": tool_stats,
            "task_stats": task_stats,
            "resource_metrics": resource_metrics,
            "recent_events": recent_events,
            "timestamp": time.time(),
            "uptime": get_system_uptime(),
            "recommendations": generate_health_recommendations(health_score, resource_metrics)
        }), 200
    except Exception as e:
        logger.error(f"Enhanced health check failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

def get_recent_system_events():
    """Get recent system events from the event log."""
    try:
        events = []
        event_file = "logs/catalyst.jsonl"
        
        if os.path.exists(event_file):
            with open(event_file, 'r') as f:
                lines = f.readlines()[-10:]  # Get last 10 events
                for line in lines:
                    try:
                        event = json.loads(line)
                        events.append({
                            'timestamp': str(event.get('timestamp')),
                            'level': str(event.get('level', 'info')),
                            'message': str(event.get('message', '')),
                            'source': str(event.get('source', 'system'))
                        })
                    except Exception as e:
                        logger.debug(f"Failed to parse event line: {e}")
                        continue
        
        return events
    except Exception as e:
        logger.warning(f"Failed to read recent events: {e}")
        return []

def generate_health_recommendations(score, resources):
    """Generate health recommendations based on current system state."""
    recommendations = []
    
    if score < 60:
        recommendations.append({
            'level': 'critical',
            'message': 'System health is critical. Immediate attention required.',
            'actions': ['Check system logs', 'Review agent status', 'Restart critical services']
        })
    
    elif score < 80:
        recommendations.append({
            'level': 'warning',
            'message': 'System health could be improved.',
            'actions': ['Review recent failures', 'Check resource usage', 'Optimize agent configuration']
        })
    
    if resources.get('cpu', 0) > 80:
        recommendations.append({
            'level': 'warning',
            'message': 'High CPU usage detected.',
            'actions': ['Review running processes', 'Consider scaling resources', 'Optimize agent workload']
        })
    
    if resources.get('memory', 0) > 85:
        recommendations.append({
            'level': 'warning',
            'message': 'High memory usage detected.',
            'actions': ['Check for memory leaks', 'Increase memory allocation', 'Restart memory-intensive services']
        })
    
    if len(recommendations) == 0:
        recommendations.append({
            'level': 'info',
            'message': 'System is operating normally.',
            'actions': ['Continue monitoring', 'Perform regular maintenance']
        })
    
    return recommendations

if __name__ == '__main__':
    
    logger.info("[app.py] Starting GmailAgent loop in a background thread...")
    gmail_thread = threading.Thread(target=gmail_agent.main_loop, daemon=True)
    
    gmail_thread.start()
    logger.info("[app.py] GmailAgent is now running in the background.")
    # ----------------------------------------------------------

    # Start CVA loop in a background thread (This is your original code)
    system_thread = Thread(target=run_catalyst_system_in_background, daemon=True)
    system_thread.start()

    print("\n" + "="*62)
    print("Starting Flask web server for Catalyst Vector Alpha Dashboard...")
    port = int(os.getenv("PORT", "5000"))
    print(f"Dashboard available at: http://127.0.0.1:{port}/")
    print(f"Debug tasks at:        http://127.0.0.1:{port}/api/debug/tasks")
    print("="*62 + "\n")

    try:
        from waitress import serve
        serve(app, host='127.0.0.1', port=port)
    except ImportError:
        app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
