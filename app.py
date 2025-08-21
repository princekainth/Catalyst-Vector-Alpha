# ==================================================================
#  app.py - Main Application Runner for Catalyst Vector Alpha
# ==================================================================

# --- Standard Library Imports ---
import logging
import sys
import os
import json
import time
from threading import Thread
from collections import deque
from datetime import datetime, timezone
from functools import wraps

# --- Third-Party Library Imports ---
from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS
import psutil

# --- Add Project Root to Python Path ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

# --- Project-Specific Imports ---
try:
    from catalyst_vector_alpha import CatalystVectorAlpha
    from core import MessageBus, EventMonitor, ToolRegistry
    from ccn_monitor_mock import MockCCNMonitor
except ImportError as e:
    print(f"FATAL IMPORT ERROR: {e}")
    print("Please make sure catalyst_vector_alpha.py, core.py, and other required files exist.")
    sys.exit(1)

# --- Global Variables & Logging ---
system_instance = None
system_thread = None
web_log_buffer = deque(maxlen=2000)
APP_START_TS = time.time()
logger = logging.getLogger("CatalystLogger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

# Console handler for stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for persistent logs
file_handler = logging.FileHandler(os.path.join(APP_ROOT, 'persistence_data', 'logs', 'app.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class WebStreamHandler(logging.Handler):
    """Custom logging handler to store log entries in a deque for web streaming."""
    def __init__(self, deque_instance):
        super().__init__()
        self.deque = deque_instance

    def emit(self, record):
        try:
            log_entry = json.loads(record.getMessage())
            self.deque.append(log_entry)
        except (json.JSONDecodeError, TypeError):
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "level": record.levelname,
                "source": record.name,
                "description": record.getMessage(),
                "details": {}
            }
            self.deque.append(log_entry)
        except Exception as e:
            print(f"WebStreamHandler failed: {e}", file=sys.stderr)

web_stream_handler = WebStreamHandler(web_log_buffer)
web_stream_handler.setLevel(logging.DEBUG)
logger.addHandler(web_stream_handler)

# --- Define Global Constants ---
PERSISTENCE_DIR = 'persistence_data'
LOGS_SUBDIR = 'logs'
ISL_SCHEMA_PATH = 'isl_schema.yaml'
SYSTEM_PAUSE_FILE_BASENAME = 'system_pause.flag'
SWARM_STATE_FILE_BASENAME = 'swarm_state.json'
PAUSED_AGENTS_FILE_BASENAME = 'agent_pause_flags.json'
CHROMA_DB_SUBDIR = 'chroma_db'
SWARM_ACTIVITY_LOG_REL_PATH = os.path.join(LOGS_SUBDIR, 'swarm_activity.jsonl')
INTENT_OVERRIDE_PREFIX = 'intent_override_'

# --- Create Necessary Directories ---
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
os.makedirs(os.path.join(PERSISTENCE_DIR, LOGS_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(PERSISTENCE_DIR, CHROMA_DB_SUBDIR), exist_ok=True)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- API Key Authentication ---
API_KEY = os.getenv("CATALYST_API_KEY", "your-secret-key")  # Set in environment variable

def require_api_key(f):
    """Decorator to require API key authentication for sensitive endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_KEY:
            return jsonify({"status": "error", "message": "Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated

# --- Background System Thread ---
def run_catalyst_system_in_background():
    """Run the Catalyst Vector Alpha system in a background thread."""
    global system_instance
    message_bus_instance = MessageBus()
    tool_registry_instance = ToolRegistry()
    event_monitor_instance = EventMonitor()
    mock_ccn_monitor_instance = MockCCNMonitor()
    
    system_instance = CatalystVectorAlpha(
        message_bus=message_bus_instance,
        tool_registry=tool_registry_instance,
        event_monitor=event_monitor_instance,
        external_log_sink=logger,
        persistence_dir=PERSISTENCE_DIR,
        swarm_activity_log=SWARM_ACTIVITY_LOG_REL_PATH,
        system_pause_file=SYSTEM_PAUSE_FILE_BASENAME,
        swarm_state_file=SWARM_STATE_FILE_BASENAME,
        paused_agents_file=PAUSED_AGENTS_FILE_BASENAME,
        isl_schema_path=ISL_SCHEMA_PATH,
        chroma_db_path=CHROMA_DB_SUBDIR,
        intent_override_prefix=INTENT_OVERRIDE_PREFIX,
        ccn_monitor_interface=mock_ccn_monitor_instance
    )
    logger.info("Catalyst Vector Alpha system is starting in background thread...")
    system_instance.run_cognitive_loop()
    logger.info("Catalyst Vector Alpha system thread has finished.")

# --- Flask API Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_system_status():
    """Get the current system status, including uptime, agent count, and resource usage."""
    uptime = int(time.time() - APP_START_TS)
    agents_online = 0
    system_paused = False

    if system_instance:
        try:
            agents_online = len(getattr(system_instance, 'agent_instances', {}) or {})
            if hasattr(system_instance, 'is_system_paused'):
                system_paused = bool(system_instance.is_system_paused())
        except AttributeError:
            system_paused = False

        return jsonify({
            "system_paused": system_paused,
            "uptime": uptime,
            "agents_online": agents_online,
            "is_running": getattr(system_instance, 'is_running', False),
            "current_cycle": getattr(getattr(system_instance, 'message_bus', None), 'current_cycle_id', 'N/A'),
            "active_agents": list((getattr(system_instance, 'agent_instances', {}) or {}).keys()),
            "intervention_needed": bool(getattr(system_instance, 'get_pending_human_intervention_requests', lambda: [])()),
            "resource_usage": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent
            }
        })

    return jsonify({
        "system_paused": True,
        "uptime": uptime,
        "agents_online": 0,
        "is_running": False,
        "message": "System not initialized."
    })

@app.route('/api/logs')
def get_logs():
    """Retrieve recent system logs with a configurable limit."""
    try:
        limit = int(request.args.get('limit', 50))
    except ValueError:
        limit = 50

    items = list(web_log_buffer)[-limit:]
    normalized = []
    for entry in items:
        ts = entry.get("timestamp") or datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lvl = entry.get("level") or entry.get("severity") or "INFO"
        msg = entry.get("message") or entry.get("description") or json.dumps(entry)
        normalized.append({"timestamp": ts, "level": lvl, "message": msg})

    return jsonify(normalized)

@app.route('/api/agents')
def get_agents_status():
    """Get status summaries for all agents."""
    agents_data = []
    if system_instance and hasattr(system_instance, 'agent_instances'):
        for name, agent in (system_instance.agent_instances or {}).items():
            summary = {}
            if hasattr(agent, 'get_status_summary'):
                try:
                    summary = agent.get_status_summary() or {}
                except AttributeError:
                    summary = {}

            normalized = {
                "name": summary.get("name") or getattr(agent, "name", name),
                "role": summary.get("role") or getattr(agent, "role", getattr(agent, "agent_type", "")),
                "status": summary.get("status") or summary.get("state") or "unknown",
            }
            normalized.update(summary)
            agents_data.append(normalized)
    return jsonify(agents_data)

@app.route('/api/agent/<agent_name>')
def get_agent_detail(agent_name):
    """Get detailed state for a specific agent."""
    if not (system_instance and hasattr(system_instance, 'agent_instances')):
        return jsonify({"error": "Agent not found"}), 404

    agent = system_instance.agent_instances.get(agent_name)
    if not agent:
        return jsonify({"error": "Agent not found"}), 404

    detail = {}
    if hasattr(agent, 'get_detailed_state'):
        try:
            detail = agent.get_detailed_state() or {}
        except AttributeError:
            detail = {}

    normalized = {
        "name": detail.get("name") or getattr(agent, "name", agent_name),
        "role": detail.get("role") or getattr(agent, "role", getattr(agent, "agent_type", "")),
        "status": detail.get("status") or detail.get("state") or "unknown",
        "intent": detail.get("intent") or detail.get("current_intent") or "—",
        "metrics": detail.get("metrics") or detail.get("stats") or {},
        "capabilities": detail.get("capabilities") or detail.get("tools") or [],
        "performance_metrics": detail.get("performance_metrics")
            or detail.get("performance")
            or detail.get("perf_metrics")
            or {},
        "recent_memories": detail.get("recent_memories")
            or detail.get("memories")
            or detail.get("memory_log")
            or [],
    }
    normalized.update(detail)
    return jsonify(normalized)

@app.route('/api/pending_interventions')
def get_pending_interventions():
    """Get a list of pending human intervention requests."""
    if system_instance:
        return jsonify(system_instance.get_pending_human_intervention_requests())
    return jsonify([])

@app.route('/api/inject_directive', methods=['POST'])
@require_api_key
def inject_directive():
    """Inject a directive or human response into the system."""
    if not system_instance:
        return jsonify({"status": "error", "message": "System not initialized."}), 500
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided."}), 400
        if data.get('type') == "HUMAN_RESPONSE":
            system_instance.handle_human_response(data['payload']['request_id'], data['payload'])
            return jsonify({"status": "success", "message": "Human response processed."})
        else:
            system_instance.inject_directives([data])
            return jsonify({"status": "success", "message": f"Directive '{data.get('type')}' injected."})
    except (KeyError, ValueError) as e:
        logger.error(f"Error injecting directive: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except AttributeError as e:
        logger.error(f"Error injecting directive: {e}")
        return jsonify({"status": "error", "message": "Invalid system state."}), 500

@app.route('/api/live_logs')
def live_logs():
    """Stream live logs using Server-Sent Events."""
    def generate():
        sent_log_count = 0
        while True:
            if sent_log_count < len(web_log_buffer):
                for i in range(sent_log_count, len(web_log_buffer)):
                    yield f"data: {json.dumps(web_log_buffer[i])}\n\n"
                sent_log_count = len(web_log_buffer)
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/system/pause', methods=['POST'])
@require_api_key
def pause_system_api():
    """Pause the system via the dashboard."""
    try:
        if system_instance and hasattr(system_instance, 'pause_system'):
            system_instance.pause_system("Paused via dashboard button.")
            return jsonify({"status": "success", "message": "System pause command issued."})
        return jsonify({"status": "error", "message": "System not running or pause function unavailable."}), 500
    except AttributeError as e:
        logger.error(f"Error pausing system: {e}")
        return jsonify({"status": "error", "message": "Invalid system state."}), 500

@app.route('/api/system/unpause', methods=['POST'])
@require_api_key
def unpause_system_api():
    """Unpause the system via the dashboard."""
    try:
        if system_instance and hasattr(system_instance, 'unpause_system'):
            system_instance.unpause_system("Unpaused via dashboard button.")
            return jsonify({"status": "success", "message": "System unpause command issued."})
        return jsonify({"status": "error", "message": "System not running or unpause function unavailable."}), 500
    except AttributeError as e:
        logger.error(f"Error unpausing system: {e}")
        return jsonify({"status": "error", "message": "Invalid system state."}), 500

@app.route('/api/agent/<agent_name>/intent', methods=['POST'])
@require_api_key
def override_agent_intent(agent_name):
    """Override the intent of a specific agent."""
    if not system_instance or agent_name not in system_instance.agent_instances:
        return jsonify({"status": "error", "message": "Agent not found."}), 404
    
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided."}), 400
        new_intent = data.get('intent')
        if not new_intent:
            return jsonify({"status": "error", "message": "New intent not provided."}), 400
        
        agent = system_instance.agent_instances[agent_name]
        agent.update_intent(new_intent)
        return jsonify({"status": "success", "message": f"Intent for {agent_name} has been updated."})
    except (KeyError, ValueError) as e:
        logger.error(f"Error overriding intent for {agent_name}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except AttributeError as e:
        logger.error(f"Error overriding intent for {agent_name}: {e}")
        return jsonify({"status": "error", "message": "Invalid agent state."}), 500

# --- Main Execution Block ---
if __name__ == '__main__':
    system_thread = Thread(target=run_catalyst_system_in_background)
    system_thread.daemon = True
    system_thread.start()

    print("\n" + "="*50)
    print("Starting Flask web server for Catalyst Vector Alpha Dashboard...")
    print("Dashboard available at: http://127.0.0.1:5000/")
    print("="*50 + "\n")

    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)