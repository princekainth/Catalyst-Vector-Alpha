# tools.py — robust, production-ready tool implementations
from __future__ import annotations

import os
import re
import time
import json
import math
import psutil
import random
import shutil
import logging
import hashlib
import subprocess
import shlex
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from ipaddress import ip_address, IPv4Address
import threading
from concurrent.futures import ThreadPoolExecutor
from shared_memory import SharedMemory
# Initialize the brain so tools can use it
SHARED_BRAIN = SharedMemory()
# Import sandbox tools so registry can find them
from sandbox_tools import execute_terminal_command as execute_terminal_command_tool
from sandbox_tools import write_sandbox_file as write_sandbox_file_tool
import functools 
from functools import lru_cache
from time import sleep
# Third-party imports with graceful fallbacks
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BEAUTIFULSOUP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF = None
    FPDF_AVAILABLE = False

try:
    from transformers import pipeline as _hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    _hf_pipeline = None
    TRANSFORMERS_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    from google_auth import authenticate_google_services
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    build = None
    authenticate_google_services = None
    GOOGLE_APIS_AVAILABLE = False

# Local imports
try:
    from sandbox_tools import execute_terminal_command, write_sandbox_file
    SANDBOX_AVAILABLE = True
except ImportError:
    execute_terminal_command = None
    write_sandbox_file = None
    SANDBOX_AVAILABLE = False

# Toolsmith generation (sandboxed)
try:
    from sandbox_toolsmith import generate_tool as toolsmith_generate
except Exception:
    toolsmith_generate = None

class ToolCache:
    """Simple TTL cache for expensive tool results."""
    def __init__(self):
        self._cache = {}
        self._ttl = {}
    
    def get(self, key: str, max_age: int = 300):
        """Get cached value if not expired (default 5 min)."""
        if key in self._cache:
            if time.time() - self._ttl[key] < max_age:
                return self._cache[key]
        return None
    
    def set(self, key: str, value):
        """Cache a value."""
        self._cache[key] = value
        self._ttl[key] = time.time()
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._ttl.clear()

_tool_cache = ToolCache()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry failed tool executions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            current_delay = delay
            cancel_event = kwargs.get("cancel_event")
            
            for attempt in range(max_retries):
                if cancel_event and cancel_event.is_set():
                    return {"ok": False, "error": "cancelled"}
                try:
                    result = func(*args, **kwargs)
                    # Check if result indicates failure
                    if isinstance(result, dict) and result.get('ok') == False:
                        raise Exception(result.get('error', 'Unknown error'))
                    return result
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        sleep(current_delay)
                        current_delay *= backoff
            
            # All retries failed
            logger.error(f"All {max_retries} retries failed for {func.__name__}: {last_error}")
            return {"ok": False, "error": f"Failed after {max_retries} retries: {last_error}"}
        return wrapper
    return decorator

from config_manager import get_config

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
_config = get_config()
_tool_cfg = _config.get("tool_timeouts") or {}


class ToolConfig:
    """Central configuration for all tools (overridable via config/*.yaml)"""
    KUBECTL_TIMEOUT = int(_tool_cfg.get("kubectl_timeout", 15))
    REQUEST_TIMEOUT = int(_tool_cfg.get("request_timeout", 12))
    SCALE_MIN_INTERVAL = float(_tool_cfg.get("scale_min_interval", float(os.getenv("CVA_SCALE_MIN_INTERVAL_S", "300"))))
    MAX_PROCESS_LIMIT = int(_tool_cfg.get("max_process_limit", 100))
    MAX_FILE_SIZE_MB = int(_tool_cfg.get("max_file_size_mb", 50))
    MAX_WEBPAGE_SIZE = int(_tool_cfg.get("max_webpage_size", 8000))
    MAX_SEARCH_RESULTS = int(_tool_cfg.get("max_search_results", 5))
    
    # Security limits
    MAX_SCAN_TARGETS = int(_tool_cfg.get("max_scan_targets", 10))
    MAX_HASH_TEXT_SIZE = int(_tool_cfg.get("max_hash_text_size", 10 * 1024 * 1024))  # 10MB default
    
    @classmethod
    def validate(cls):
        """Validate configuration on startup"""
        if cls.SCALE_MIN_INTERVAL < 0:
            raise ValueError("SCALE_MIN_INTERVAL must be positive")
        if cls.MAX_PROCESS_LIMIT <= 0:
            raise ValueError("MAX_PROCESS_LIMIT must be positive")
        return True

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("CatalystLogger")

# ------------------------------------------------------------------------------
# Utility Classes & Helpers
# ------------------------------------------------------------------------------
class ToolUsageTracker:
    """Thread-safe tool usage tracking"""
    def __init__(self):
        self._lock = threading.RLock()
        self._stats: Dict[str, int] = {}
        self._errors: Dict[str, int] = {}
    
    def track_usage(self, tool_name: str, success: bool = True, execution_time: float = 0.0, error: str = None):
        with self._lock:
            self._stats[tool_name] = self._stats.get(tool_name, 0) + 1
            if not success:
                self._errors[tool_name] = self._errors.get(tool_name, 0) + 1
        # Log to database
        try:
            from database import cva_db
            cva_db.record_tool_usage(tool_name, success, execution_time, error)
        except Exception:
            pass  # Don't break tools if DB fails

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_calls = sum(self._stats.values())
            error_count = sum(self._errors.values())
            error_rate = error_count / max(1, total_calls)
            return {
                "usage": dict(self._stats),
                "errors": dict(self._errors),
                "total_calls": total_calls,
                "error_rate": round(error_rate, 4)
            }

# Global tracker instance
_usage_tracker = ToolUsageTracker()

class SafeSubprocess:
    """Safe subprocess execution with timeouts and error handling"""
    
    @staticmethod
    def run(cmd: List[str], timeout: float = 10, **kwargs) -> Tuple[bool, str, str]:
        """Execute command safely with timeout"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # We'll handle non-zero returns
                **kwargs
            )
            return True, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except FileNotFoundError:
            return False, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, "", f"Subprocess error: {str(e)}"
    
    @staticmethod
    def check_available(command: str) -> bool:
        """Check if a command is available in PATH"""
        return shutil.which(command) is not None

class RetrySession:
    """HTTP session with retry logic"""
    
    @staticmethod
    def create_session(retries: int = 3, backoff_factor: float = 0.5):
        """Create requests session with retry logic"""
        if not REQUESTS_AVAILABLE:
            return None
            
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

# ------------------------------------------------------------------------------
# Core Utilities
# ------------------------------------------------------------------------------
_PLACEHOLDERS = {"", " ", "string", "placeholder", "tbd", "todo", "none", "null", "n/a", "na", "<placeholder>"}

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS

def _require_non_placeholder(name: str, value: Any) -> Optional[str]:
    return None if not _is_placeholder(value) else f"'{name}' is missing or placeholder."

def _valid_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in {"http", "https"} and bool(u.netloc)
    except Exception:
        return False

def _validate_integer(value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Optional[str]:
    """Validate integer with optional min/max bounds"""
    try:
        int_val = int(value)
        if min_val is not None and int_val < min_val:
            return f"Value must be >= {min_val}"
        if max_val is not None and int_val > max_val:
            return f"Value must be <= {max_val}"
        return None
    except (ValueError, TypeError):
        return "Value must be an integer"

def standardize_response(status: str, data: Any = None, error: str | None = None, **meta) -> dict:
    """Consistent response wrapper across all tools"""
    res = {
        "status": status, 
        "timestamp": _now_iso(),
        "success": status.lower() in ("ok", "success")
    }
    if data is not None:
        res["data"] = data
    if error:
        res["error"] = error
    if meta:
        res.update(meta)
    return res

# ------------------------------------------------------------------------------
# Validators
# ------------------------------------------------------------------------------
def _v_url(field: str) -> Callable[[Dict[str, Any]], Optional[str]]:
    def _v(args: Dict[str, Any]) -> Optional[str]:
        val = (args.get(field) or "").strip()
        return None if _valid_url(val) else f"'{field}' must be a valid http(s) URL."
    return _v

def _v_enum(field: str, allowed: set[str]) -> Callable[[Dict[str, Any]], Optional[str]]:
    allowed_l = {a.lower() for a in allowed}
    def _v(args: Dict[str, Any]) -> Optional[str]:
        v = (args.get(field) or "").strip().lower()
        return None if v in allowed_l else f"'{field}' must be one of {sorted(list(allowed))}."
    return _v

def _v_ipv4(field: str) -> Callable[[Dict[str, Any]], Optional[str]]:
    def _v(args: Dict[str, Any]) -> Optional[str]:
        val = args.get(field)
        if val is None:
            return f"Missing required field '{field}'."
        try:
            ip = ip_address(val)
            if ip.version != 4:
                return f"'{field}' must be an IPv4 address."
        except Exception:
            return f"Invalid IPv4 address for '{field}'."
        return None
    return _v

def _v_has_namespace(args: dict) -> Optional[str]:
    ns = (args.get("namespace") or "").strip()
    return None if ns and not _is_placeholder(ns) else "'namespace' is required"

def _v_k8s_scale_args(args: dict) -> Optional[str]:
    name = args.get("deployment") or args.get("name")
    if not (isinstance(name, str) and name.strip() and not _is_placeholder(name)):
        return "either 'deployment' or 'name' is required"
    
    # Relaxed: if replicas is missing but action is present, we allow it (handled in registry)
    replicas = args.get("replicas")
    action = args.get("action")
    
    if replicas is None and action is None:
        # We'll allow missing replicas for "up/down" style calls or default to 1
        pass 
    elif replicas is not None:
        try:
            r = int(replicas)
            if r < 1:
                return "'replicas' must be >= 1"
        except Exception:
            return "'replicas' must be an integer"
    return None

# ------------------------------------------------------------------------------
# Prometheus Helpers
# ------------------------------------------------------------------------------
def _prom_url() -> Optional[str]:
    return os.getenv("PROMETHEUS_URL")

def _prom_request(path: str, params: Dict[str, Any], timeout: float = 10.0) -> dict:
    if not REQUESTS_AVAILABLE:
        return standardize_response("error", error="python-requests not installed")
    
    base = _prom_url()
    if not base:
        return standardize_response("error", error="PROMETHEUS_URL not set")
    
    session = RetrySession.create_session()
    try:
        url = f"{base.rstrip('/')}{path}"
        if session:
            response = session.get(url, params=params, timeout=timeout)
        else:
            response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return standardize_response("ok", data=response.json())
    except Exception as e:
        return standardize_response("error", error=f"Prometheus request failed: {str(e)}")

# ------------------------------------------------------------------------------
# Tool Implementations
# ------------------------------------------------------------------------------


# ---- Environment / World / Knowledge Tools ----

def get_environmental_data_tool(location: Optional[str] = "server_room_3",
                                data_type: str = "all",
                                use_real_sensors: bool = False) -> dict:
    """Get environmental sensor data"""
    _usage_tracker.track_usage("get_environmental_data_tool")
    
    # Try real sensors if requested
    if use_real_sensors and requests is not None:
        try:
            base = os.getenv("SENSOR_API_URL")
            if base and location:
                session = RetrySession.create_session()
                url = f"{base.rstrip('/')}/sensors/{location}"
                response = session.get(url, timeout=5) if session else requests.get(url, timeout=5)
                if response.status_code == 200:
                    return standardize_response("ok", data=response.json(), location=location, source="sensor_api")
        except Exception as e:
            logger.warning(f"Sensor API failed, using mock data: {e}")
    
    # Mock data fallback
    reading = {
        "temperature_celsius": round(19.5 + random.random() * 6.0, 2),
        "humidity_percent": round(30 + random.random() * 25, 1),
        "air_quality_index": int(40 + random.random() * 40),
        "pressure_hpa": round(1013 + random.random() * 10, 1),
        "noise_level_db": round(35 + random.random() * 20, 1)
    }
    
    allowed = {"all", "temperature_celsius", "humidity_percent", "air_quality_index", "pressure_hpa", "noise_level_db"}
    if data_type not in allowed:
        return standardize_response("error", error=f"Unsupported data_type. Use one of: {sorted(allowed)}")
    
    payload = reading if data_type == "all" else {data_type: reading[data_type]}
    return standardize_response("ok", data=payload, location=location, data_type=data_type, source="simulated")

@retry_on_failure(max_retries=3, delay=1.0)
def web_search_tool(query: str, max_results: int = 3, cancel_event: Optional[threading.Event] = None) -> dict:
    """Search the web using DuckDuckGo"""
    _usage_tracker.track_usage("web_search_tool")
    
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    if not query or _is_placeholder(query):
        return standardize_response("error", error="Query cannot be empty")
    
    if DDGS is None:
        return standardize_response("error", error="DuckDuckGo search not available")
    
    try:
        max_results = min(max(1, max_results), ToolConfig.MAX_SEARCH_RESULTS)
        results = []
        for res in DDGS().text(query, max_results=max_results):
            if cancel_event and cancel_event.is_set():
                return standardize_response(
                    "error",
                    error="cancelled",
                    data={"results": results, "query": query, "count": len(results)}
                )
            results.append(res)
        return standardize_response("ok", 
                                  data={"results": results, "query": query, "count": len(results)},
                                  summary=f"Found {len(results)} results for '{query}'")
    except Exception as e:
        _usage_tracker.track_usage("web_search_tool", success=False)
        return standardize_response("error", error=f"Search failed: {str(e)}")

def reply_to_user(message: str) -> dict:
    """Save reply to user-readable file"""
    _usage_tracker.track_usage("reply_to_user")
    
    if not message or _is_placeholder(message):
        return standardize_response("error", error="Message cannot be empty")
    
    try:
        # Ensure directory exists
        os.makedirs("persistence_data", exist_ok=True)
        
        with open("persistence_data/latest_response.txt", "w", encoding='utf-8') as f:
            f.write(message)
        
        return standardize_response("ok", data={"message_length": len(message)}, summary="Reply saved successfully")
    except Exception as e:
        _usage_tracker.track_usage("reply_to_user", success=False)
        return standardize_response("error", error=str(e))

def update_resource_allocation_tool(resource_type: str, target_agent_name: str, new_allocation_percentage: int = None) -> dict:
    """Update resource allocation for agents"""
    _usage_tracker.track_usage("update_resource_allocation_tool")
    
    for k, v in (("resource_type", resource_type), ("target_agent_name", target_agent_name)):
        err = _require_non_placeholder(k, v)
        if err:
            return standardize_response("error", error=err)
    
    # Validate percentage if provided
    if new_allocation_percentage is not None:
        err = _validate_integer(new_allocation_percentage, 0, 100)
        if err:
            return standardize_response("error", error=f"Invalid allocation percentage: {err}")
    
    data = {
        "resource_type": resource_type,
        "target_agent": target_agent_name,
        "new_allocation": new_allocation_percentage,
        "timestamp": _now_iso()
    }
    
    return standardize_response("ok", data=data, summary=f"Updated {resource_type} for {target_agent_name}")

def long_sleep_tool(seconds: int = 60, cancel_event: Optional[threading.Event] = None) -> dict:
    """Sleep for N seconds (testing timeout behavior) with cooperative cancel."""
    try:
        s = int(seconds)
    except Exception:
        return standardize_response("error", error="seconds must be an integer")
    if s < 1:
        return standardize_response("error", error="seconds must be >= 1")
    s = min(s, 900)  # cap to 15 minutes to avoid runaway hangs
    end_time = time.time() + s
    while time.time() < end_time:
        if cancel_event and cancel_event.is_set():
            return standardize_response("error", error="sleep cancelled")
        time.sleep(1)
    return standardize_response("ok", data={"slept_seconds": s}, summary=f"Slept for {s} seconds")

def system_diagnostics_tool(cancel_event: Optional[threading.Event] = None) -> dict:
    """Return high-level system diagnostics for CVA."""
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    diag = {}
    try:
        diag["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        diag["memory_percent"] = psutil.virtual_memory().percent
    except Exception as e:
        diag["resource_error"] = str(e)

    try:
        diag["python_threads"] = threading.active_count()
    except Exception as e:
        diag["thread_error"] = str(e)

    # Active agents if orchestrator is available
    active_agents = None
    try:
        cva = globals().get("_cva_instance")
        if cva:
            if hasattr(cva, "_agents_lock"):
                with cva._agents_lock:
                    active_agents = len(getattr(cva, "agent_instances", {}) or {})
            else:
                active_agents = len(getattr(cva, "agent_instances", {}) or {})
        diag["active_agents"] = active_agents
    except Exception as e:
        diag["active_agents_error"] = str(e)

    # DB connectivity (best-effort)
    try:
        from db_postgres import health_check
        if health_check():
            diag["db_status"] = "ok"
        else:
            diag["db_status"] = "error: connection failed"
    except Exception as e:
        diag["db_status"] = f"error: {e}"

    # Recent tool timeouts/errors from logs (best-effort tail scan)
    try:
        log_path = Path("logs/catalyst.log")
        timeouts = 0
        failures = 0
        if log_path.exists():
            lines = log_path.read_text(errors="ignore").splitlines()[-500:]
            for line in lines:
                if "TOOL TIMEOUT" in line:
                    timeouts += 1
                if "TOOL FAILED" in line:
                    failures += 1
        diag["recent_tool_timeouts"] = timeouts
        diag["recent_tool_errors"] = failures
    except Exception as e:
        diag["log_scan_error"] = str(e)

    # Tool breaker snapshot (visibility)
    try:
        from tool_registry import tool_registry as _global_registry
        if _global_registry and hasattr(_global_registry, "_breaker_status_tool"):
            breaker = _global_registry._breaker_status_tool()
            diag["tool_breakers"] = breaker.get("data")
    except Exception as e:
        diag["tool_breakers_error"] = str(e)

    return standardize_response("ok", data=diag, summary="System diagnostics collected")

def self_test_tool(cancel_event: Optional[threading.Event] = None) -> dict:
    """Quick smoke test: DB ping, registry check, trivial tool call, agent instantiation check."""
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    report = {}

    # DB ping
    try:
        from db_postgres import health_check
        if health_check():
            report["db_ping"] = "ok"
        else:
            report["db_ping"] = "error: connection failed"
    except Exception as e:
        report["db_ping"] = f"error: {e}"

    # Registry initialized
    try:
        from tool_registry import tool_registry
        report["tool_registry_initialized"] = bool(tool_registry.list_tool_names())
    except Exception as e:
        report["tool_registry_initialized"] = f"error: {e}"

    # Trivial tool call
    try:
        from tool_registry import tool_registry
        trivial = tool_registry.safe_call("measure_responsiveness")
        status = trivial.get("status") if isinstance(trivial, dict) else "unknown"
        report["trivial_tool_call"] = status
    except Exception as e:
        report["trivial_tool_call"] = f"error: {e}"

    # Agent instantiation check (best effort)
    try:
        cva = globals().get("_cva_instance")
        if cva:
            with getattr(cva, "_agents_lock", threading.Lock()):
                report["agent_instances_detected"] = len(getattr(cva, "agent_instances", {}) or {})
        else:
            report["agent_instances_detected"] = "cva_instance_not_set"
    except Exception as e:
        report["agent_instances_detected"] = f"error: {e}"

    status = "ok"
    if any(isinstance(v, str) and v.startswith("error") for v in report.values()):
        status = "error"
    return standardize_response(status, data=report, summary="Self test completed")

def restart_agent_tool(agent_name: str, cancel_event: Optional[threading.Event] = None) -> dict:
    """Remove an agent from the registry and re-instantiate via AgentFactory."""
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")
    if not agent_name:
        return standardize_response("error", error="agent_name is required")

    cva = globals().get("_cva_instance")
    if not cva:
        return standardize_response("error", error="CVA instance not available")

    removed = False
    try:
        lock = getattr(cva, "_agents_lock", None)
        if lock:
            with lock:
                removed = cva.agent_instances.pop(agent_name, None) is not None
        else:
            removed = cva.agent_instances.pop(agent_name, None) is not None
    except Exception as e:
        return standardize_response("error", error=f"failed to remove agent: {e}")

    # Attempt re-spawn via AgentFactory if a spec exists
    respawned = False
    new_id = None
    try:
        if hasattr(cva, "agent_factory"):
            # Best-effort: reuse a generic spawn (purpose=agent_name)
            result = cva.agent_factory.spawn_agent(purpose=agent_name, context={}, parent_agent="restart_agent_tool", ttl_hours=24.0)
            if result and not isinstance(result, dict):
                respawned = True
                new_id = getattr(result, "spec", {}).agent_id if hasattr(result, "spec") else None
                # register into orchestrator
                lock = getattr(cva, "_agents_lock", None)
                if lock:
                    with lock:
                        cva.agent_instances[result.spec.agent_id] = result
                else:
                    cva.agent_instances[result.spec.agent_id] = result
    except Exception as e:
        return standardize_response("error", error=f"failed to respawn: {e}", data={"removed": removed})

    return standardize_response("ok", data={"removed": removed, "respawned": respawned, "new_agent_id": new_id}, summary="Agent restart attempted")

@retry_on_failure(max_retries=3, delay=1.0)
def read_webpage_tool(url: str, cancel_event: Optional[threading.Event] = None) -> dict:
    """Read and extract text content from webpage"""
    _usage_tracker.track_usage("read_webpage_tool")
    
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    if not url or _is_placeholder(url):
        return standardize_response("error", error="URL cannot be empty")
    
    if not _valid_url(url):
        return standardize_response("error", error="Invalid URL format")
    
    if requests is None:
        return standardize_response("error", error="Requests library not available")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        if cancel_event and cancel_event.is_set():
            return standardize_response("error", error="cancelled", data={"content": ""})

        session = requests.Session()
        timeout = min(ToolConfig.REQUEST_TIMEOUT, 20) if hasattr(ToolConfig, "REQUEST_TIMEOUT") else 20
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # will raise for 4xx/5xx including 401/403/404

        if cancel_event and cancel_event.is_set():
            return standardize_response("error", error="cancelled", data={"content": ""})

        response.encoding = response.apparent_encoding
        html = response.text
        text = re.sub('<[^<]+?>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > ToolConfig.MAX_WEBPAGE_SIZE:
            text = text[:ToolConfig.MAX_WEBPAGE_SIZE] + "... [truncated]"

        data = {
            "content": text,
            "url": url,
            "content_length": len(text),
            "encoding": response.encoding,
            "status_code": response.status_code
        }

        return standardize_response("ok", data=data, summary=f"Read {len(text)} characters from {url}")
    except Exception as e:
        _usage_tracker.track_usage("read_webpage_tool", success=False)
        err_msg = f"Error reading page: {str(e)}"
        return standardize_response("error", error=err_msg, data={"content": err_msg, "url": url})
        return standardize_response("error", error=str(e))

def send_desktop_notification_tool(title: str, message: str, cancel_event: Optional[threading.Event] = None) -> dict:
    """
    Send a desktop notification to the user.
    
    Args:
        title: The notification title
        message: The notification body text
        
    Returns:
        Dict with success status
    """
    import subprocess
    import platform
    
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    try:
        system = platform.system()
        
        if system == "Linux":
            proc = subprocess.Popen(["notify-send", title, message])
            waited = 0
            while proc.poll() is None and waited < 5:
                if cancel_event and cancel_event.is_set():
                    proc.terminate()
                    return standardize_response("error", error="cancelled")
                time.sleep(0.1)
                waited += 0.1
            if proc.poll() is None:
                proc.terminate()
                return standardize_response("error", error="timeout")
        elif system == "Darwin":  # macOS
            proc = subprocess.Popen(["osascript", "-e", f'display notification "{message}" with title "{title}"'])
            waited = 0
            while proc.poll() is None and waited < 5:
                if cancel_event and cancel_event.is_set():
                    proc.terminate()
                    return standardize_response("error", error="cancelled")
                time.sleep(0.1)
                waited += 0.1
            if proc.poll() is None:
                proc.terminate()
                return standardize_response("error", error="timeout")
        elif system == "Windows":
            try:
                import importlib
                win10toast = importlib.import_module('win10toast')
                toaster = win10toast.ToastNotifier()
                toaster.show_toast(title, message, duration=5)
            except ImportError:
                return standardize_response("ok", data={"simulated": True, "platform": system, "title": title, "message": message}, summary="Simulated Windows toast (win10toast missing)")
        return standardize_response("ok", data={"title": title, "message": message, "platform": system}, summary="Notification sent")
        
    except Exception as e:
        return standardize_response("error", error=str(e))

def redact_pii_tool(text: str) -> dict:
    """Redact PII (Personally Identifiable Information) from text"""
    _usage_tracker.track_usage("redact_pii_tool")
    
    if not text or _is_placeholder(text):
        return standardize_response("error", error="Text cannot be empty")
    
    try:
        # Enhanced PII patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b'
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        credit_card_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        
        redacted_text = text
        redactions = {}
        
        # Count and redact each type
        for pattern, replacement, label in [
            (email_pattern, '[EMAIL_REDACTED]', 'emails'),
            (phone_pattern, '[PHONE_REDACTED]', 'phones'),
            (ip_pattern, '[IP_REDACTED]', 'ips'),
            (ssn_pattern, '[SSN_REDACTED]', 'ssns'),
            (credit_card_pattern, '[CREDIT_CARD_REDACTED]', 'credit_cards')
        ]:
            matches = re.findall(pattern, redacted_text)
            redactions[label] = len(matches)
            redacted_text = re.sub(pattern, replacement, redacted_text)
        
        data = {
            "redacted_text": redacted_text,
            "redactions_applied": redactions,
            "original_length": len(text),
            "redacted_length": len(redacted_text)
        }
        
        total_redactions = sum(redactions.values())
        return standardize_response("ok", data=data, summary=f"Redacted {total_redactions} PII elements")
    except Exception as e:
        _usage_tracker.track_usage("redact_pii_tool", success=False)
        return standardize_response("error", error=str(e))

def check_calendar_tool(time_min_utc: str = None, time_max_utc: str = None, **kwargs) -> dict:
    """
    Check Google Calendar for events in a time range.
    
    Args:
        time_min_utc: Start time in ISO format (e.g., "2024-11-20T00:00:00Z")
        time_max_utc: End time in ISO format (e.g., "2024-11-20T23:59:59Z")
        date: Optional YYYY-MM-DD; if provided and time_min_utc is missing, use full day window
        
    Returns:
        Dict with calendar events
    """
    # 1. Handle synonyms (AI hallucinations)
    # The AI sometimes uses 'start_time' or 'start_date' instead of strict keys
    if kwargs.get("start_time") and not time_min_utc:
        time_min_utc = kwargs.get("start_time")
    if kwargs.get("start_date") and not time_min_utc:
        time_min_utc = kwargs.get("start_date")
        
    if kwargs.get("end_time") and not time_max_utc:
        time_max_utc = kwargs.get("end_time")
    if kwargs.get("end_date") and not time_max_utc:
        time_max_utc = kwargs.get("end_date")

    # 2. Handle 'date' shortcut (Laziness)
    if kwargs.get("date") and not time_min_utc:
        try:
            day = kwargs.get("date")
            time_min_utc = f"{day}T00:00:00Z"
            time_max_utc = f"{day}T23:59:59Z"
        except Exception:
            pass

    # 3. Guard Clause
    if not time_min_utc or not time_max_utc:
        return {
            "success": False,
            "data": None,
            "error": f"Missing required arguments. Received: time_min={time_min_utc}, date={kwargs.get('date')}"
        }

    # 4. Main Logic
    try:
        # Import inside function to avoid circular imports
        from gmail_agent import get_calendar_events
        
        # Call the function you just added to gmail_agent.py
        events = get_calendar_events(time_min_utc, time_max_utc)
        
        # Ensure we always return a dict, even if events is None/Empty
        safe_events = events if events else []
        
        return {
            "success": True,
            "data": {
                "events": safe_events,
                "count": len(safe_events)
            },
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }
    
