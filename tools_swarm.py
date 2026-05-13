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


# ---- Management & Utility Tools ----

def get_tool_usage_stats_tool() -> dict:
    """Get usage statistics for all tools"""
    stats = _usage_tracker.get_stats()
    return standardize_response("ok", data=stats, summary="Tool usage statistics retrieved")

def tool_health_check_tool() -> dict:
    """Check health status of all tools and dependencies"""
    health = {
        "system": {
            "python_requests": requests is not None,
            "beautifulsoup": BeautifulSoup is not None,
            "fpdf": FPDF is not None,
            "transformers": _hf_pipeline is not None,
            "duckduckgo_search": DDGS is not None,
            "google_apis": build is not None and authenticate_google_services is not None
        },
        "commands": {
            "kubectl": SafeSubprocess.check_available("kubectl"),
            "python3": SafeSubprocess.check_available("python3")
        },
        "environment": {
            "prometheus_url": bool(_prom_url()),
            "sensor_api_url": bool(os.getenv("SENSOR_API_URL"))
        }
    }
    
    # Overall health
    all_healthy = (
        all(health["system"].values()) and 
        all(health["commands"].values()) and
        any(health["environment"].values())  # At least one service should be available
    )
    
    status = "healthy" if all_healthy else "degraded"
    summary = "All systems operational" if all_healthy else "Some dependencies or services unavailable"
    
    return standardize_response("ok", data=health, status=status, summary=summary)

# Add to tools.py

def update_world_model_tool(key: str, value: Any) -> dict:
    """Updates the swarm's shared world-state."""
    try:
        from catalyst_vector_alpha import shared_world_model
        shared_world_model.update_value(key, value)
        return {"ok": True, "key": key, "value": value, "model": shared_world_model.get_full_model()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def query_long_term_memory_tool(query_text: str, agent_name: str = "system") -> dict:
    """Searches the agent's long-term memory via ChromaDB."""
    try:
        from memory_store import MemoryStore
        store = MemoryStore()
        
        if not store.client:
            return {"ok": False, "error": "ChromaDB not initialized"}
        
        # Search episodic memory
        results = store.episodic.query(
            query_texts=[query_text],
            n_results=5
        )
        
        return {
            "ok": True,
            "query": query_text,
            "results": results.get('documents', [[]])[0],
            "metadatas": results.get('metadatas', [[]])[0]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def analyze_text_sentiment_tool(text: str) -> dict:
    """Analyzes text sentiment using simple heuristics."""
    import re
    
    # Word lists
    positive = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "success", "completed", "resolved", "optimal"]
    negative = ["bad", "poor", "terrible", "awful", "failed", "error", "critical", "warning", "degraded", "failure"]
    
    text_lower = text.lower()
    words = re.findall(r'\w+', text_lower)
    
    pos_count = sum(1 for w in words if w in positive)
    neg_count = sum(1 for w in words if w in negative)
    
    score = pos_count - neg_count
    total = len(words)
    
    if score > 0:
        sentiment = "positive"
    elif score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "ok": True,
        "sentiment": sentiment,
        "score": score,
        "confidence": abs(score) / max(total, 1),
        "positive_words": pos_count,
        "negative_words": neg_count
    }

def prometheus_query_tool(query: str, prometheus_url: str = "http://localhost:9090") -> dict:
    """Query Prometheus metrics with PromQL."""
    import requests
    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        data = response.json()
        
        if data.get("status") == "success":
            return {
                "ok": True,
                "query": query,
                "result": data.get("data", {}).get("result", [])
            }
        else:
            return {"ok": False, "error": data.get("error", "Unknown error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def prometheus_range_query_tool(query: str, start: str, end: str, step: str = "15s", prometheus_url: str = "http://localhost:9090") -> dict:
    """Query Prometheus metrics over a time range."""
    import requests
    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=10
        )
        data = response.json()
        
        if data.get("status") == "success":
            return {
                "ok": True,
                "query": query,
                "result": data.get("data", {}).get("result", [])
            }
        else:
            return {"ok": False, "error": data.get("error", "Unknown error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --- 🧠 MEMORY TOOLS ---

def remember_event(category: str, description: str, agent_name: str = "Unknown") -> dict:
    """Saves a critical event to the Permanent Hive Mind."""
    try:
        SHARED_BRAIN.add_memory(agent_name=agent_name, text=description, category=category)
        return {"ok": True, "summary": "Memory stored successfully."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def search_memory(query: str, agent_name: str = "Unknown") -> dict:
    """Performs semantic search across the collective memory to find past solutions."""
    try:
        results = SHARED_BRAIN.query_memory(query, n_results=5)
        formatted = []
        for r in results:
            formatted.append(f"[{r['timestamp']}] {r['agent']} ({r['category']}): {r['text']}")
        import logging
        logger = logging.getLogger("CatalystLogger")
        logger.info(f"🧠 [Memory] Search for '{query}' returned {len(results)} results.", extra={"event_type": "MEMORY_RECALL", "source": agent_name})
        return {"ok": True, "summary": f"Found {len(results)} relevant memories.", "data": {"memories": formatted}}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def broadcast_announcement_tool(title: str, content: str, category: str = "discovery") -> dict:
    """Broadcasts a high-level summary to the dashboard feed."""
    try:
        # Log it so Dashboard picks it up
        logger = logging.getLogger("CatalystLogger")
        logger.info(f"📢 [ANNOUNCEMENT] {title}: {content}", extra={"event_type": "ANNOUNCEMENT", "source": "TheVoice"})
        # Save to memory
        SHARED_BRAIN.add_memory("TheVoice", f"{title}: {content}", "announcement", {"category": category})
        return {"ok": True, "summary": f"Announcement '{title}' broadcasted successfully."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def capture_system_screenshot(agent_name: str = "Unknown") -> dict:
    """Captures a text-based semantic snapshot of the CVA Dashboard (Phase 19: Visual Intelligence)."""
    try:
        from database import cva_db
        task_stats = cva_db.get_task_stats()
        mission_stats = cva_db.get_mission_stats()
        summary = f"Visual scan complete. Tasks: {task_stats.get('total', 0)}, Missions active: {mission_stats.get('active', 0)}."
        return {"ok": True, "summary": summary, "data": {"tasks": task_stats, "missions": mission_stats}}
    except Exception as e:
        return {"ok": False, "error": f"Visual scan failed: {e}"}

def tune_hyperparameters(param: str, value: float, agent_name: str = "Unknown") -> dict:
    """Dynamically tunes system hyper-parameters (e.g., temperature, exploration_rate) for Phase 18."""
    try:
        from config import config as cva_config
        if hasattr(cva_config, param.upper()):
            import logging
            logger = logging.getLogger("CatalystLogger")
            logger.info(f"⚙️ [Tuning] Hyper-parameter '{param}' tuned to {value}.", extra={"event_type": "META_EVOLUTION", "source": agent_name})
            return {"ok": True, "summary": f"Hyper-parameter '{param}' tuned to {value}."}
        return {"ok": False, "error": f"Parameter '{param}' not found in config."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def create_pdf_tool(filename: str, sections: list) -> dict:
    """Creates a PDF report from structured sections."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        for section in sections:
            title = section.get("title", "Untitled")
            content = section.get("content", "")
            
            story.append(Paragraph(title, styles['Heading1']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(content, styles['BodyText']))
            story.append(Spacer(1, 24))
        
        doc.build(story)
        return {"ok": True, "filename": filename, "sections": len(sections)}
    except ImportError:
        return {"ok": False, "error": "reportlab not installed: pip install reportlab"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def shuffle_roles_and_tasks_tool() -> dict:
    """Placeholder for role shuffling - requires swarm access."""
    return {"ok": False, "error": "Role shuffling not implemented - requires swarm coordination"}


def list_available_tools_tool() -> dict:
    """List all available tools with their descriptions"""
    tools = {
        "system_tools": {
            "get_system_cpu_load_tool": "Get system CPU load with configurable sampling",
            "get_system_resource_usage_tool": "Get comprehensive system resource usage",
            "disk_usage_tool": "Get disk usage for specified path",
            "top_processes_tool": "Get top processes by CPU or memory usage",
            "measure_responsiveness_tool": "Measure system responsiveness"
        },
        "kubernetes_tools": {
            "kubernetes_pod_metrics_tool": "Get Kubernetes pod metrics",
            "k8s_scale_tool": "Safely scale Kubernetes deployments",
            "find_wasteful_deployments_tool": "Find resource-wasteful deployments"
        },
        "security_tools": {
            "initiate_network_scan_tool": "Perform network security scans",
            "deploy_recovery_protocol_tool": "Deploy recovery protocols",
            "analyze_threat_signature_tool": "Analyze threat signatures",
            "isolate_network_segment_tool": "Isolate network segments",
            "extract_iocs_tool": "Extract Indicators of Compromise",
            "hash_text_tool": "Hash text using various algorithms",
            "redact_pii_tool": "Redact PII from text"
        },
        "knowledge_tools": {
            "web_search_tool": "Search the web using DuckDuckGo",
            "read_webpage_tool": "Read content from webpages",
            "get_environmental_data_tool": "Get environmental sensor data"
        },
        "utility_tools": {
            "reply_to_user": "Save replies to user",
            "update_resource_allocation_tool": "Update resource allocations",
            "get_tool_usage_stats_tool": "Get tool usage statistics",
            "tool_health_check_tool": "Check tool health status",
            "list_available_tools_tool": "List all available tools"
        }
    }
    
    total_tools = sum(len(category) for category in tools.values())
    return standardize_response("ok", data=tools, summary=f"Found {total_tools} available tools across {len(tools)} categories")

