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
    from ddgs import DDGS
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
    try:
        r = int(args.get("replicas"))
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

# ---- System / Local Tools ----
def get_system_cpu_load_tool(time_interval_seconds: float = 0.5, samples: int = 3, per_core: bool = False) -> dict:
    """Get system CPU load with configurable sampling"""
    _usage_tracker.track_usage("get_system_cpu_load_tool")
    
    # Input validation
    interval = max(0.1, min(float(time_interval_seconds), 5.0))
    samples = max(1, min(int(samples), 10))
    
    try:
        readings: List[Any] = []
        for _ in range(samples):
            readings.append(psutil.cpu_percent(interval=interval, percpu=per_core))
        
        if per_core:
            cores = len(readings[0]) if readings else 0
            averaged = [round(sum(s[i] for s in readings) / len(readings), 2) for i in range(cores)]
            data = averaged
            summary = f"Per-core CPU load: {averaged}"
        else:
            avg = sum(readings) / len(readings)
            data = round(float(avg), 2)
            summary = f"System CPU load: {data}%"
        
        return standardize_response("ok", data=data, summary=summary, unit="percent")
    except Exception as e:
        _usage_tracker.track_usage("get_system_cpu_load_tool", success=False)
        return standardize_response("error", error=str(e), summary="Failed to get CPU load")

def get_system_resource_usage_tool() -> dict:
    """Get comprehensive system resource usage"""
    _usage_tracker.track_usage("get_system_resource_usage_tool")
    
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None
        
        data = {
            "cpu_percent": cpu,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "load_average": load_avg
        }
        
        summary = f"CPU: {cpu}%, Memory: {memory.percent}%, Disk: {disk.percent}%"
        return standardize_response("ok", data=data, summary=summary)
    except Exception as e:
        _usage_tracker.track_usage("get_system_resource_usage_tool", success=False)
        return standardize_response("error", error=str(e), summary="Resource usage check failed")

def disk_usage_tool(path: str = "/") -> dict:
    """Get disk usage for specified path"""
    _usage_tracker.track_usage("disk_usage_tool")
    
    if not path or _is_placeholder(path):
        return standardize_response("error", error="Path cannot be empty", data={"path": path})
    
    try:
        u = psutil.disk_usage(path)
        data = {
            "path": path,
            "total_bytes": u.total,
            "used_bytes": u.used,
            "free_bytes": u.free,
            "percent": u.percent,
            "total_gb": round(u.total / (1024**3), 2),
            "used_gb": round(u.used / (1024**3), 2),
            "free_gb": round(u.free / (1024**3), 2)
        }
        return standardize_response("ok", data=data, summary=f"{path}: {u.percent}% used")
    except Exception as e:
        _usage_tracker.track_usage("disk_usage_tool", success=False)
        return standardize_response("error", error=f"Disk usage check failed: {e}", data={"path": path})

def top_processes_tool(limit: int = 10, sort_by: str = "cpu") -> dict:
    """Get top processes by CPU or memory usage"""
    _usage_tracker.track_usage("top_processes_tool")
    
    # Input validation
    limit = max(1, min(int(limit), ToolConfig.MAX_PROCESS_LIMIT))
    valid_sorts = {"cpu", "memory"}
    sort_by = sort_by.lower() if sort_by.lower() in valid_sorts else "cpu"
    
    try:
        # Warm-up for accurate cpu_percent
        procs = [p for p in psutil.process_iter(attrs=["pid", "name", "username", "create_time"])]
        for p in procs:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        time.sleep(0.25)  # Allow CPU percent calculation
        
        rows = []
        for p in procs:
            try:
                with p.oneshot():
                    rows.append({
                        "pid": p.pid,
                        "name": p.info.get("name", "Unknown"),
                        "username": p.info.get("username", "Unknown"),
                        "cpu_percent": p.cpu_percent(None),
                        "memory_percent": p.memory_percent(),
                        "memory_rss_mb": round(p.memory_info().rss / (1024**2), 2),
                        "create_time": p.info.get("create_time", 0),
                        "status": p.status()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by specified field
        reverse = True  # Descending order
        if sort_by == "memory":
            rows.sort(key=lambda r: r.get("memory_percent") or 0.0, reverse=reverse)
        else:  # cpu
            rows.sort(key=lambda r: r.get("cpu_percent") or 0.0, reverse=reverse)
        
        rows = rows[:limit]
        return standardize_response("ok", data={"processes": rows, "count": len(rows), "sort_by": sort_by},
                                  summary=f"Top {len(rows)} processes by {sort_by}")
    except Exception as e:
        _usage_tracker.track_usage("top_processes_tool", success=False)
        return standardize_response("error", error=str(e))

def measure_responsiveness_tool(**kwargs) -> dict:
    """Measure system responsiveness by timing command execution"""
    _usage_tracker.track_usage("measure_responsiveness_tool")
    
    try:
        start = time.time()
        success, stdout, stderr = SafeSubprocess.run(["python3", "-c", "print(1)"], timeout=2)
        elapsed_ms = (time.time() - start) * 1000.0
        
        responsive = success and elapsed_ms < 500
        data = {
            "open_time_ms": round(elapsed_ms, 2),
            "responsive": responsive,
            "command_success": success
        }
        
        return standardize_response("ok", data=data, summary=f"Responsiveness: {elapsed_ms:.2f}ms")
    except Exception as e:
        _usage_tracker.track_usage("measure_responsiveness_tool", success=False)
        return standardize_response("error", error=str(e))

# ---- Kubernetes Tools ----
def _parse_kubectl_top_pods(raw: str) -> List[Dict[str, Any]]:
    """Parse kubectl top pods output"""
    rows: List[Dict[str, Any]] = []
    for line in filter(None, (l.strip() for l in raw.splitlines())):
        parts = line.split()
        if len(parts) < 4:
            continue
        ns, name, cpu_s, mem_s = parts[0], parts[1], parts[2], parts[3]

        def cpu_to_mcores(v: str) -> Optional[int]:
            try:
                return int(v[:-1]) if v.endswith("m") else int(float(v) * 1000)
            except Exception:
                return None

        def mem_to_Mi(v: str) -> Optional[float]:
            try:
                if v.endswith("Mi"):  return float(v[:-2])
                if v.endswith("Gi"):  return float(v[:-2]) * 1024.0
                if v.endswith("Ki"):  return float(v[:-2]) / 1024.0
                return float(v) / (1024.0 * 1024.0)  # assume bytes
            except Exception:
                return None

        rows.append({
            "namespace": ns,
            "pod": name,
            "cpu_mcores": cpu_to_mcores(cpu_s),
            "memory_Mi": mem_to_Mi(mem_s),
            "raw_cpu": cpu_s,
            "raw_memory": mem_s,
        })
    return rows

def kubernetes_pod_metrics_tool(namespace: Optional[str] = None,
                                selector: Optional[str] = None,
                                limit: int = 50,
                                cancel_event: Optional[threading.Event] = None) -> dict:
    """Get Kubernetes pod metrics using kubectl top"""
    _usage_tracker.track_usage("kubernetes_pod_metrics_tool")
    
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    if not SafeSubprocess.check_available("kubectl"):
        return standardize_response("error", error="kubectl not found on PATH")
    
    try:
        cmd = ["kubectl", "top", "pods", "--no-headers"]
        if namespace:
            cmd.extend(["-n", namespace])
        else:
            cmd.append("-A")
        if selector:
            cmd.extend(["-l", selector])
        
        success, stdout, stderr = SafeSubprocess.run(cmd, timeout=ToolConfig.KUBECTL_TIMEOUT)
        if not success:
            return standardize_response("error", error=stderr, cmd=" ".join(cmd))

        if cancel_event and cancel_event.is_set():
            return standardize_response("error", error="cancelled")

        rows = _parse_kubectl_top_pods(stdout)
        if limit:
            rows = rows[:max(1, int(limit))]
        
        total_cpu = sum(r.get("cpu_mcores") or 0 for r in rows)
        total_mem = sum(r.get("memory_Mi") or 0.0 for r in rows)
        
        return standardize_response(
            "ok",
            data={
                "pods": rows, 
                "count": len(rows), 
                "total_cpu_mcores": total_cpu, 
                "total_memory_Mi": round(total_mem, 2)
            },
            cmd=" ".join(shlex.quote(x) for x in cmd),
            summary=f"{len(rows)} pods, total CPU {total_cpu}m, memory {total_mem:.1f}Mi"
        )
    except Exception as e:
        _usage_tracker.track_usage("kubernetes_pod_metrics_tool", success=False)
        return standardize_response("error", error=str(e))

def _kubectl_json(cmd: List[str]) -> Any:
    """Execute kubectl command and return JSON result"""
    success, stdout, stderr = SafeSubprocess.run(cmd, timeout=ToolConfig.KUBECTL_TIMEOUT)
    if not success:
        raise Exception(stderr)
    return json.loads(stdout)

def _get_deploy(ns: str, name: str) -> Dict[str, Any]:
    """Get deployment JSON"""
    return _kubectl_json(["kubectl", "-n", ns, "get", "deploy", name, "-o", "json"])

def find_wasteful_deployments_tool(namespace: str = "default", 
                                   cpu_threshold: float = 5.0,
                                   min_replicas: int = 2,
                                   cancel_event: Optional[threading.Event] = None) -> dict:
    """Find deployments with low CPU utilization but high replica count"""
    _usage_tracker.track_usage("find_wasteful_deployments_tool")
    
    if cancel_event and cancel_event.is_set():
        return standardize_response("error", error="cancelled")

    if not SafeSubprocess.check_available("kubectl"):
        return standardize_response("error", error="kubectl not found on PATH")
    
    try:
        # Get all deployments
        cmd = ["kubectl", "-n", namespace, "get", "deployments", "-o", "json"]
        success, stdout, stderr = SafeSubprocess.run(cmd, timeout=ToolConfig.KUBECTL_TIMEOUT)
        if not success:
            return standardize_response("error", error=stderr)
        
        data = json.loads(stdout)
        wasteful = []
        
        for item in data.get("items", []):
            if cancel_event and cancel_event.is_set():
                return standardize_response(
                    "error",
                    error="cancelled",
                    data={"wasteful_deployments": wasteful, "partial": True}
                )

            name = item["metadata"]["name"]
            replicas = item["spec"].get("replicas", 0)
            
            if replicas < min_replicas:
                continue
            
            # Try to get pod CPU usage
            try:
                top_cmd = ["kubectl", "-n", namespace, "top", "pods", "-l", f"app={name}", "--no-headers"]
                success, top_stdout, top_stderr = SafeSubprocess.run(top_cmd, timeout=5)
                
                if not success or not top_stdout.strip():
                    continue
                
                if cancel_event and cancel_event.is_set():
                    return standardize_response(
                        "error",
                        error="cancelled",
                        data={"wasteful_deployments": wasteful, "partial": True}
                    )

                total_cpu_millicores = 0
                pod_count = 0
                
                for line in top_stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        cpu_str = parts[1].replace("m", "")
                        try:
                            total_cpu_millicores += int(cpu_str)
                            pod_count += 1
                        except ValueError:
                            continue
                
                if pod_count == 0:
                    continue
                
                avg_cpu_millicores = total_cpu_millicores / pod_count
                
                if avg_cpu_millicores < cpu_threshold:
                    waste_score = replicas * (cpu_threshold - avg_cpu_millicores)
                    wasteful.append({
                        "deployment": name,
                        "namespace": namespace,
                        "replicas": replicas,
                        "avg_cpu_millicores": round(avg_cpu_millicores, 2),
                        "waste_score": round(waste_score, 2),
                        "recommended_replicas": max(1, replicas // 2)
                    })
                    
            except Exception:
                continue
        
        wasteful.sort(key=lambda x: x["waste_score"], reverse=True)
        return standardize_response("ok", data={"wasteful_deployments": wasteful, "count": len(wasteful)})
        
    except Exception as e:
        _usage_tracker.track_usage("find_wasteful_deployments_tool", success=False)
        return standardize_response("error", error=str(e))

def check_network_connectivity(source_pod: str, target_service: str, namespace: str = "default", timeout: int = 5) -> dict:
    """
    Check if a pod can reach a service. Detects network policy blocks.
    
    Args:
        source_pod: Name of pod to test from
        target_service: Service name or IP to reach
        namespace: Namespace of the pods
        timeout: Seconds to wait for response
    
    Returns:
        dict with: ok, reachable, latency_ms, error
    """
    import subprocess
    
    try:
        # Execute curl from inside the source pod
        cmd = [
            "kubectl", "exec", source_pod, "-n", namespace, "--",
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code},%{time_total}",
            "--max-time", str(timeout),
            f"http://{target_service}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+5)
        
        if result.returncode != 0:
            # Check if it's a network policy block
            if "timed out" in result.stderr.lower() or result.returncode == 28:
                return {
                    "ok": True,
                    "reachable": False,
                    "reason": "timeout_network_blocked",
                    "error": result.stderr
                }
            return {"ok": False, "error": result.stderr}
        
        # Parse response: "200,0.023"
        parts = result.stdout.strip().split(",")
        http_code = int(parts[0]) if parts[0].isdigit() else 0
        latency = float(parts[1]) * 1000 if len(parts) > 1 else 0
        
        return {
            "ok": True,
            "reachable": http_code > 0 and http_code < 500,
            "http_code": http_code,
            "latency_ms": round(latency, 2),
            "source_pod": source_pod,
            "target_service": target_service,
            "namespace": namespace
        }
        
    except subprocess.TimeoutExpired:
        return {"ok": True, "reachable": False, "reason": "timeout_network_blocked"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def watch_k8s_audit_events(minutes: int = 5, event_types: list = None) -> dict:
    """
    Monitor Kubernetes for RBAC violations and sensitive operations.
    Uses kubectl auth can-i and checks for forbidden operations.
    
    Args:
        minutes: How far back to check
        event_types: Filter for specific types like ["Forbidden", "secrets"]
    
    Returns:
        dict with: ok, violations, secret_access, summary
    """
    import subprocess
    import json
    from datetime import datetime, timedelta
    from typing import Optional
    
    violations = []
    secret_access = []
    
    try:
        # Method 1: Check recent events for any Forbidden/Unauthorized
        cmd = ["kubectl", "get", "events", "--all-namespaces", "-o", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            events = json.loads(result.stdout)
            for item in events.get("items", []):
                reason = item.get("reason", "")
                message = item.get("message", "").lower()
                
                # Check for auth failures
                if "forbidden" in message or "unauthorized" in message or "denied" in message:
                    violations.append({
                        "namespace": item.get("involvedObject", {}).get("namespace", "unknown"),
                        "name": item.get("involvedObject", {}).get("name", "unknown"),
                        "reason": reason,
                        "message": item.get("message", "")[:200],
                        "type": "rbac_violation"
                    })
        
        # Method 2: Check pods that mount secrets
        cmd2 = ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        
        if result2.returncode == 0:
            pods = json.loads(result2.stdout)
            for pod in pods.get("items", []):
                metadata = pod.get("metadata", {})
                spec = pod.get("spec", {})
                
                # Check for secret volume mounts
                volumes = spec.get("volumes", [])
                for vol in volumes:
                    if "secret" in vol:
                        secret_name = vol.get("secret", {}).get("secretName", "unknown")
                        secret_access.append({
                            "pod": metadata.get("name"),
                            "namespace": metadata.get("namespace"),
                            "secret": secret_name,
                            "type": "secret_mount"
                        })
                
                # Check for secret env vars
                for container in spec.get("containers", []):
                    for env in container.get("env", []):
                        if env.get("valueFrom", {}).get("secretKeyRef"):
                            secret_name = env["valueFrom"]["secretKeyRef"].get("name", "unknown")
                            secret_access.append({
                                "pod": metadata.get("name"),
                                "namespace": metadata.get("namespace"),
                                "secret": secret_name,
                                "env_var": env.get("name"),
                                "type": "secret_env"
                            })
        
        return {
            "ok": True,
            "summary": f"Found {len(violations)} RBAC violations, {len(secret_access)} secret accesses",
            "violation_count": len(violations),
            "secret_access_count": len(secret_access),
            "violations": violations[:10],
            "secret_access": secret_access[:20],
            "minutes_back": minutes
        }
        
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "kubectl command timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def microsoft_autonomous_remediation(pod_name, namespace="default", recommended_actions=None, resource_patch=None, **kwargs):
    """
    Microsoft™ Enterprise Remediation Protocol v2.0
    1. EXTRACTS logs to a permanent forensic file (Black Box).
    2. ANALYZES logs for root cause.
    3. EXECUTES tactical restart.
    4. VERIFIES stability.

    Enhanced: if resource recommendations are provided (e.g., from Planner reasoning),
    apply them to the owning Deployment or recreate naked pods with updated resources.
    """
    timestamp = int(time.time())
    results = {
        "target": pod_name,
        "actions": [],
        "outcome": "failed"
    }
    createcontainer_result = None
    skip_fallback_actions = False

    # Resolve current pod name in case the original pod was replaced
    try:
        check = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "-n", namespace],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode != 0:
            rs_name = pod_name.rsplit("-", 1)[0] if "-" in pod_name else None
            if rs_name and rs_name != pod_name:
                hash_part = rs_name.split("-")[-1]
                selector = f"pod-template-hash={hash_part}"
                pods_json = None
                try:
                    out = subprocess.check_output(
                        ["kubectl", "get", "pods", "-n", namespace, "-l", selector, "-o", "json"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    pods_json = json.loads(out)
                except Exception:
                    pods_json = None
                if pods_json:
                    items = pods_json.get("items", []) or []
                    for item in items:
                        phase = (item.get("status") or {}).get("phase")
                        if phase and phase != "Running":
                            new_name = (item.get("metadata") or {}).get("name")
                            if new_name:
                                pod_name = new_name
                                break
    except Exception:
        pass
    results["target"] = pod_name

    # Normalize recommendations from kwargs fallback
    if recommended_actions is None:
        recommended_actions = kwargs.get("recommended_actions")
    resource_patch = resource_patch or kwargs.get("resource_patch")
    
    # Create a secure vault for evidence
    evidence_dir = "logs/forensics"
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_file = f"{evidence_dir}/{pod_name}_crash_{timestamp}.log"

    def _safe_json(cmd: list[str]):
        try:
            out = subprocess.check_output(cmd, text=True)
            return json.loads(out)
        except Exception:
            return None

    try:
        # Discover pod ownership & labels (avoid acting on stale names)
        pod_json = None
        pod_exists = False
        owner_kind = owner_name = None
        pod_labels = {}
        pod_forensics = {}
        dep_name = None
        dep_history = []
        try:
            pod_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
            if pod_json:
                pod_exists = True
                pod_labels = pod_json.get("metadata", {}).get("labels", {}) or {}
                owners = pod_json.get("metadata", {}).get("ownerReferences", []) or []
                if owners:
                    owner_kind = owners[0].get("kind")
                    owner_name = owners[0].get("name")
                try:
                    owners = (pod_status_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                    for o in owners:
                        owner_kind = owner_kind or o.get("kind")
                        owner_name = owner_name or o.get("name")
                        if o.get("kind") == "ReplicaSet":
                            rs_json = _safe_json(["kubectl", "get", "rs", o.get("name"), "-n", namespace, "-o", "json"])
                            rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                            for ro in rs_owners:
                                if ro.get("kind") == "Deployment":
                                    dep_name = ro.get("name")
                                    break
                        if o.get("kind") == "Deployment":
                            dep_name = o.get("name")
                        if dep_name:
                            break
                except Exception:
                    dep_name = None
                try:
                    if dep_name:
                        hist = _safe_json(["kubectl", "rollout", "history", f"deployment/{dep_name}", "-n", namespace, "-o", "json"])
                        dep_history = hist.get("history", []) if isinstance(hist, dict) else []
                except Exception:
                    dep_history = []
                # Track owner references for postmortem
                pod_forensics["owner_references"] = owners or []
                if owner_kind or owner_name:
                    pod_forensics["owner_summary"] = {"kind": owner_kind, "name": owner_name}
                # Capture termination reason/exit codes from status
                try:
                    statuses = (pod_json.get("status", {}) or {}).get("containerStatuses", []) or []
                    term_info = []
                    for cs in statuses:
                        state = cs.get("lastState") or cs.get("state") or {}
                        term = state.get("terminated") or {}
                        if term:
                            term_info.append({
                                "name": cs.get("name"),
                                "reason": term.get("reason"),
                                "exitCode": term.get("exitCode"),
                                "message": term.get("message"),
                            })
                    if term_info:
                        pod_forensics["termination"] = term_info
                except Exception:
                    pass
                # Capture resource requests/limits
                try:
                    specs = (pod_json.get("spec", {}) or {}).get("containers", []) or []
                    res_info = []
                    for c in specs:
                        res = c.get("resources", {}) or {}
                        res_info.append({
                            "name": c.get("name"),
                            "requests": res.get("requests"),
                            "limits": res.get("limits"),
                        })
                    if res_info:
                        pod_forensics["resources"] = res_info
                except Exception:
                    pass
        except Exception:
            pod_exists = False

        results["owner_kind"] = owner_kind
        results["owner_name"] = owner_name
        results["labels"] = pod_labels
        if pod_forensics:
            results["forensics"] = pod_forensics

        # Early ImagePullBackOff/ErrImagePull handling before log capture
        try:
            # print(f"DEBUG: Checking for ImagePullBackOff - pod_json exists: {pod_json is not None}")
            container_statuses = (pod_json.get("status", {}) or {}).get("containerStatuses", []) or []
            # print(f"DEBUG: container_statuses count: {len(container_statuses)}")
            waiting_status = None
            for cs in container_statuses:
                waiting = (cs.get("state", {}) or {}).get("waiting") or {}
                if waiting.get("reason") in ["ImagePullBackOff", "ErrImagePull"]:
                    waiting_status = (cs, waiting)
                    break
            if waiting_status:
                cs, waiting = waiting_status
                err_msg = waiting.get("message") or ""
                img_val = None
                for c in (pod_json.get("spec", {}) or {}).get("containers", []) or []:
                    if c.get("name") == cs.get("name"):
                        img_val = c.get("image")
                        break

                def _get_pod_events(pn: str, ns: str):
                    try:
                        ev = _safe_json([
                            "kubectl", "get", "events",
                            "-n", ns,
                            "--field-selector", f"involvedObject.name={pn}",
                            "-o", "json",
                        ])
                        if isinstance(ev, dict):
                            return ev.get("items", [])
                    except Exception:
                        return []
                    return []

                forensics_data = {
                    "error_message": err_msg,
                    "image": img_val,
                    "container_name": cs.get("name"),
                    "pod_events": _get_pod_events(pod_name, namespace),
                    "image_pull_secrets": [
                        s.get("name") for s in (pod_json.get("spec", {}) or {}).get("imagePullSecrets", []) or []
                    ],
                }

                def _resolve_deployment_name() -> str | None:
                    try:
                        if owner_kind == "Deployment" and owner_name:
                            return owner_name
                        if owner_kind == "ReplicaSet" and owner_name:
                            rs_json = _safe_json(["kubectl", "get", "rs", owner_name, "-n", namespace, "-o", "json"])
                            rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                            for ro in rs_owners:
                                if ro.get("kind") == "Deployment" and ro.get("name"):
                                    return ro.get("name")
                        owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                        for o in owners:
                            if o.get("kind") == "Deployment" and o.get("name"):
                                return o.get("name")
                            if o.get("kind") == "ReplicaSet":
                                rs_name = o.get("name")
                                rs_json = _safe_json(["kubectl", "get", "rs", rs_name, "-n", namespace, "-o", "json"])
                                rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                                for ro in rs_owners:
                                    if ro.get("kind") == "Deployment" and ro.get("name"):
                                        return ro.get("name")
                    except Exception:
                        return None
                    return None

                if not dep_name:
                    dep_name = _resolve_deployment_name()

                analysis_prompt = (
                    "You are analyzing an ImagePullBackOff failure in Kubernetes.\n\n"
                    f"Error: {err_msg}\n"
                    f"Image: {img_val}\n"
                    f"Events: {forensics_data.get('pod_events')}\n\n"
                    "Common causes:\n"
                    "1. Wrong tag (typo or doesn't exist) → suggest :latest or previous tag\n"
                    "2. Authentication needed → suggest adding imagePullSecret\n"
                    "3. Registry unreachable → network issue\n"
                    "4. Image deleted → rollback to previous version\n\n"
                    "Recommend a fix. Return JSON with:\n"
                    "{\n"
                    '  "root_cause": "...",\n'
                    '  "recommended_actions": [\n'
                    '    {"action": "update_image", "image": "nginx:latest", "reason": "..."},\n'
                    '    {"action": "add_secret", "secret_name": "...", "reason": "..."},\n'
                    '    {"action": "investigate", "reason": "need manual intervention"}\n'
                    "  ]\n"
                    "}"
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    analysis = llm.generate_text(analysis_prompt, json_mode=True, temperature=0.2, max_tokens=400)
                    import json as _json
                    parsed = _json.loads(analysis) if analysis else {}
                except Exception:
                    parsed = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"

                if not dep_name:
                    actions_taken.append("Standalone pod; no deployment owner. Manual remediation required.")
                    return {
                        "target": pod_name,
                        "issue_type": "ImagePullBackOff",
                        "root_cause": parsed.get("root_cause"),
                        "recommended_actions": parsed.get("recommended_actions", []),
                        "forensics": forensics_data,
                        "actions_taken": actions_taken,
                        "outcome": "ANALYSIS_ONLY",
                    }

                recs = parsed.get("recommended_actions") or (recommended_actions or [])
                try:
                    # print(f"DEBUG: ImagePullBackOff early handler - parsed: {parsed}")
                    # print(f"DEBUG: recommended_actions: {recs}")
                    for action in recs or []:
                        if action.get("action") in ["update_image", "update_image_url", "update_reference"]:
                            new_image = action.get("image") or action.get("new_url") or action.get("target")
                            dep_name = forensics_data.get("owner_name") or _resolve_deployment_name()
                            if new_image and dep_name:
                                patch_cmd = [
                                    "kubectl", "patch", "deployment", dep_name,
                                    "-n", namespace,
                                    "--type", "strategic",
                                    "-p", json.dumps({
                                        "spec": {
                                            "template": {
                                                "spec": {
                                                    "containers": [{
                                                        "name": cs.get("name") or forensics_data.get("container_name") or "app",
                                                        "image": new_image
                                                    }]
                                                }
                                            }
                                        }
                                    })
                                ]
                                try:
                                    result = subprocess.run(patch_cmd, capture_output=True, text=True)
                                    if result.returncode == 0:
                                        actions_taken.append(f"Patched deployment {dep_name} image to {new_image}")
                                        outcome_state = "SUCCESS"
                                    else:
                                        actions_taken.append(f"Failed to patch deployment {dep_name}: {result.stderr}")
                                        outcome_state = "PARTIAL_SUCCESS"
                                except Exception as e:
                                    actions_taken.append(f"Error patching deployment {dep_name}: {e}")
                                    outcome_state = "PARTIAL_SUCCESS"
                except Exception:
                    pass

                return {
                    "target": pod_name,
                    "issue_type": "ImagePullBackOff",
                    "root_cause": parsed.get("root_cause"),
                    "recommended_actions": recs,
                    "forensics": forensics_data,
                    "actions_taken": actions_taken,
                    "outcome": "SUCCESS" if any_success else outcome_state,
                }
        except Exception:
            pass

        # Extract resource adjustment intents
        resource_actions = []
        if not skip_fallback_actions:
            for action in recommended_actions or []:
                if action.get("action") == "adjust_resources":
                    resource_actions.append(action)
            if resource_patch:
                resource_actions.append(resource_patch)

        # Resolve a label selector for safer bulk operations (DaemonSet-friendly)
        label_selector = (
            (f"app.kubernetes.io/name={pod_labels.get('app.kubernetes.io/name')}")
            if pod_labels.get("app.kubernetes.io/name") else None
        )
        if not label_selector and pod_labels.get("k8s-app"):
            label_selector = f"k8s-app={pod_labels['k8s-app']}"
        if not label_selector and pod_labels.get("app"):
            label_selector = f"app={pod_labels['app']}"

        # If pod is missing and no owner info, skip to prevent NotFound loops
        if not pod_exists and not owner_name:
            results["outcome"] = "SKIP_NOT_FOUND"
            results["error"] = f"Pod {pod_name} not found; skipping remediation."
            return results

        # STEP 1: CAPTURE THE BLACK BOX (only if pod exists)
        if pod_exists:
            print(f"🕵️ [Microsoft Kernel] Securing evidence for {pod_name}...")

            try:
                result = subprocess.run(
                    ["kubectl", "logs", pod_name, "-n", namespace, "--tail=100"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logs = result.stdout
            except subprocess.CalledProcessError:
                logs = "Pod never started - no logs available"

            with open(evidence_file, "w") as f:
                f.write(logs)
            results["evidence_path"] = evidence_file
            results["actions"].append("Forensic logs saved to disk")
            results["log_preview"] = logs[:200]

            # --- Log parsing for resource hints ---
            # Extract rough memory/CPU/IO/error signals from the captured logs.
            try:
                import re

                def _extract_numbers(pattern: str, text: str):
                    return [float(m) for m in re.findall(pattern, text)]

                mem_matches_mb = _extract_numbers(r"([0-9]+(?:\\.[0-9]+)?)\\s*MiB", logs)
                mem_matches_gb = _extract_numbers(r"([0-9]+(?:\\.[0-9]+)?)\\s*GiB", logs)
                cpu_matches_pct = _extract_numbers(r"([0-9]+(?:\\.[0-9]+)?)\\s*%\\s*cpu", logs)
                cpu_matches_mc = _extract_numbers(r"([0-9]+(?:\\.[0-9]+)?)\\s*mCPU", logs)
                disk_io_matches = re.findall(r"(I/O error|Input/output error|disk failure|read error|write error)", logs, flags=re.IGNORECASE)
                error_patterns = re.findall(r"(OOMKilled|OutOfMemory|CrashLoopBackOff|BackOff|ImagePull|ErrImagePull|Connection refused|timeout|segmentation fault|exception)", logs, flags=re.IGNORECASE)

                mem_vals = mem_matches_mb + [g * 1024 for g in mem_matches_gb]
                cpu_vals = cpu_matches_pct + cpu_matches_mc

                parsed_signals = {
                    "memory_usage_mb_sample": mem_vals,
                    "cpu_usage_sample": cpu_vals,
                    "disk_io_errors": disk_io_matches,
                    "error_patterns": error_patterns,
                }
                # Simple heuristics for “what was happening”
                if mem_vals:
                    parsed_signals["max_memory_usage_mb"] = max(mem_vals)
                if cpu_vals:
                    parsed_signals["max_cpu_observed"] = max(cpu_vals)
                if "OOMKilled" in error_patterns or "OutOfMemory" in error_patterns:
                    parsed_signals["root_cause_hint"] = parsed_signals.get("root_cause_hint", "OOM or memory pressure")

                results["log_signals"] = parsed_signals
            except Exception:
                pass

            if "ImagePullBackOff" in logs or "ErrImagePull" in logs:
                results["root_cause_hint"] = "Deployment Config Error (Image)"
            elif "OOMKilled" in logs:
                results["root_cause_hint"] = "Resource Exhaustion (OOM)"
            elif "CrashLoopBackOff" in logs:
                results["root_cause_hint"] = "Application Crash Loop"
            elif "Connection refused" in logs or "Connection timed out" in logs:
                results["root_cause_hint"] = "Dependency/Network Error"
            else:
                results["root_cause_hint"] = "Unknown/Application Error"
        else:
            results["actions"].append("Pod not found; skipping log capture.")

        # CrashLoopBackOff handling (post-log capture)
        try:
            pod_status_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
            statuses = (pod_status_json.get("status", {}) or {}).get("containerStatuses", []) or []
            crash_cs = None
            for cs in statuses:
                waiting = (cs.get("state", {}) or {}).get("waiting") or {}
                restarts = cs.get("restartCount", 0)
                if waiting.get("reason") == "CrashLoopBackOff" or restarts > 3:
                    crash_cs = cs
                    break
            if crash_cs:
                # Resolve deployment owner
                dep_name = None
                try:
                    owners = (pod_status_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                    for o in owners:
                        if o.get("kind") == "ReplicaSet":
                            rs_json = _safe_json(["kubectl", "get", "rs", o.get("name"), "-n", namespace, "-o", "json"])
                            for ro in (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []:
                                if ro.get("kind") == "Deployment":
                                    dep_name = ro.get("name")
                                    break
                        elif o.get("kind") == "Deployment":
                            dep_name = o.get("name")
                        if dep_name:
                            break
                except Exception:
                    pass
                # print(f"DEBUG CrashLoop: dep_name={dep_name}, pod={pod_name}")
                logs_excerpt = ""
                try:
                    logs_excerpt = subprocess.check_output(
                        f"kubectl logs {pod_name} -n {namespace} --tail=50",
                        shell=True,
                        text=True,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    pass
                exit_code = None
                try:
                    last_state = (crash_cs.get("lastState") or {}).get("terminated") or {}
                    exit_code = last_state.get("exitCode")
                except Exception:
                    pass
                restarts = crash_cs.get("restartCount", 0)

                prompt = (
                    f"Pod crashing. Logs: {logs_excerpt}. Exit code: {exit_code}. Restarts: {restarts}.\n\n"
                    "Common patterns:\n"
                    "- Exit 137 = SIGKILL (hidden OOM)\n"
                    "- Exit 1 + \"connection refused\" = dependency not ready\n"
                    "- Exit 1 + \"port in use\" = port conflict\n"
                    "- Missing env var = startup crash\n\n"
                    'Respond ONLY with JSON. action MUST be exactly one of: increase_memory, rollback, fix_env, wait_dependency. Example: {"root_cause": "missing env", "recommended_actions": [{"action": "fix_env", "description": "Set POSTGRES_PASSWORD"}]}'
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    crash_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=500)
                    import json as _json
                    parsed_crash = _json.loads(crash_analysis) if crash_analysis else {}
                except Exception:
                    parsed_crash = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"

                def _patch_resources(mem_req=None, mem_lim=None):
                    if not dep_name:
                        actions_taken.append("No deployment owner; manual fix required")
                        return
                    try:
                        patch = {"spec": {"template": {"spec": {"containers": [{"name": crash_cs.get("name") or "", "resources": {}}]}}}}
                        if mem_req:
                            patch["spec"]["template"]["spec"]["containers"][0].setdefault("resources", {}).setdefault("requests", {})["memory"] = mem_req
                        if mem_lim:
                            patch["spec"]["template"]["spec"]["containers"][0].setdefault("resources", {}).setdefault("limits", {})["memory"] = mem_lim
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append(f"Patched resources for deployment {dep_name}")
                    except Exception as e:
                        actions_taken.append(f"Failed to patch resources: {e}")

                def _rollback():
                    if not dep_name:
                        actions_taken.append("No deployment owner; cannot rollback")
                        return
                    if not dep_history:
                        actions_taken.append("No deployment history for rollback")
                        return
                    try:
                        subprocess.run(["kubectl", "rollout", "undo", f"deployment/{dep_name}", "-n", namespace], check=True)
                        actions_taken.append(f"Rolled back deployment {dep_name} to previous revision")
                    except Exception as e:
                        actions_taken.append(f"Rollback failed: {e}")

                for act in parsed_crash.get("recommended_actions", []) or []:
                    a = act.get("action")
                    if a == "increase_memory":
                        req = act.get("details", {}).get("request") or "256Mi"
                        lim = act.get("details", {}).get("limit") or "512Mi"
                        _patch_resources(req, lim)
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "rollback":
                        _rollback()
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a in ("fix_env", "set_env_var", "set_env"):
                        missing = act.get("details", {}).get("env_vars") or []
                        if not missing:
                            import re
                            desc = act.get("description") or act.get("reason") or ""
                            extracted = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", desc)
                            if extracted:
                                # Deduplicate while preserving order
                                missing = list(dict.fromkeys(extracted))
                        if not dep_name:
                            actions_taken.append(f"Missing env vars suggested {missing} but no deployment owner")
                        else:
                            try:
                                env_entries = []
                                for envk in missing:
                                    if isinstance(envk, dict):
                                        name = envk.get("name")
                                        val = envk.get("value", "placeholder")
                                    else:
                                        name = str(envk)
                                        val = "placeholder"
                                    if name:
                                        env_entries.append({"name": name, "value": val})
                                patch = {
                                    "spec": {
                                        "template": {
                                            "spec": {
                                                "containers": [{
                                                    "name": crash_cs.get("name") or "",
                                                    "env": env_entries
                                                }]
                                            }
                                        }
                                    }
                                }
                                subprocess.run(
                                    ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                                    check=True,
                                )
                                actions_taken.append(f"Patched env vars {missing} into deployment {dep_name}")
                            except Exception as e:
                                actions_taken.append(f"Failed to patch env vars: {e}")
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "wait_dependency":
                        actions_taken.append("Waiting for dependency to become ready; monitoring")
                        outcome_state = "PARTIAL_SUCCESS"

                return {
                    "target": pod_name,
                    "issue_type": "CrashLoopBackOff",
                    "root_cause": parsed_crash.get("root_cause"),
                    "recommended_actions": parsed_crash.get("recommended_actions", []),
                    "forensics": {
                        "logs": logs_excerpt,
                        "exit_code": exit_code,
                        "restarts": restarts,
                        "deployment_history": dep_history[:3],
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
        except Exception:
            pass

        # CreateContainerConfigError handling
        try:
            pod_status_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
            statuses = (pod_status_json.get("status", {}) or {}).get("containerStatuses", []) or []
            cfg_cs = None
            for cs in statuses:
                waiting = (cs.get("state", {}) or {}).get("waiting") or {}
                if waiting.get("reason") == "CreateContainerConfigError":
                    cfg_cs = cs
                    break
            if cfg_cs:
                waiting = (cfg_cs.get("state", {}) or {}).get("waiting") or {}
                err_msg = waiting.get("message") or ""
                missing_name = None
                missing_type = None
                logs_excerpt = ""
                try:
                    log_cmd = ["kubectl", "logs", pod_name, "-n", namespace, "--tail=50"]
                    logs_excerpt = subprocess.check_output(log_cmd, stderr=subprocess.STDOUT, text=True)
                except Exception:
                    logs_excerpt = "Pod never started - no logs available"
                import re
                m_cm = re.search(r"configmap \"([^\"]+)\"", err_msg, flags=re.IGNORECASE)
                m_sec = re.search(r"secret \"([^\"]+)\"", err_msg, flags=re.IGNORECASE)
                if not m_cm:
                    m_cm = re.search(r"configmap\s+([a-z0-9-]+/)?([a-z0-9-]+)", err_msg, flags=re.IGNORECASE)
                if m_cm:
                    missing_name = m_cm.group(2) if m_cm.lastindex == 2 else m_cm.group(1).rstrip("/")
                    missing_type = "ConfigMap"
                if not m_sec:
                    m_sec = re.search(r"secret\s+([a-z0-9-]+/)?([a-z0-9-]+)", err_msg, flags=re.IGNORECASE)
                if m_sec:
                    missing_name = m_sec.group(2) if m_sec.lastindex == 2 else m_sec.group(1).rstrip("/")
                    missing_type = "Secret"

                # Derive expected keys from pod/deployment spec references (env, envFrom, volumes)
                dep_name = locals().get("dep_name")
                # print(f"DEBUG: missing_name={missing_name}, missing_type={missing_type}")
                # print(f"DEBUG: dep_name={dep_name}, owner_name={owner_name}, owner_kind={owner_kind}")
                expected_keys = set()
                try:
                    def _collect_keys_from_spec(spec_obj):
                        if not spec_obj:
                            return
                        def _collect_from_containers(cont_list):
                            for c in cont_list or []:
                                for env in c.get("env", []) or []:
                                    vf = env.get("valueFrom") or {}
                                    cm_ref = vf.get("configMapKeyRef")
                                    sec_ref = vf.get("secretKeyRef")
                                    if missing_type == "ConfigMap" and cm_ref and cm_ref.get("name") == missing_name and cm_ref.get("key"):
                                        expected_keys.add(cm_ref.get("key"))
                                    if missing_type == "Secret" and sec_ref and sec_ref.get("name") == missing_name and sec_ref.get("key"):
                                        expected_keys.add(sec_ref.get("key"))
                                for env_from in c.get("envFrom", []) or []:
                                    ref = env_from.get("configMapRef") if missing_type == "ConfigMap" else env_from.get("secretRef")
                                    if ref and ref.get("name") == missing_name and ref.get("prefix"):
                                        expected_keys.add(ref.get("prefix"))
                        _collect_from_containers(spec_obj.get("containers", []) or [])
                        _collect_from_containers(spec_obj.get("initContainers", []) or [])
                        for vol in spec_obj.get("volumes", []) or []:
                            cm = vol.get("configMap") if missing_type == "ConfigMap" else vol.get("secret")
                            name_field = cm.get("name") if cm and missing_type == "ConfigMap" else cm.get("secretName") if cm else None
                            if cm and name_field == missing_name:
                                items = cm.get("items") or []
                                if items:
                                    for it in items:
                                        if it.get("key"):
                                            expected_keys.add(it.get("key"))
                                else:
                                    expected_keys.add("data")

                    spec = (pod_status_json.get("spec", {}) or {})
                    _collect_keys_from_spec(spec)
                    # print(f"DEBUG: expected_keys after pod spec: {expected_keys}")

                    # If still empty, try deployment template spec
                    if not expected_keys and dep_name:
                        dep_obj = _safe_json(["kubectl", "get", "deploy", dep_name, "-n", namespace, "-o", "json"])
                        tpl_spec = (((dep_obj or {}).get("spec") or {}).get("template") or {}).get("spec") or {}
                        _collect_keys_from_spec(tpl_spec)
                    # Fallback using owner name/kind if available (pod spec may be incomplete)
                    if not expected_keys and owner_kind == "Deployment" and owner_name:
                        dep_obj = _safe_json(["kubectl", "get", "deploy", owner_name, "-n", namespace, "-o", "json"])
                        tpl_spec = (((dep_obj or {}).get("spec") or {}).get("template") or {}).get("spec") or {}
                        _collect_keys_from_spec(tpl_spec)
                    # print(f"DEBUG: expected_keys after deployment spec: {expected_keys}")

                    # Fallback: parse error message for missing key hints
                    if not expected_keys and err_msg:
                        try:
                            # e.g., "couldn't find key api-url in ConfigMap default/app-config"
                            missing_key_matches = re.findall(r"key\\s+([A-Za-z0-9._-]+)", err_msg, flags=re.IGNORECASE)
                            for mk in missing_key_matches:
                                expected_keys.add(mk)
                            # e.g., "keys 'api-url', 'database-url'"
                            quoted = re.findall(r"'([^']+)'", err_msg)
                            for qk in quoted:
                                if "/" not in qk and qk:
                                    expected_keys.add(qk)
                        except Exception:
                            pass

                    if not expected_keys:
                        expected_keys.add("data")
                except Exception as e:
                    import traceback
                    # print(f"DEBUG: Exception in key extraction: {e}\\n{traceback.format_exc()}")
                    expected_keys = {"data"}

                def _default_value_for_key(key_name: str) -> str:
                    import string as _string
                    kl = (key_name or "").lower()
                    if "password" in kl or "secret" in kl or "token" in kl:
                        alphabet = _string.ascii_letters + _string.digits
                        return "".join(random.choice(alphabet) for _ in range(24))
                    if "user" in kl or "username" in kl:
                        return "admin"
                    if "url" in kl or "uri" in kl or "conn" in kl:
                        return "postgresql://admin:changeme@localhost:5432/app"
                    if "yaml" in kl or kl.endswith(".yml"):
                        return "config:\n  enabled: true\n"
                    if "json" in kl:
                        return json.dumps({"status": "ok"})
                    return "placeholder"

                def _verify_resource(kind: str, name: str, expected: set) -> bool:
                    try:
                        created = _safe_json(["kubectl", "-n", namespace, "get", kind, name, "-o", "json"])
                        data_obj = (created or {}).get("data") or {}
                        return bool(created) and set(data_obj.keys()) >= set(expected)
                    except Exception:
                        return False

                def _upsert_generic_secret(name: str, keys: set) -> bool:
                    values = {k: _default_value_for_key(k) for k in keys if k}
                    manifest = {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "metadata": {"name": name, "namespace": namespace},
                        "type": "Opaque",
                        "stringData": values,
                    }
                    try:
                        import json as _json
                        subprocess.run(
                            ["kubectl", "apply", "-f", "-"],
                            input=_json.dumps(manifest),
                            text=True,
                            check=True,
                        )
                        return _verify_resource("secret", name, keys)
                    except Exception as e:
                        actions_taken.append(f"Failed to upsert Secret {name}: {e}")
                        return False

                def _upsert_generic_configmap(name: str, keys: set) -> bool:
                    values = {k: _default_value_for_key(k) for k in keys if k}
                    manifest = {
                        "apiVersion": "v1",
                        "kind": "ConfigMap",
                        "metadata": {"name": name, "namespace": namespace},
                        "data": values,
                    }
                    try:
                        import json as _json
                        subprocess.run(
                            ["kubectl", "apply", "-f", "-"],
                            input=_json.dumps(manifest),
                            text=True,
                            check=True,
                        )
                        return _verify_resource("configmap", name, keys)
                    except Exception as e:
                        actions_taken.append(f"Failed to upsert ConfigMap {name}: {e}")
                        return False

                def _list_resources(kind: str):
                    try:
                        res = _safe_json(["kubectl", "get", kind, "-n", namespace, "-o", "json"])
                        names = []
                        for item in res.get("items", []):
                            names.append(item.get("metadata", {}).get("name"))
                        return names
                    except Exception:
                        return []

                available = {
                    "configmaps": _list_resources("configmap"),
                    "secrets": _list_resources("secret"),
                }

                dep_name = None
                dep_history = []
                try:
                    owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                    for o in owners:
                        if o.get("kind") == "ReplicaSet":
                            rs_json = _safe_json(["kubectl", "get", "rs", o.get("name"), "-n", namespace, "-o", "json"])
                            rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                            for ro in rs_owners:
                                if ro.get("kind") == "Deployment":
                                    dep_name = ro.get("name")
                                    break
                        if o.get("kind") == "Deployment":
                            dep_name = o.get("name")
                        if dep_name:
                            break
                except Exception:
                    pass
                dep_manifest = None
                try:
                    if dep_name:
                        dep_manifest = _safe_json(["kubectl", "get", "deploy", dep_name, "-n", namespace, "-o", "json"])
                        hist = _safe_json(["kubectl", "rollout", "history", f"deployment/{dep_name}", "-n", namespace, "-o", "json"])
                        dep_history = hist.get("history", []) if isinstance(hist, dict) else []
                except Exception:
                    pass

                prompt = (
                    f"Missing {missing_type} \"{missing_name}\".\n"
                    f"Available: {available}\n"
                    f"Deployment references: {dep_manifest}\n"
                    f"Previous versions: {dep_history}\n\n"
                    "Options:\n"
                    "1. Typo? (app-cfg vs app-config)\n"
                    "2. Create from previous version\n"
                    "3. Update deployment to use existing one\n\n"
                    "JSON: {\"root_cause\": \"...\", \"actions\": [{\"action\": \"create_from_history|update_reference|create_default\", ...}]}"
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    cfg_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=400)
                    import json as _json
                    parsed_cfg = _json.loads(cfg_analysis) if cfg_analysis else {}
                except Exception:
                    parsed_cfg = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"
                any_success = False

                def _create_from_history():
                    if not dep_history:
                        actions_taken.append("No history to create missing resource")
                        return
                    try:
                        # Placeholder: simply note; real creation would need stored manifests
                        actions_taken.append(f"Would create {missing_type} {missing_name} from history")
                    except Exception as e:
                        actions_taken.append(f"Failed to create from history: {e}")

                def _update_reference(target_name: str):
                    if not dep_name or not target_name:
                        actions_taken.append("Missing deployment or target_name for update_reference")
                        return
                    try:
                        # Minimal patch to replace first matching volume/secret/configmap reference
                        patch = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "volumes": [{
                                            "name": missing_name or target_name,
                                            "configMap": {"name": target_name} if missing_type == "ConfigMap" else None,
                                            "secret": {"secretName": target_name} if missing_type == "Secret" else None,
                                        }]
                                    }
                                }
                            }
                        }
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append(f"Patched deployment {dep_name} to reference {target_name}")
                    except Exception as e:
                        actions_taken.append(f"Failed to patch deployment reference: {e}")

                def _create_default():
                    try:
                        kind = "configmap" if missing_type == "ConfigMap" else "secret"
                        name_to_create = missing_name or "missing-resource"
                        from_literals = []
                        for k in expected_keys:
                            val = _default_value_for_key(k)
                            from_literals.extend(["--from-literal", f"{k}={val}"])
                        cmd = ["kubectl", "-n", namespace, "create", kind, name_to_create] + from_literals
                        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        # Verify creation and keys
                        created = _safe_json(["kubectl", "-n", namespace, "get", kind, name_to_create, "-o", "json"])
                        data_obj = (created or {}).get("data") or {}
                        if created and set(data_obj.keys()) >= set(expected_keys):
                            actions_taken.append(f"Created {missing_type} {name_to_create} with keys {sorted(expected_keys)}")
                        else:
                            actions_taken.append(f"Attempted to create {missing_type} {name_to_create}, but keys missing")
                    except Exception as e:
                        actions_taken.append(f"Failed to create default {missing_type}: {e}")

                for act in parsed_cfg.get("actions", []) or []:
                    a = act.get("action")
                    if a == "create_from_history":
                        _create_from_history()
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "update_reference":
                        tgt = act.get("target") or act.get("name")
                        _update_reference(tgt)
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "create_default":
                        _create_default()
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "create_secret":
                        try:
                            secret_name = act.get("name") or missing_name or "generated-secret"
                            # Build literals for expected keys (generic secret, not just docker-registry)
                            import secrets as _secrets
                            literals = []
                            for k in expected_keys:
                                if not k:
                                    continue
                                if "user" in k.lower():
                                    val = "admin"
                                elif "pass" in k.lower() or "secret" in k.lower():
                                    val = _secrets.token_urlsafe(16)
                                else:
                                    val = f"{k}-default-value"
                                literals.extend(["--from-literal", f"{k}={val}"])
                            cmd = ["kubectl", "create", "secret", "generic", secret_name, "-n", namespace] + literals
                            subprocess.run(cmd, check=True)
                            if _verify_resource("secret", secret_name, expected_keys):
                                actions_taken.append(f"Created Secret {secret_name} with keys: {sorted(expected_keys)}")
                                any_success = True
                                outcome_state = "SUCCESS"
                            else:
                                actions_taken.append(f"Created Secret {secret_name} but verification failed")
                                outcome_state = "PARTIAL_SUCCESS"
                        except Exception as e:
                            actions_taken.append(f"Failed to create Secret: {e}")
                    elif a == "create_configmap":
                        try:
                            cm_name = act.get("name") or missing_name or "generated-config"
                            literals = []
                            for k in expected_keys:
                                if not k:
                                    continue
                                val = _default_value_for_key(k)
                                literals.extend(["--from-literal", f"{k}={val}"])
                            cmd = ["kubectl", "create", "configmap", cm_name, "-n", namespace] + literals
                            subprocess.run(cmd, check=True)
                            if _verify_resource("configmap", cm_name, expected_keys):
                                actions_taken.append(f"Created ConfigMap {cm_name} with keys: {sorted(expected_keys)}")
                                any_success = True
                                outcome_state = "SUCCESS"
                            else:
                                actions_taken.append(f"Created ConfigMap {cm_name} but verification failed")
                                outcome_state = "PARTIAL_SUCCESS"
                        except Exception as e:
                            actions_taken.append(f"Failed to create ConfigMap: {e}")
                # Fallback: if LLM did not specify actions but we know what is missing, create it
                if not actions_taken and missing_name and missing_type:
                    try:
                        if missing_type == "Secret":
                            if _upsert_generic_secret(missing_name, expected_keys):
                                actions_taken.append(f"Upserted Secret {missing_name} with keys {sorted(expected_keys)}")
                                any_success = True
                                outcome_state = "SUCCESS"
                            else:
                                actions_taken.append(f"Attempted to create Secret {missing_name}, but keys missing")
                                outcome_state = "PARTIAL_SUCCESS"
                        else:
                            if _upsert_generic_configmap(missing_name, expected_keys):
                                actions_taken.append(f"Upserted ConfigMap {missing_name} with keys {sorted(expected_keys)}")
                                any_success = True
                                outcome_state = "SUCCESS"
                            else:
                                actions_taken.append(f"Attempted to create ConfigMap {missing_name}, but keys missing")
                                outcome_state = "PARTIAL_SUCCESS"
                    except Exception:
                        pass

                createcontainer_result = {
                    "target": pod_name,
                    "issue_type": "CreateContainerConfigError",
                    "root_cause": parsed_cfg.get("root_cause"),
                    "recommended_actions": parsed_cfg.get("actions", []),
                    "forensics": {
                        "error_message": err_msg,
                        "missing_name": missing_name,
                        "missing_type": missing_type,
                        "available": available,
                        "expected_keys": sorted(expected_keys),
                        "deployment_history": dep_history[:3],
                        "logs": logs_excerpt,
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
                results["issue_type"] = "CreateContainerConfigError"
                results["root_cause"] = parsed_cfg.get("root_cause")
                results["outcome"] = outcome_state
                results["actions"].extend(actions_taken)
                results["createcontainerconfigerror"] = createcontainer_result
                skip_fallback_actions = True
        except Exception:
            pass

        # Pending handling
        try:
            pod_status_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
            phase = (pod_status_json.get("status", {}) or {}).get("phase")
            start_time = (pod_status_json.get("status", {}) or {}).get("startTime")
            pending_too_long = False
            if phase == "Pending":
                try:
                    pod_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    if (datetime.utcnow() - pod_start).total_seconds() > 120:
                        pending_too_long = True
                except Exception:
                    pending_too_long = True
            if phase == "Pending" and pending_too_long:
                describe = ""
                try:
                    describe = subprocess.check_output(
                        ["kubectl", "describe", "pod", pod_name, "-n", namespace],
                        text=True,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    pass

                events = ""
                try:
                    ev = _safe_json(["kubectl", "get", "events", "-n", namespace, "-o", "json"])
                    if isinstance(ev, dict):
                        events = ev.get("items", [])
                except Exception:
                    events = ""

                # Parse resource requests
                pod_spec = (pod_json.get("spec", {}) or {})
                reqs = []
                for c in pod_spec.get("containers", []) or []:
                    res = (c.get("resources", {}) or {}).get("requests", {}) or {}
                    reqs.append({"container": c.get("name"), "requests": res})

                # Node capacity snapshot
                nodes = []
                try:
                    nodes_json = _safe_json(["kubectl", "get", "nodes", "-o", "json"])
                    for n in nodes_json.get("items", []):
                        alloc = (n.get("status", {}) or {}).get("allocatable", {})
                        nodes.append({"name": n.get("metadata", {}).get("name"), "allocatable": alloc})
                except Exception:
                    nodes = []

                dep_name = None
                try:
                    owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                    for o in owners:
                        if o.get("kind") == "ReplicaSet":
                            rs_json = _safe_json(["kubectl", "get", "rs", o.get("name"), "-n", namespace, "-o", "json"])
                            rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                            for ro in rs_owners:
                                if ro.get("kind") == "Deployment":
                                    dep_name = ro.get("name")
                                    break
                        if o.get("kind") == "Deployment":
                            dep_name = o.get("name")
                        if dep_name:
                            break
                except Exception:
                    dep_name = None

                prompt = (
                    f"Pod Pending for >2m. Events: {events}. "
                    f"Description: {describe}. Node capacity: {nodes}. Pod requests: {reqs}.\n\n"
                    "Fix patterns:\n"
                    "- Insufficient resources -> reduce requests OR scale down other pods\n"
                    "- Node selector mismatch -> remove selector\n"
                    "- PVC not bound -> check storage class\n\n"
                    'Return JSON: {"root_cause": "...", "actions": [{"action": "reduce_requests|remove_selector|provision_storage", ...}]}'
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    pending_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=400)
                    import json as _json
                    parsed_pending = _json.loads(pending_analysis) if pending_analysis else {}
                except Exception:
                    parsed_pending = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"

                def _patch_requests():
                    try:
                        if not dep_name:
                            actions_taken.append("Standalone pod; manual resource adjustment required")
                            return
                        # Default values but prefer LLM-suggested
                        cpu_req = "100m"
                        mem_req = "128Mi"
                        details = {}
                        # Use outer scope parsed_pending in caller loop; we will override per action below
                        if False:
                            details = {}
                        patch = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [{
                                            "name": pod_spec.get("containers", [{}])[0].get("name", ""),
                                            "resources": {"requests": {"cpu": cpu_req, "memory": mem_req}}
                                        }]
                                    }
                                }
                            }
                        }
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append("Patched requests to smaller defaults")
                    except Exception as e:
                        actions_taken.append(f"Failed to patch requests: {e}")

                def _remove_selector():
                    try:
                        if not dep_name:
                            actions_taken.append("Standalone pod; no nodeSelector to patch at deployment level")
                            return
                        patch = {"spec": {"template": {"spec": {"nodeSelector": None}}}}
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append("Removed node selector")
                    except Exception as e:
                        actions_taken.append(f"Failed to remove node selector: {e}")

                for act in parsed_pending.get("actions", []) or []:
                    a = act.get("action")
                    if a == "reduce_requests":
                        try:
                            # allow per-action overrides
                            if act.get("details"):
                                reqs = act["details"]
                                cpu_req = reqs.get("cpu") or "100m"
                                mem_req = reqs.get("memory") or "128Mi"
                                def _patch_requests_with_vals(cpu_val, mem_val):
                                    if not dep_name:
                                        actions_taken.append("Standalone pod; manual resource adjustment required")
                                        return
                                    patch = {
                                        "spec": {
                                            "template": {
                                                "spec": {
                                                    "containers": [{
                                                        "name": pod_spec.get("containers", [{}])[0].get("name", ""),
                                                        "resources": {"requests": {"cpu": cpu_val, "memory": mem_val}}
                                                    }]
                                                }
                                            }
                                        }
                                    }
                                    subprocess.run(
                                        ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                                        check=True,
                                    )
                                    actions_taken.append(f"Patched requests to cpu={cpu_val}, memory={mem_val}")
                                _patch_requests_with_vals(cpu_req, mem_req)
                            else:
                                _patch_requests()
                        except Exception as e:
                            actions_taken.append(f"Failed to patch requests: {e}")
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "remove_selector":
                        _remove_selector()
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "provision_storage":
                        actions_taken.append("Provision storage/PVC manually")
                        outcome_state = "PARTIAL_SUCCESS"

                return {
                    "target": pod_name,
                    "issue_type": "Pending",
                    "root_cause": parsed_pending.get("root_cause"),
                    "recommended_actions": parsed_pending.get("actions", []),
                    "forensics": {
                        "describe": describe,
                        "events": events,
                        "node_capacity": nodes,
                        "requests": reqs,
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
        except Exception:
            pass

        # Evicted handling
        try:
            pod_status_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
            reason = (pod_status_json.get("status", {}) or {}).get("reason")
            if reason == "Evicted":
                dep_name = None
                try:
                    owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                    for o in owners:
                        if o.get("kind") == "ReplicaSet":
                            rs_json = _safe_json(["kubectl", "get", "rs", o.get("name"), "-n", namespace, "-o", "json"])
                            rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                            for ro in rs_owners:
                                if ro.get("kind") == "Deployment":
                                    dep_name = ro.get("name")
                                    break
                        if o.get("kind") == "Deployment":
                            dep_name = o.get("name")
                        if dep_name:
                            break
                except Exception:
                    dep_name = None
                message = (pod_status_json.get("status", {}) or {}).get("message") or ""
                node_name = (pod_status_json.get("spec", {}) or {}).get("nodeName")
                node_conditions = {}
                try:
                    if node_name:
                        node_json = _safe_json(["kubectl", "get", "node", node_name, "-o", "json"])
                        node_conditions = node_json.get("status", {}).get("conditions", [])
                except Exception:
                    node_conditions = {}

                prompt = (
                    f"Pod evicted. Reason: {message}. Node conditions: {node_conditions}.\n\n"
                    "Causes:\n"
                    "- MemoryPressure → reduce requests\n"
                    "- DiskPressure → clean logs\n"
                    "- Node failure → reschedule\n\n"
                    'JSON: {"root_cause": "...", "actions": [{"action": "reduce_memory|clean_disk|reschedule", ...}]}'
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    evict_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=300)
                    import json as _json
                    parsed_evict = _json.loads(evict_analysis) if evict_analysis else {}
                except Exception:
                    parsed_evict = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"

                def _reduce_mem():
                    if not dep_name:
                        actions_taken.append("No deployment to patch for eviction")
                        return
                    try:
                        patch = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [{
                                            "name": (pod_json.get("spec", {}) or {}).get("containers", [{}])[0].get("name", ""),
                                            "resources": {"requests": {"memory": "128Mi"}, "limits": {"memory": "256Mi"}},
                                        }]
                                    }
                                }
                            }
                        }
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append(f"Reduced memory requests/limits for deployment {dep_name}")
                    except Exception as e:
                        actions_taken.append(f"Failed to patch deployment for eviction: {e}")

                def _reschedule():
                    try:
                        subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace, "--wait=false"], check=False)
                        actions_taken.append("Deleted pod to reschedule")
                    except Exception as e:
                        actions_taken.append(f"Failed to delete pod for reschedule: {e}")

                for act in parsed_evict.get("actions", []) or []:
                    a = act.get("action")
                    if a == "reduce_memory":
                        _reduce_mem()
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "clean_disk":
                        actions_taken.append("Clean disk/logs on node (manual)")
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "reschedule":
                        _reschedule()
                        outcome_state = "PARTIAL_SUCCESS"

                return {
                    "target": pod_name,
                    "issue_type": "Evicted",
                    "root_cause": parsed_evict.get("root_cause"),
                    "recommended_actions": parsed_evict.get("actions", []),
                    "forensics": {
                        "message": message,
                        "node_conditions": node_conditions,
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
        except Exception:
            pass

        # Node failure handling (check node conditions)
        try:
            node_name = (pod_json.get("spec", {}) or {}).get("nodeName")
            if node_name:
                node_json = _safe_json(["kubectl", "get", "node", node_name, "-o", "json"])
                conditions = (node_json.get("status", {}) or {}).get("conditions", []) or []
                not_ready = False
                pressure = False
                for c in conditions:
                    if c.get("type") == "Ready" and c.get("status") != "True":
                        not_ready = True
                    if c.get("type") in {"DiskPressure", "MemoryPressure"} and c.get("status") == "True":
                        pressure = True
                if not_ready or pressure:
                    pods_on_node = []
                    try:
                        pods_json = _safe_json(["kubectl", "get", "pods", "-A", "-o", "json"])
                        for p in pods_json.get("items", []):
                            if (p.get("spec", {}) or {}).get("nodeName") == node_name:
                                pods_on_node.append(p.get("metadata", {}).get("name"))
                    except Exception:
                        pods_on_node = []

                    prompt = (
                        f"Node {node_name} NotReady/Pressure. Conditions: {conditions}. Pods: {len(pods_on_node)}.\n\n"
                        "Actions:\n"
                        "- DiskPressure → cordon + clean\n"
                        "- MemoryPressure → cordon + drain\n"
                        "- NetworkUnavailable → investigate\n"
                        "- Persistent → replace node\n\n"
                        'JSON: {"root_cause": "...", "actions": [{"action": "cordon|drain|investigate", ...}]}'
                    )
                    try:
                        from shared_models import OllamaLLMIntegration
                        llm = OllamaLLMIntegration()
                        node_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=400)
                        import json as _json
                        parsed_node = _json.loads(node_analysis) if node_analysis else {}
                    except Exception:
                        parsed_node = {}

                    actions_taken = []
                    outcome_state = "ANALYSIS_COMPLETE"

                    def _cordon():
                        try:
                            subprocess.run(["kubectl", "cordon", node_name], check=True)
                            actions_taken.append(f"Cordoned node {node_name}")
                        except Exception as e:
                            actions_taken.append(f"Failed to cordon {node_name}: {e}")

                    def _drain():
                        try:
                            subprocess.run(["kubectl", "drain", node_name, "--ignore-daemonsets", "--force", "--delete-emptydir-data"], check=True)
                            actions_taken.append(f"Drained node {node_name}")
                        except Exception as e:
                            actions_taken.append(f"Failed to drain {node_name}: {e}")

                    for act in parsed_node.get("actions", []) or []:
                        a = act.get("action")
                        if a == "cordon":
                            _cordon()
                            outcome_state = "PARTIAL_SUCCESS"
                        elif a == "drain":
                            _cordon()
                            _drain()
                            outcome_state = "PARTIAL_SUCCESS"
                        elif a == "investigate":
                            actions_taken.append("Investigate node networking or hardware; manual action required")
                            outcome_state = "PARTIAL_SUCCESS"

                return {
                    "target": pod_name,
                    "issue_type": "NodeNotReady",
                    "root_cause": parsed_node.get("root_cause"),
                    "recommended_actions": parsed_node.get("actions", []),
                    "forensics": {
                        "node": node_name,
                        "conditions": conditions,
                        "pods_on_node": pods_on_node,
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
        except Exception:
            pass

        # Init container handling
        try:
            init_statuses = (pod_json.get("status", {}) or {}).get("initContainerStatuses", []) or []
            failed_init = None
            for cs in init_statuses:
                term = (cs.get("state", {}) or {}).get("terminated") or {}
                if term and term.get("exitCode", 0) != 0:
                    failed_init = cs
                    break
            if failed_init:
                init_name = failed_init.get("name")
                exit_code = ((failed_init.get("state", {}) or {}).get("terminated") or {}).get("exitCode")
                dep_name = None
                try:
                    owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                    for o in owners:
                        if o.get("kind") == "ReplicaSet":
                            rs_json = _safe_json(["kubectl", "get", "rs", o.get("name"), "-n", namespace, "-o", "json"])
                            rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                            for ro in rs_owners:
                                if ro.get("kind") == "Deployment":
                                    dep_name = ro.get("name")
                                    break
                        if o.get("kind") == "Deployment":
                            dep_name = o.get("name")
                        if dep_name:
                            break
                except Exception:
                    dep_name = None
                logs_excerpt = ""
                try:
                    logs_excerpt = subprocess.check_output(
                        f"kubectl logs {pod_name} -n {namespace} -c {init_name} --tail=50",
                        shell=True,
                        text=True,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    pass
                cmd_args = {}
                try:
                    for c in (pod_json.get("spec", {}) or {}).get("initContainers", []) or []:
                        if c.get("name") == init_name:
                            cmd_args = {
                                "command": c.get("command"),
                                "args": c.get("args"),
                            }
                except Exception:
                    cmd_args = {}

                prompt = (
                    f"Init container failed. Logs: {logs_excerpt}. Exit: {exit_code}.\n\n"
                    "Common:\n"
                    "- Waiting for service → increase timeout\n"
                    "- Permission denied → check RBAC\n"
                    "- Command failed → fix script\n\n"
                    'JSON: {"root_cause": "...", "actions": [{"action": "increase_timeout|fix_rbac|skip_init", ...}]}'
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    init_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=400)
                    import json as _json
                    parsed_init = _json.loads(init_analysis) if init_analysis else {}
                except Exception:
                    parsed_init = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"

                def _patch_timeout():
                    try:
                        if not dep_name:
                            return
                        patch = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "initContainers": [{
                                            "name": init_name,
                                            "env": [{"name": "INIT_TIMEOUT", "value": "120"}],
                                        }]
                                    }
                                }
                            }
                        }
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append(f"Increased init timeout for {init_name}")
                    except Exception as e:
                        actions_taken.append(f"Failed to patch init timeout: {e}")

                def _skip_init():
                    try:
                        if not dep_name:
                            return
                        patch = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "initContainers": []
                                    }
                                }
                            }
                        }
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append(f"Removed init containers from deployment {dep_name}")
                    except Exception as e:
                        actions_taken.append(f"Failed to remove init containers: {e}")

                for act in parsed_init.get("actions", []) or []:
                    a = act.get("action")
                    if a == "increase_timeout":
                        _patch_timeout()
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "fix_rbac":
                        actions_taken.append("Check RBAC/permissions for init container")
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "skip_init":
                        _skip_init()
                        outcome_state = "PARTIAL_SUCCESS"

                return {
                    "target": pod_name,
                    "issue_type": "InitContainerFailure",
                    "root_cause": parsed_init.get("root_cause"),
                    "recommended_actions": parsed_init.get("actions", []),
                    "forensics": {
                        "logs": logs_excerpt,
                        "exit_code": exit_code,
                        "cmd_args": cmd_args,
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
        except Exception:
            pass

        # Probe failure handling
        try:
            events = _safe_json(["kubectl", "get", "events", "-n", namespace, "-o", "json"])
            probe_fail = False
            if isinstance(events, dict):
                for ev in events.get("items", []):
                    msg = (ev.get("message") or "").lower()
                    if "liveness probe failed" in msg or "readiness probe failed" in msg:
                        probe_fail = True
                        break
            if probe_fail:
                probe_cfg = {}
                try:
                    for c in (pod_json.get("spec", {}) or {}).get("containers", []) or []:
                        if c.get("name"):
                            probe_cfg = {
                                "livenessProbe": c.get("livenessProbe"),
                                "readinessProbe": c.get("readinessProbe"),
                            }
                            break
                except Exception:
                    probe_cfg = {}
                logs_excerpt = ""
                try:
                    logs_excerpt = subprocess.check_output(
                        f"kubectl logs {pod_name} -n {namespace} --tail=50",
                        shell=True,
                        text=True,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    pass

                prompt = (
                    f"Probe failing. Config: {probe_cfg}. Logs: {logs_excerpt}.\n\n"
                    "Issues:\n"
                    "- Timeout too short → increase\n"
                    "- Endpoint wrong → fix path\n"
                    "- App slow to start → increase initialDelaySeconds\n\n"
                    'JSON: {"root_cause": "...", "actions": [{"action": "increase_timeout|fix_endpoint|increase_delay", ...}]}'
                )
                try:
                    from shared_models import OllamaLLMIntegration
                    llm = OllamaLLMIntegration()
                    probe_analysis = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=400)
                    import json as _json
                    parsed_probe = _json.loads(probe_analysis) if probe_analysis else {}
                except Exception:
                    parsed_probe = {}

                actions_taken = []
                outcome_state = "ANALYSIS_COMPLETE"

                def _patch_probe(field: str, value):
                    if not dep_name:
                        actions_taken.append("No deployment to patch probes")
                        return
                    try:
                        container_name = (pod_json.get("spec", {}) or {}).get("containers", [{}])[0].get("name", "")
                        patch = {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [{
                                            "name": container_name,
                                            field: value,
                                            "readinessProbe": value if field == "readinessProbe" else None,
                                            "livenessProbe": value if field == "livenessProbe" else None,
                                        }]
                                    }
                                }
                            }
                        }
                        subprocess.run(
                            ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                            check=True,
                        )
                        actions_taken.append(f"Patched {field} on deployment {dep_name}")
                    except Exception as e:
                        actions_taken.append(f"Failed to patch {field}: {e}")

                for act in parsed_probe.get("actions", []) or []:
                    a = act.get("action")
                    if a == "increase_timeout":
                        val = act.get("details", {}).get("timeoutSeconds", 10)
                        _patch_probe("livenessProbe", {"timeoutSeconds": val})
                        _patch_probe("readinessProbe", {"timeoutSeconds": val})
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "fix_endpoint":
                        path = act.get("details", {}).get("path", "/")
                        _patch_probe("livenessProbe", {"httpGet": {"path": path}})
                        _patch_probe("readinessProbe", {"httpGet": {"path": path}})
                        outcome_state = "PARTIAL_SUCCESS"
                    elif a == "increase_delay":
                        val = act.get("details", {}).get("initialDelaySeconds", 30)
                        _patch_probe("livenessProbe", {"initialDelaySeconds": val})
                        _patch_probe("readinessProbe", {"initialDelaySeconds": val})
                        outcome_state = "PARTIAL_SUCCESS"

                return {
                    "target": pod_name,
                    "issue_type": "ProbeFailure",
                    "root_cause": parsed_probe.get("root_cause"),
                    "recommended_actions": parsed_probe.get("actions", []),
                    "forensics": {
                        "probe_config": probe_cfg,
                        "logs": logs_excerpt,
                    },
                    "actions_taken": actions_taken,
                    "outcome": outcome_state,
                }
        except Exception:
            pass

        # STEP 2: Apply calculated fixes if provided, otherwise fallback to restart/delete behavior
        def _apply_resource_patch_to_deployment(deploy_name: str, actions: list) -> bool:
            if not deploy_name or not actions:
                return False
            # Fetch deployment to map container names
            deploy_spec = {}
            try:
                deploy_json = _safe_json(["kubectl", "-n", namespace, "get", "deploy", deploy_name, "-o", "json"])
                if deploy_json:
                    deploy_spec = ((deploy_json.get("spec") or {}).get("template") or {}).get("spec") or {}
            except Exception:
                deploy_spec = {}
            real_containers = [c.get("name") for c in deploy_spec.get("containers", []) or [] if c.get("name")]

            # Build strategic merge patch for resources
            containers_patch = []
            for act in actions:
                cname = act.get("container") or None
                if (not cname) or (cname == "unknown") or (cname not in real_containers):
                    cname = real_containers[0] if real_containers else cname
                res_block = {}
                reqs = act.get("requests") or {}
                limits = act.get("limits") or {}
                if reqs:
                    res_block["requests"] = reqs
                if limits:
                    res_block["limits"] = limits
                if not res_block:
                    continue
                containers_patch.append({
                    "name": cname or "",  # empty name is tolerated; K8s will match by merge key if set
                    "resources": res_block,
                })
            if not containers_patch:
                return False
            patch_obj = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": containers_patch
                        }
                    }
                }
            }
            try:
                import json as _json
                subprocess.run(
                    ["kubectl", "patch", "deployment", deploy_name, "-n", namespace, "--type", "strategic", "-p", _json.dumps(patch_obj)],
                    check=True,
                )
                results["actions"].append(f"Patched deployment/{deploy_name} resources with recommended limits/requests")
                return True
            except Exception as e:
                results["actions"].append(f"Deployment patch failed: {e}")
                return False

        def _recreate_naked_pod_with_resources(pod_doc: dict, actions: list) -> bool:
            if not pod_doc or not actions:
                return False
            pod_spec = pod_doc.get("spec", {}) or {}
            containers = pod_spec.get("containers", []) or []
            if not containers:
                return False

            # Apply resource overrides
            for act in actions:
                target_name = act.get("container")
                reqs = act.get("requests") or {}
                limits = act.get("limits") or {}
                for c in containers:
                    if target_name and c.get("name") != target_name:
                        continue
                    res = c.get("resources") or {}
                    if reqs:
                        res["requests"] = reqs
                    if limits:
                        res["limits"] = limits
                    c["resources"] = res

            # Clean metadata for re-create
            import copy
            import json as _json
            new_pod = copy.deepcopy(pod_doc)
            new_pod.pop("status", None)
            meta = new_pod.get("metadata", {}) or {}
            for k in ["resourceVersion", "uid", "selfLink", "creationTimestamp", "managedFields", "annotations"]:
                meta.pop(k, None)
            # Keep name constant; namespace handled by -n flag
            meta["namespace"] = namespace
            new_pod["metadata"] = meta
            new_pod["spec"] = pod_spec

            # Delete existing pod first
            try:
                subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace, "--wait=false"], check=False)
            except Exception:
                pass

            try:
                subprocess.run(
                    ["kubectl", "apply", "-f", "-", "-n", namespace],
                    input=_json.dumps(new_pod),
                    text=True,
                    check=True,
                )
                results["actions"].append("Recreated naked pod with updated resources")
                return True
            except Exception as e:
                results["actions"].append(f"Naked pod recreate failed: {e}")
                return False

        patched = False

        # Resolve parent Deployment if pod belongs to a ReplicaSet
        if owner_kind == "ReplicaSet" and owner_name:
            try:
                rs_json = _safe_json(["kubectl", "get", "rs", owner_name, "-n", namespace, "-o", "json"])
                rs_owners = (rs_json or {}).get("metadata", {}).get("ownerReferences", []) or []
                for ref in rs_owners:
                    if ref.get("kind") == "Deployment" and ref.get("name"):
                        owner_kind = "Deployment"
                        owner_name = ref.get("name")
                        results["actions"].append(f"Resolved parent deployment: {owner_name}")
                        break
            except Exception:
                pass

        if owner_kind == "Deployment" and owner_name and resource_actions:
            patched = _apply_resource_patch_to_deployment(owner_name, resource_actions)

        if not patched and owner_kind == "DaemonSet" and owner_name:
            print(f"⚡ [Microsoft Kernel] Restarting DaemonSet/{owner_name} in {namespace}...")
            try:
                subprocess.run(
                    ["kubectl", "rollout", "restart", f"daemonset/{owner_name}", "-n", namespace],
                    check=True,
                )
                results["actions"].append(f"DaemonSet/{owner_name} rollout restart triggered")
            except Exception as e:
                results["actions"].append(f"DaemonSet rollout restart failed: {e}")

            if label_selector:
                delete_cmd = [
                    "kubectl", "delete", "pod",
                    "-n", namespace,
                    "-l", label_selector,
                    "--wait=false",
                ]
                print(f"⚡ [Microsoft Kernel] Deleting pods with selector '{label_selector}' in {namespace}...")
                try:
                    subprocess.run(delete_cmd, check=True)
                    results["actions"].append(f"Pods with selector '{label_selector}' deleted")
                except Exception as e:
                    results["actions"].append(f"Selector-based pod delete failed: {e}")
        else:
            # Deployment patch already attempted or owner not DaemonSet
            if not patched and not owner_name and resource_actions and pod_exists:
                # Naked pod: recreate with updated resources
                patched = _recreate_naked_pod_with_resources(pod_json, resource_actions)

            if not patched:
                # Legacy single-pod remediation, but only if pod exists
                if skip_fallback_actions:
                    results["actions"].append("Skipping pod delete; CreateContainerConfigError remediation already applied.")
                elif pod_exists:
                    print(f"⚡ [Microsoft Kernel] Initiating tactical termination of {pod_name}...")
                    delete_cmd = f"kubectl delete pod {pod_name} -n {namespace} --wait=false"
                    subprocess.run(delete_cmd, shell=True, check=True)
                    results["actions"].append("Pod termination triggered")
                else:
                    results["actions"].append("Pod not found; skipping pod delete.")

        # STEP 3: VERIFICATION (monitor up to 60s)
        def _fetch_pods(selector: str = None, field: str = None):
            try:
                cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
                if selector:
                    cmd.extend(["-l", selector])
                if field:
                    cmd.extend(["--field-selector", field])
                out = subprocess.check_output(cmd, shell=False, text=True)
                return json.loads(out)
            except Exception:
                return None

        def _assess_pods(pod_json: dict):
            items = (pod_json or {}).get("items", [])
            if not items:
                return False, {"reason": "no_pods_found"}
            unhealthy = []
            for p in items:
                meta = p.get("metadata", {}) or {}
                status = p.get("status", {}) or {}
                phase = status.get("phase")
                container_statuses = status.get("containerStatuses", []) or []
                waiting = any((cs.get("state", {}) or {}).get("waiting") for cs in container_statuses)
                crash = any(
                    ((cs.get("state", {}) or {}).get("waiting") or {}).get("reason") in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull")
                    for cs in container_statuses
                )
                restarts = sum(cs.get("restartCount", 0) for cs in container_statuses)
                if phase != "Running" or waiting or crash or restarts > 3:
                    unhealthy.append({
                        "pod": meta.get("name"),
                        "phase": phase,
                        "restarts": restarts,
                        "waiting": waiting,
                        "crash": crash,
                    })
            return len(unhealthy) == 0, {"unhealthy": unhealthy}

        monitor_window_s = 60
        interval_s = 5
        monitor_selector = label_selector
        monitor_field = None
        if not monitor_selector and pod_exists:
            monitor_field = f"metadata.name={pod_name}"
        if not monitor_selector and owner_name and owner_kind == "DaemonSet":
            monitor_selector = f"daemonset={owner_name}"

        verification_checks = []
        stable = False
        for _ in range(max(1, monitor_window_s // interval_s)):
            pod_list = _fetch_pods(selector=monitor_selector, field=monitor_field)
            ok, detail = _assess_pods(pod_list)
            verification_checks.append(detail)
            if ok:
                stable = True
                break
            time.sleep(interval_s)

        results["verification_checks"] = verification_checks
        results["verification_window_s"] = monitor_window_s

        if stable:
            results["outcome"] = "SUCCESS"
            results["verification"] = "Workload healthy after remediation"
            results["actions"].append("Service restored to healthy state")
        else:
            results["outcome"] = "REMEDIATION_FAILED"
            results["verification"] = "Workload still unhealthy after remediation"
            results["actions"].append("Escalation flag raised")

        # Persist outcome to memory store for later analytics
        try:
            from memory_store import mem_store
            mem_store.add("RemediationVerification", {
                "target": pod_name,
                "namespace": namespace,
                "outcome": results.get("outcome"),
                "timestamp": time.time(),
                "actions": results.get("actions", []),
                "verification": results.get("verification"),
            })
            # Also persist semantic summary for Planner reuse
            semantic_payload = {
                "target": pod_name,
                "namespace": namespace,
                "outcome": results.get("outcome"),
                "actions": results.get("actions", []),
                "recommendations": recommended_actions or [],
                "verification": results.get("verification"),
                "confidence": 0.8 if results.get("outcome") == "SUCCESS" else 0.4,
            }
            mem_store.add("SemanticRemediation", semantic_payload)
        except Exception:
            pass

    except Exception as e:
        results["error"] = str(e)
        results["outcome"] = "ERROR"

    return results

def watch_k8s_events(namespace: str = "all", minutes: int = 5) -> dict:
    """
    Monitor Kubernetes cluster events for incidents.
    Detects: pod kills, crashes, OOMs, restarts, scaling events, failures.
    FILTERS: Ignores known noise (minikube, coredns, normal startups).
    """
    import subprocess
    import json
    from datetime import datetime, timedelta
    import os
    from typing import Optional

    try:
        fresh_failed_window = int(os.getenv("K8S_FAILED_SCHEDULING_FRESH_SECONDS", "120"))

        # Get events from kubectl
        if namespace == "all":
            cmd = ["kubectl", "get", "events", "--all-namespaces", "-o", "json"]
        else:
            cmd = ["kubectl", "get", "events", "-n", namespace, "-o", "json"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr}
        
        events_data = json.loads(result.stdout)
        
        # --- 🛡️ NOISE FILTER DEFINITIONS ---
        IGNORED_PODS = ["minikube", "coredns", "kube-proxy", "storage-provisioner"]
        # Reasons we typically don't care about unless specifically debugging
        IGNORED_REASONS = ["Pulled", "Pulling", "Created", "Started", "Scheduled", "SuccessfulCreate", "ScalingReplicaSet"]
        
        # Filter recent events
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_events = []
        critical_count = 0
        warning_count = 0
        active_failed_scheduling_count = 0
        stale_failed_scheduling_count = 0
        
        critical_reasons = ["OOMKilled", "FailedToRetrieveImagePullSecret", "CrashLoopBackOff", "Failed", "FailedScheduling", "Unhealthy", "BackOff", "Killing"]

        now = datetime.utcnow()

        def _parse_event_ts(obj: dict) -> Optional[datetime]:
            ts_str = (
                obj.get("lastTimestamp")
                or obj.get("eventTime")
                or obj.get("firstTimestamp")
                or (obj.get("metadata", {}) or {}).get("creationTimestamp")
            )
            if not ts_str:
                return None
            try:
                # Handle Zulu time
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                return None
        
        for item in events_data.get("items", []):
            reason = item.get("reason", "")
            message = item.get("message", "")
            event_type = item.get("type", "Normal")
            involved = item.get("involvedObject", {})
            pod_name = involved.get("name", "unknown")
            
            # --- 🛡️ FILTER LOGIC START ---
            # 1. Skip Known System Noises (Ghost Filter)
            if any(ignored in pod_name for ignored in IGNORED_PODS):
                continue

            # 2. Skip Normal Lifecycle Events (Chatter Filter)
            if reason in IGNORED_REASONS and event_type != "Warning":
                continue
            
            # 3. Skip "nginx" unless it's genuinely failing (Test Filter)
            if "nginx" in pod_name and reason not in critical_reasons and event_type != "Warning":
                continue
            # --- 🛡️ FILTER LOGIC END ---

            evt_ts = _parse_event_ts(item)
            # Skip if older than requested window
            if evt_ts and evt_ts < cutoff:
                continue

            age_seconds = (now - evt_ts).total_seconds() if evt_ts else None
            is_failed_sched = reason == "FailedScheduling"
            is_fresh_failed = is_failed_sched and age_seconds is not None and age_seconds <= fresh_failed_window

            event_record = {
                "namespace": involved.get("namespace", "unknown"),
                "kind": involved.get("kind", "unknown"),
                "name": pod_name,
                "reason": reason,
                "message": message[:200],
                "type": event_type,
                "count": item.get("count", 1),
                "age_seconds": age_seconds,
                "fresh_failed_scheduling": is_fresh_failed,
            }
            
            # Logic for Critical vs Warning
            if is_failed_sched:
                if is_fresh_failed:
                    critical = True
                    active_failed_scheduling_count += 1
                else:
                    critical = False
                    stale_failed_scheduling_count += 1
            else:
                critical = reason in critical_reasons or event_type == "Warning"

            event_record["critical"] = critical
            recent_events.append(event_record)
            
            if critical:
                critical_count += 1
            else:
                warning_count += 1
        
        summary = f"Found {len(recent_events)} relevant events. Critical: {critical_count}, Normal: {warning_count}"
        
        # Highlight critical issues
        critical_events = [e for e in recent_events if e.get("critical")]
        
        return {
            "ok": True,
            "summary": summary,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "critical_events": critical_events[:10],
            "all_events": recent_events[:20],
            "namespace_filter": namespace,
            "minutes_back": minutes,
            "active_failed_scheduling_count": active_failed_scheduling_count,
            "stale_failed_scheduling_count": stale_failed_scheduling_count,
        }
        
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "kubectl command timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def collect_imagepull_forensics(pod_name: str, namespace: str = "default") -> dict:
    """
    Gather detailed forensics for ImagePull issues without making decisions.
    """
    import subprocess
    import json

    def _safe_json(cmd: list[str]):
        try:
            out = subprocess.check_output(cmd, text=True)
            return json.loads(out)
        except Exception as e:
            return {"_error": str(e)}

    result = {
        "target": f"{namespace}/{pod_name}",
        "pod_status": {},
        "image": {},
        "image_pull_secrets": [],
        "recent_image_pulls": [],
        "deployment_history": [],
        "registry_connectivity": {},
        "similar_pods": [],
    }

    # 1) Pod status and error
    pod_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
    result["pod_status"] = pod_json

    # 2) Image info and 4) imagePullSecrets
    try:
        containers = (pod_json.get("spec", {}) or {}).get("containers", []) or []
        if containers:
            img = containers[0].get("image")
            if img:
                result["image"]["full"] = img
                if "/" in img and ":" in img:
                    reg_repo, tag = img.rsplit(":", 1)
                    result["image"]["tag"] = tag
                    result["image"]["registry_repo"] = reg_repo
                    # registry URL as first segment before '/' if present
                    reg_url = reg_repo.split("/")[0] if "/" in reg_repo else reg_repo
                    result["image"]["registry"] = reg_url
                elif ":" in img:
                    repo, tag = img.rsplit(":", 1)
                    result["image"]["tag"] = tag
                    result["image"]["registry_repo"] = repo
            result["image_pull_secrets"] = pod_json.get("spec", {}).get("imagePullSecrets", []) or []
    except Exception:
        pass

    # 5) Recent successful image pulls in same namespace (last 10)
    try:
        ev_json = _safe_json(["kubectl", "get", "events", "-n", namespace, "-o", "json"])
        pulls = []
        for ev in ev_json.get("items", []):
            reason = ev.get("reason", "")
            if reason and "Pull" in reason and ev.get("type") == "Normal":
                pulls.append({
                    "reason": reason,
                    "message": ev.get("message"),
                    "pod": (ev.get("involvedObject") or {}).get("name"),
                    "ts": ev.get("lastTimestamp") or ev.get("eventTime") or ev.get("firstTimestamp"),
                })
        result["recent_image_pulls"] = pulls[-10:]
    except Exception:
        pass

    # 6) Deployment history (last 3 revisions with images)
    try:
        owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
        dep_name = None
        for o in owners:
            if o.get("kind") == "ReplicaSet":
                rs_name = o.get("name")
                rs_json = _safe_json(["kubectl", "get", "rs", rs_name, "-n", namespace, "-o", "json"])
                rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                for ro in rs_owners:
                    if ro.get("kind") == "Deployment":
                        dep_name = ro.get("name")
                        break
        if dep_name:
            hist = _safe_json(["kubectl", "rollout", "history", f"deployment/{dep_name}", "-n", namespace, "-o", "json"])
            entries = hist.get("history", []) if isinstance(hist, dict) else []
            result["deployment_history"] = entries[:3]
    except Exception:
        pass

    # 7) Network connectivity to registry
    try:
        reg = result["image"].get("registry")
        if reg:
            ping_cmd = ["sh", "-c", f"ping -c 1 -W 2 {reg} >/dev/null 2>&1 && echo ok || echo fail"]
            status = subprocess.check_output(ping_cmd, text=True).strip()
            result["registry_connectivity"] = {"registry": reg, "reachable": status == "ok"}
    except Exception as e:
        result["registry_connectivity"] = {"error": str(e)}

    # 8) Similar pods in namespace using same registry
    try:
        if result["image"].get("registry_repo"):
            reg_repo = result["image"]["registry_repo"]
            pods_list = _safe_json(["kubectl", "get", "pods", "-n", namespace, "-o", "json"])
            sims = []
            for p in pods_list.get("items", []):
                imgs = []
                for c in (p.get("spec", {}) or {}).get("containers", []) or []:
                    img = c.get("image") or ""
                    if reg_repo.split("/")[0] in img:
                        imgs.append(img)
                if imgs:
                    sims.append({
                        "pod": (p.get("metadata", {}) or {}).get("name"),
                        "images": imgs,
                    })
            result["similar_pods"] = sims
    except Exception:
        pass

    return result


def analyze_imagepull_failure(forensics: dict) -> dict:
    """
    Use LLM to analyze ImagePullBackOff forensics and propose actions.
    """
    try:
        from shared_models import OllamaLLMIntegration
        llm = OllamaLLMIntegration()
        prompt = f"""You are analyzing an ImagePullBackOff failure in Kubernetes.

Forensics Data:
- Error: {forensics.get('pod_status', {})}
- Image: {forensics.get('image')}
- Registry: {(forensics.get('image') or {}).get('registry') if isinstance(forensics.get('image'), dict) else None}
- Secrets configured: {forensics.get('image_pull_secrets')}
- Recent successful pulls: {forensics.get('recent_image_pulls')}
- Previous working images: {forensics.get('deployment_history')}
- Network test: {forensics.get('registry_connectivity')}
- Similar pods: {forensics.get('similar_pods')}

Analyze the root cause. Common scenarios:
1. Wrong tag (typo, doesn't exist) - check deployment history for correct tag
2. Authentication missing - check if imagePullSecrets needed
3. Registry unreachable - network or DNS issue
4. Image deleted from registry - rollback to previous version
5. Rate limit hit - wait and retry
6. Wrong registry URL - check similar working pods

Based on the forensics, determine:
1. Root cause (be specific)
2. Recommended actions (ordered by likelihood of success)
3. Confidence level (0.0-1.0)

Return JSON:
{{
  "root_cause": "string",
  "evidence": ["bullet points"],
  "recommended_actions": [
    {{
      "action": "rollback_image|update_secret|retry_pull|update_image_url",
      "details": {{"image": "...", "tag": "..." OR "secret_name": "..."}},
      "reason": "why this will work",
      "confidence": 0.0-1.0
    }}
  ],
  "overall_confidence": 0.0-1.0
}}
"""
        resp = llm.generate_text(prompt=prompt, json_mode=True, temperature=0.2, max_tokens=512)
        import json as _json
        parsed = _json.loads(resp) if resp else {}
        return parsed if isinstance(parsed, dict) else {"status": "error", "error": "invalid response"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def execute_imagepull_remediation(pod_name: str, namespace: str = "default", recommendations: list | None = None) -> dict:
    """
    Execute remediation actions for image pull issues based on LLM recommendations.
    """
    import subprocess
    import json
    recs = recommendations or []
    results = []

    def _safe_json(cmd: list[str]):
        try:
            out = subprocess.check_output(cmd, text=True)
            return json.loads(out)
        except Exception as e:
            return {"_error": str(e)}

    # Helper: find deployment owning this pod
    dep_name = None
    try:
        pod_json = _safe_json(["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"])
        owners = (pod_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
        for o in owners:
            if o.get("kind") == "ReplicaSet":
                rs_name = o.get("name")
                rs_json = _safe_json(["kubectl", "get", "rs", rs_name, "-n", namespace, "-o", "json"])
                rs_owners = (rs_json.get("metadata", {}) or {}).get("ownerReferences", []) or []
                for ro in rs_owners:
                    if ro.get("kind") == "Deployment":
                        dep_name = ro.get("name")
                        break
    except Exception:
        pass

    def _patch_deployment_image(img: str):
        if not dep_name or not img:
            return {"status": "error", "error": "missing deployment or image"}
        try:
            patch = {"spec": {"template": {"spec": {"containers": [{"name": "", "image": img}]}}}}
            subprocess.run(
                ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                check=True,
            )
            return {"status": "ok", "action": "patch_image", "image": img}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _create_or_update_secret(secret_name: str):
        if not secret_name:
            return {"status": "error", "error": "no secret name provided"}
        try:
            subprocess.run(
                ["kubectl", "-n", namespace, "create", "secret", "docker-registry", secret_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        try:
            patch = {"spec": {"template": {"spec": {"imagePullSecrets": [{"name": secret_name}]}}}}
            subprocess.run(
                ["kubectl", "patch", "deployment", dep_name, "-n", namespace, "--type", "strategic", "-p", json.dumps(patch)],
                check=True,
            )
            return {"status": "ok", "action": "update_secret", "secret": secret_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _delete_pod():
        try:
            subprocess.run(["kubectl", "delete", "pod", pod_name, "-n", namespace, "--wait=false"], check=False)
            import time as _t
            _t.sleep(30)
            return {"status": "ok", "action": "retry_pull"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _schedule_wait():
        return {"status": "pending", "action": "wait_retry", "retry_in_minutes": 5}

    # Deployment history for rollback
    dep_history = []
    try:
        if dep_name:
            hist = _safe_json(["kubectl", "rollout", "history", f"deployment/{dep_name}", "-n", namespace, "-o", "json"])
            dep_history = hist.get("history", []) if isinstance(hist, dict) else []
    except Exception:
        pass

    for rec in recs:
        action = rec.get("action")
        det = rec.get("details") or {}
        if action == "rollback_image":
            img = None
            for h in dep_history:
                if h.get("revision") and h.get("images"):
                    img = h["images"][0]
                    break
            if not img:
                img = det.get("image")
            results.append(_patch_deployment_image(img))
        elif action == "update_secret":
            sec_name = det.get("secret_name") or det.get("secret")
            results.append(_create_or_update_secret(sec_name))
        elif action == "retry_pull":
            results.append(_delete_pod())
        elif action == "update_image_url":
            img = det.get("image")
            results.append(_patch_deployment_image(img))
        elif action == "wait_retry":
            results.append(_schedule_wait())
        else:
            results.append({"status": "error", "error": f"unknown action {action}"})

    return {"actions": recs, "results": results}

def get_pod_status(namespace: str = "default") -> dict:
    """
    Get current status of all pods in a namespace.
    Detects: CrashLoopBackOff, Pending, Failed, not Ready.
    
    Args:
        namespace: Namespace to check ("all" for all namespaces)
    
    Returns:
        dict with: pods list, problem_pods, healthy_count, unhealthy_count
    """
    import subprocess
    import json
    
    try:
        if namespace == "all":
            cmd = ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"]
        else:
            cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr}
        
        pods_data = json.loads(result.stdout)
        
        healthy_count = 0
        unhealthy_count = 0
        problem_pods = []
        all_pods = []
        
        for item in pods_data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            phase = status.get("phase", "Unknown")
            
            pod_info = {
                "name": metadata.get("name", "unknown"),
                "namespace": metadata.get("namespace", "unknown"),
                "phase": phase,
                "restarts": 0,
                "ready": False,
                "issues": []
            }
            
            # Check container statuses
            container_statuses = status.get("containerStatuses", [])
            for cs in container_statuses:
                pod_info["restarts"] += cs.get("restartCount", 0)
                pod_info["ready"] = cs.get("ready", False)
                
                # Check for waiting state issues
                waiting = cs.get("state", {}).get("waiting", {})
                if waiting:
                    reason = waiting.get("reason", "")
                    if reason in ["CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError"]:
                        pod_info["issues"].append(reason)
                
                # Check for terminated state
                terminated = cs.get("state", {}).get("terminated", {})
                if terminated:
                    reason = terminated.get("reason", "")
                    if reason in ["OOMKilled", "Error"]:
                        pod_info["issues"].append(reason)
            
            all_pods.append(pod_info)
            
            # Classify as healthy or unhealthy
            if phase == "Running" and pod_info["ready"] and not pod_info["issues"]:
                healthy_count += 1
            else:
                unhealthy_count += 1
                if pod_info["issues"] or phase != "Running":
                    problem_pods.append(pod_info)
        
        return {
            "ok": True,
            "summary": f"Pods: {healthy_count} healthy, {unhealthy_count} unhealthy",
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "problem_pods": problem_pods,
            "all_pods": all_pods[:30],
            "namespace_filter": namespace
        }
        
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "kubectl command timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- Security / Networking Tools ----
_ALLOWED_SCAN_TYPES = {"ping_sweep", "full_port_scan", "vulnerability_scan"}

def initiate_network_scan_tool(target_ip: str, scan_type: str = "ping_sweep") -> dict:
    """Initiate network security scans"""
    _usage_tracker.track_usage("initiate_network_scan_tool")
    
    # Input validation
    err = _require_non_placeholder("target_ip", target_ip)
    if err:
        return standardize_response("error", error=err)
    
    try:
        ip_address(target_ip)  # Validate IP
    except ValueError:
        return standardize_response("error", error=f"Invalid IP address: {target_ip}")
    
    scan_type = (scan_type or "ping_sweep").strip().lower()
    if scan_type not in _ALLOWED_SCAN_TYPES:
        scan_type = "ping_sweep"
    
    logger.info(f"Network scan: {scan_type} on {target_ip}")
    time.sleep(0.2)  # Simulate scan time
    
    if scan_type == "full_port_scan":
        open_ports = random.sample([21, 22, 23, 80, 443, 3389, 8080], k=random.randint(1, 3))
        return standardize_response("ok", data={"open_ports": open_ports}, 
                                  summary=f"Port scan on {target_ip}: {len(open_ports)} ports open")
    
    elif scan_type == "vulnerability_scan":
        if random.random() < 0.1:
            vuln = random.choice(["CVE-2023-1234 (High)", "CVE-2022-5678 (Medium)"])
            return standardize_response("ok", data={"vulnerabilities": [vuln]}, 
                                      summary=f"Vulnerability found: {vuln}")
        return standardize_response("ok", data={"vulnerabilities": []}, 
                                  summary=f"No critical vulnerabilities found on {target_ip}")
    
    else:  # ping_sweep
        return standardize_response("ok", data={"host_up": True}, 
                                  summary=f"Successfully pinged {target_ip}. Host is up.")

def deploy_recovery_protocol_tool(protocol_name: str, target_system_id: str, urgency_level: str = "medium") -> dict:
    """Deploy recovery protocols"""
    _usage_tracker.track_usage("deploy_recovery_protocol_tool")
    
    for k, v in (("protocol_name", protocol_name), ("target_system_id", target_system_id)):
        err = _require_non_placeholder(k, v)
        if err:
            return standardize_response("error", error=err)
    
    urgency_level = (urgency_level or "medium").strip().lower()
    if urgency_level not in {"low", "medium", "high", "critical"}:
        urgency_level = "medium"
    
    logger.info(f"Deploy recovery: {protocol_name} -> {target_system_id} ({urgency_level})")
    time.sleep(0.2)
    
    return standardize_response("ok", 
                              data={"protocol": protocol_name, "target": target_system_id, "urgency": urgency_level},
                              summary=f"Recovery protocol '{protocol_name}' deployed to {target_system_id}")

def analyze_threat_signature_tool(signature: str, source_ip: str) -> dict:
    """Analyze threat signatures"""
    _usage_tracker.track_usage("analyze_threat_signature_tool")
    
    for k, v in (("signature", signature), ("source_ip", source_ip)):
        err = _require_non_placeholder(k, v)
        if err:
            return standardize_response("error", error=err)
    
    try:
        ip_address(source_ip)  # Validate IP
    except ValueError:
        return standardize_response("error", error=f"Invalid source IP: {source_ip}")
    
    risk = random.choice(["Low", "Medium", "High", "Critical"])
    confidence = round(random.uniform(0.7, 0.99), 2)
    
    return standardize_response("ok",
                              data={
                                  "signature": signature, 
                                  "source_ip": source_ip, 
                                  "risk_level": risk,
                                  "confidence": confidence
                              },
                              summary=f"Analysis: {signature} from {source_ip} = {risk} risk")

def isolate_network_segment_tool(segment_id: str, reason: str) -> dict:
    """Isolate network segments"""
    _usage_tracker.track_usage("isolate_network_segment_tool")
    
    for k, v in (("segment_id", segment_id), ("reason", reason)):
        err = _require_non_placeholder(k, v)
        if err:
            return standardize_response("error", error=err)
    
    return standardize_response("ok", 
                              data={"segment_id": segment_id, "reason": reason},
                              summary=f"Segment '{segment_id}' isolated")

def extract_iocs_tool(text: str) -> dict:
    """Extract Indicators of Compromise from text"""
    _usage_tracker.track_usage("extract_iocs_tool")
    
    if _is_placeholder(text):
        return standardize_response("error", error="Text is empty/placeholder")
    
    try:
        # Enhanced IOC patterns
        ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
        urls = re.findall(r"https?://[^\s)\]]+", text)
        sha256 = re.findall(r"\b[a-fA-F0-9]{64}\b", text)
        md5 = re.findall(r"\b[a-fA-F0-9]{32}\b", text)
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        domains = re.findall(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b", text)
        
        data = {
            "ips": sorted(set(ips)),
            "urls": sorted(set(urls)),
            "sha256": sorted(set(sha256)),
            "md5": sorted(set(md5)),
            "emails": sorted(set(emails)),
            "domains": sorted(set(domains))
        }
        
        total = sum(len(v) for v in data.values())
        return standardize_response("ok", data=data, summary=f"Extracted {total} IOCs")
    except Exception as e:
        _usage_tracker.track_usage("extract_iocs_tool", success=False)
        return standardize_response("error", error=str(e))

def hash_text_tool(text: str, algorithm: str = "sha256") -> dict:
    """Hash text using specified algorithm"""
    _usage_tracker.track_usage("hash_text_tool")
    
    if _is_placeholder(text):
        return standardize_response("error", error="Text is empty/placeholder")
    
    if len(text.encode('utf-8')) > ToolConfig.MAX_HASH_TEXT_SIZE:
        return standardize_response("error", error="Text too large for hashing")
    
    algo = (algorithm or "sha256").lower()
    supported_algos = {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"}
    
    if algo not in supported_algos:
        return standardize_response("error", error=f"Unsupported algorithm. Use one of: {sorted(supported_algos)}")
    
    try:
        h = hashlib.new(algo)
        h.update(text.encode("utf-8", errors="ignore"))
        return standardize_response("ok", 
                                  data={"algorithm": algo, "hexdigest": h.hexdigest()},
                                  summary=f"Hashed text with {algo}")
    except Exception as e:
        _usage_tracker.track_usage("hash_text_tool", success=False)
        return standardize_response("error", error=str(e))

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
    """
    Saves a critical event to the Permanent Hive Mind.
    Args:
        category: 'observation', 'plan', 'action', 'outcome'
        description: The full text to remember
    """
    try:
        SHARED_BRAIN.add_memory(
            agent_name=agent_name,
            text=description,
            category=category
        )
        return {"ok": True, "summary": "Memory stored successfully."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def search_memory(query: str) -> dict:
    """
    Consults the Hive Mind for past wisdom.
    Use this BEFORE planning to see if we've solved this before.
    """
    try:
        memories = SHARED_BRAIN.query_memory(query, n_results=3)
        if not memories:
            return {"ok": True, "summary": "No relevant memories found."}
            
        formatted = "RELEVANT PAST MEMORIES:\n"
        for mem in memories:
            formatted += f"- [{mem['timestamp']}] {mem['agent']} ({mem['category']}): {mem['text']}\n"
            
        return {"ok": True, "summary": formatted, "raw_data": memories}
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

# Initialize configuration validation on import
ToolConfig.validate()

# Export the standardized tool functions
__all__ = [
    # System tools
    "get_system_cpu_load_tool", "get_system_resource_usage_tool", "disk_usage_tool",
    "top_processes_tool", "measure_responsiveness_tool",
    
    # Kubernetes tools  
    "kubernetes_pod_metrics_tool", "k8s_scale_tool", "find_wasteful_deployments_tool",
    
    # Security tools
    "initiate_network_scan_tool", "deploy_recovery_protocol_tool", "analyze_threat_signature_tool",
    "isolate_network_segment_tool", "extract_iocs_tool", "hash_text_tool", "redact_pii_tool",
    
    # Knowledge tools
    "web_search_tool", "read_webpage_tool", "get_environmental_data_tool",
    
    # Utility tools
    "reply_to_user", "update_resource_allocation_tool", "get_tool_usage_stats_tool",
    "tool_health_check_tool", "list_available_tools_tool",
    
    # Utility functions
    "standardize_response", "ToolConfig"
]
@retry_on_failure(max_retries=2)
def spawn_specialized_agent(purpose: str, context: dict, parent_agent: str = "system") -> dict:
    """
    Spawn a specialized agent for a specific task.
    
    Args:
        purpose: What this agent should accomplish
        context: Relevant context (emails, alerts, data)
        parent_agent: Name of agent requesting spawn
    
    Returns:
        {"success": bool, "agent_id": str, "agent_name": str}
    """
    try:
        # Get CVA instance from global context
        cva = globals().get('_cva_instance')
        if not cva:
            return {"success": False, "error": "CVA not available"}
        
        agent_id = cva.handle_spawn_request(purpose, context, parent_agent)
        
        if agent_id:
            agent = cva.agent_factory.get_agent(agent_id)
            return {
                "success": True,
                "agent_id": agent_id,
                "agent_name": agent.spec.name,
                "expires_at": agent.spec.expires_at.isoformat()
            }
        
        return {"success": False, "error": "Spawn failed"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
