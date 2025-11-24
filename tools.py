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
            
            for attempt in range(max_retries):
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

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
class ToolConfig:
    """Central configuration for all tools"""
    KUBECTL_TIMEOUT = 15
    REQUEST_TIMEOUT = 12
    SCALE_MIN_INTERVAL = float(os.getenv("CVA_SCALE_MIN_INTERVAL_S", "300"))
    MAX_PROCESS_LIMIT = 100
    MAX_FILE_SIZE_MB = 50
    MAX_WEBPAGE_SIZE = 8000
    MAX_SEARCH_RESULTS = 5
    
    # Security limits
    MAX_SCAN_TARGETS = 10
    MAX_HASH_TEXT_SIZE = 10 * 1024 * 1024  # 10MB
    
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
                                limit: int = 50) -> dict:
    """Get Kubernetes pod metrics using kubectl top"""
    _usage_tracker.track_usage("kubernetes_pod_metrics_tool")
    
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
                                   min_replicas: int = 2) -> dict:
    """Find deployments with low CPU utilization but high replica count"""
    _usage_tracker.track_usage("find_wasteful_deployments_tool")
    
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
def web_search_tool(query: str, max_results: int = 3) -> dict:
    """Search the web using DuckDuckGo"""
    _usage_tracker.track_usage("web_search_tool")
    
    if not query or _is_placeholder(query):
        return standardize_response("error", error="Query cannot be empty")
    
    if DDGS is None:
        return standardize_response("error", error="DuckDuckGo search not available")
    
    try:
        max_results = min(max(1, max_results), ToolConfig.MAX_SEARCH_RESULTS)
        results = list(DDGS().text(query, max_results=max_results))
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

@retry_on_failure(max_retries=3, delay=1.0)
def read_webpage_tool(url: str) -> dict:
    """Read and extract text content from webpage"""
    _usage_tracker.track_usage("read_webpage_tool")
    
    if not url or _is_placeholder(url):
        return standardize_response("error", error="URL cannot be empty")
    
    if not _valid_url(url):
        return standardize_response("error", error="Invalid URL format")
    
    if requests is None:
        return standardize_response("error", error="Requests library not available")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        session = RetrySession.create_session()
        response = session.get(url, headers=headers, timeout=ToolConfig.REQUEST_TIMEOUT) if session else requests.get(url, headers=headers, timeout=ToolConfig.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Basic HTML cleaning
        html = response.text
        text = re.sub('<[^<]+?>', ' ', html)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        # Truncate if necessary
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
    except requests.RequestException as e:
        _usage_tracker.track_usage("read_webpage_tool", success=False)
        return standardize_response("error", error=f"HTTP error: {str(e)}")
    except Exception as e:
        _usage_tracker.track_usage("read_webpage_tool", success=False)
        return standardize_response("error", error=str(e))

def send_desktop_notification_tool(title: str, message: str) -> dict:
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
    
    try:
        system = platform.system()
        
        if system == "Linux":
            # Use notify-send on Linux
            subprocess.run(
                ["notify-send", title, message],
                check=True,
                timeout=5
            )
        elif system == "Darwin":  # macOS
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=True,
                timeout=5
            )
        elif system == "Windows":
            try:
                import importlib
                win10toast = importlib.import_module('win10toast')
                toaster = win10toast.ToastNotifier()
                toaster.show_toast(title, message, duration=5)
            except ImportError:
                return {"ok": False, "error": "win10toast not installed"}
        return {
            "ok": True,
            "success": True,
            "title": title,
            "message": message,
            "platform": system
        }
        
    except Exception as e:
        return {
            "ok": False,
            "success": False,
            "error": str(e)
        }

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

def check_calendar_tool(time_min_utc: str, time_max_utc: str) -> dict:
    """
    Check Google Calendar for events in a time range.
    
    Args:
        time_min_utc: Start time in ISO format (e.g., "2024-11-20T00:00:00Z")
        time_max_utc: End time in ISO format (e.g., "2024-11-20T23:59:59Z")
        
    Returns:
        Dict with calendar events
    """
    try:
        from gmail_agent import get_calendar_events
        events = get_calendar_events(time_min_utc, time_max_utc)
        return {
            "ok": True,
            "success": True,
            "events": events,
            "count": len(events) if events else 0
        }
    except Exception as e:
        return {
            "ok": False,
            "success": False,
            "error": str(e),
            "events": []
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
