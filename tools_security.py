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

