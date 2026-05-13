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

from tools_base import SafeSubprocess, standardize_response

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

# (standardize_response imported from tools_base)

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


# ------------------------------------------------------------------------------
# Submodule Imports
# ------------------------------------------------------------------------------
from tools_system import *
from tools_k8s import *
from tools_k8s import (
    k8s_patch_deployment_image,
    k8s_patch_probe,
    k8s_rollout_undo,
)
from tools_security import *
from tools_web import *
from tools_swarm import *

# Backward compatibility aliases
get_system_cpu_load_tool = system_get_cpu_load
get_system_resource_usage_tool = get_system_resource_usage_tool
disk_usage_tool = system_get_disk_usage

# Initialize configuration validation on import
ToolConfig.validate()


# --- PROCESS SPAWNING (Mitosis) ---
def spawn_agent(purpose: str, context: Optional[Dict[str, Any]] = None, ttl_hours: float = 4.0) -> dict:
    """
    Spawns a new dynamic agent (sub-agent) for a specific task.
    
    Args:
        purpose: Clear description of what the new agent should do.
        context: Optional dictionary of context/memory to pass to the agent.
        ttl_hours: How long the agent should live (max 24).
    """
    _usage_tracker.track_usage("spawn_agent")
    
    # Validation
    if not purpose or len(purpose) < 10:
        return standardize_response("error", error="Purpose must be descriptive (min 10 chars)")
    
    if ttl_hours > 24:
        return standardize_response("error", error="TTL cannot exceed 24 hours")

    # Access global system instance
    try:
        # Get CVA instance from global context (injected by app.py)
        cva = globals().get('_cva_instance')
        
        if not cva:
            return standardize_response("error", error="System instance not available for spawning (Context missing)")
        
        directive = {
            "type": "SPAWN_DYNAMIC_AGENT",
            "purpose": purpose,
            "context": context or {},
            "requester_agent": "Tool_Caller", 
            "timestamp": _now_iso()
        }
        
        # Inject directive
        cva.inject_directives([directive])
        
        return standardize_response("ok", 
                                  summary=f"Spawn request for '{purpose}' submitted to system kernel.",
                                  data={"status": "queued", "purpose": purpose})
                                  
    except Exception as e:
        _usage_tracker.track_usage("spawn_agent", success=False)
        return standardize_response("error", error=str(e))

def send_email(to: str, subject: str, body: str) -> dict:
    """Sends a summary email (simulated)."""
    _usage_tracker.track_usage("send_email")
    logger.info(f"[EMAIL] To: {to}, Subject: {subject}, Body: {body[:100]}...")
    return standardize_response("ok", data={"to": to, "subject": subject}, summary=f"Email summary sent to {to}")

# --- PHASE 8: SELF-MODIFICATION (The Singularity) ---
# Safety: Allowlist of files that CAN be modified
SELF_PATCH_ALLOWLIST = [
    "agents.py",
    "curiosity_loop.py",
    "prompts.py",
    "shared_models.py",
    # Explicitly FORBIDDEN: app.py, start.sh, catalyst_vector_alpha.py
]

def self_patch(target_file: str, search_pattern: str, replacement: str) -> dict:
    """
    [PHASE 8: SINGULARITY] Patches a source file with safety gates.
    
    Args:
        target_file: Basename of file to patch (e.g., "agents.py")
        search_pattern: Exact string to find and replace
        replacement: String to replace with
    
    Returns:
        {"ok": True/False, "diff": "...", "reason": "..."}
    
    Safety Protocol:
        1. Only files in SELF_PATCH_ALLOWLIST can be edited.
        2. A backup is created before any change.
        3. After patching, `pytest tests/` is run.
        4. If tests FAIL, the backup is restored automatically.
    """
    import shutil
    import subprocess
    from datetime import datetime
    
    _usage_tracker.track_usage("self_patch")
    
    # --- Security Gate 1: Allowlist Check ---
    basename = os.path.basename(target_file)
    if basename not in SELF_PATCH_ALLOWLIST:
        return standardize_response("error", 
            error=f"SECURITY: '{basename}' is not in the modification allowlist. "
                  f"Allowed: {SELF_PATCH_ALLOWLIST}")
    
    # Resolve full path relative to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(project_root, basename)
    
    if not os.path.exists(full_path):
        return standardize_response("error", error=f"File not found: {full_path}")
    
    # --- Safety Gate 2: Backup ---
    backup_dir = "/tmp/cva_backup"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{basename}.{timestamp}.bak")
    
    try:
        shutil.copy2(full_path, backup_path)
    except Exception as e:
        return standardize_response("error", error=f"Backup failed: {e}")
    
    # --- Read and Patch ---
    try:
        with open(full_path, 'r') as f:
            original_content = f.read()
        
        if search_pattern not in original_content:
            return standardize_response("error", 
                error=f"Search pattern not found in {basename}. No changes made.",
                data={"backup_path": backup_path})
        
        patched_content = original_content.replace(search_pattern, replacement, 1)
        
        # Calculate diff preview
        diff_preview = f"--- {basename} (original)\n+++ {basename} (patched)\n"
        diff_preview += f"-{search_pattern[:100]}...\n+{replacement[:100]}...\n"
        
        # --- Apply Patch ---
        with open(full_path, 'w') as f:
            f.write(patched_content)
            
    except Exception as e:
        # Restore from backup on write failure
        shutil.copy2(backup_path, full_path)
        return standardize_response("error", error=f"Patch write failed, restored backup: {e}")
    
    # --- Safety Gate 3: Run Regression Tests ---
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        tests_passed = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        # Restore on timeout
        shutil.copy2(backup_path, full_path)
        return standardize_response("error", 
            error="Test suite timed out. Patch reverted.",
            data={"backup_path": backup_path})
    except Exception as e:
        # Restore on any test failure
        shutil.copy2(backup_path, full_path)
        return standardize_response("error", 
            error=f"Test execution failed. Patch reverted: {e}",
            data={"backup_path": backup_path})
    
    # --- Safety Gate 4: Revert if Tests Failed ---
    if not tests_passed:
        shutil.copy2(backup_path, full_path)
        return standardize_response("error",
            error="Regression tests FAILED. Patch automatically reverted.",
            data={
                "backup_path": backup_path,
                "test_stdout": result.stdout[-500:] if result.stdout else "",
                "test_stderr": result.stderr[-500:] if result.stderr else "",
            })
    
    # --- SUCCESS ---
    return standardize_response("ok",
        summary=f"Patch applied to {basename} and tests passed!",
        data={
            "file": basename,
            "backup_path": backup_path,
            "diff_preview": diff_preview,
            "test_output": result.stdout[-200:] if result.stdout else "Tests passed."
        })

def export_system_state_tool(destination: str = "backups/") -> dict:
    """Exports the entire evolved state of CVA for Digital Immortality (Phase 20)."""
    try:
        import zipfile
        os.makedirs(destination, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = os.path.join(destination, f"cva_state_export_{timestamp}.zip")
        
        # Files/Dirs to include
        targets = ["evolved_tools", "persistence_data", "config.py", "tool_registry.py", "prompts.py"]
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for target in targets:
                path = os.path.join(os.getcwd(), target)
                if os.path.isfile(path):
                    zipf.write(path, target)
                elif os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            zipf.write(os.path.join(root, file), 
                                       os.path.relpath(os.path.join(root, file), os.getcwd()))
        
        return {"ok": True, "summary": f"System state exported to {zip_filename}. Immortality sequence initiated."}
    except Exception as e:
        return {"ok": False, "error": f"Export failed: {e}"}

__all__ = ['ToolConfig', 'standardize_response', 'ToolCache', 'spawn_agent', 'self_patch', 'get_pod_status', 'check_network_connectivity', 'watch_k8s_events', 'watch_k8s_audit_events', 'microsoft_autonomous_remediation', 'collect_imagepull_forensics', 'analyze_imagepull_failure', 'execute_imagepull_remediation', 'reply_to_user', 'remember_event', 'search_memory', 'capture_system_screenshot', 'tune_hyperparameters', "export_system_state_tool","kubernetes_pod_metrics_tool","find_wasteful_deployments_tool","initiate_network_scan_tool","deploy_recovery_protocol_tool","analyze_threat_signature_tool","isolate_network_segment_tool","extract_iocs_tool","hash_text_tool","get_tool_usage_stats_tool","tool_health_check_tool","update_world_model_tool","query_long_term_memory_tool","analyze_text_sentiment_tool","prometheus_query_tool","prometheus_range_query_tool","broadcast_announcement_tool","create_pdf_tool","shuffle_roles_and_tasks_tool","list_available_tools_tool","system_get_cpu_load","system_get_memory_usage","system_get_disk_usage","top_processes_tool","measure_responsiveness_tool","get_environmental_data_tool","web_search_tool","update_resource_allocation_tool","long_sleep_tool","system_diagnostics_tool","self_test_tool","restart_agent_tool","read_webpage_tool","send_desktop_notification_tool","redact_pii_tool","check_calendar_tool","execute_terminal_command","write_sandbox_file","send_email", "k8s_get_pod_logs", "k8s_get_pod_status", "k8s_describe_pod", "k8s_rollout_restart", "k8s_patch_deployment_env", "k8s_patch_resource_limits", "system_check_port", "system_tail_log_file", "system_restart_allowed_service", "get_system_cpu_load_tool", "get_system_resource_usage_tool", "disk_usage_tool"]
__all__ += [
    "k8s_patch_deployment_image",
    "k8s_patch_probe",
    "k8s_rollout_undo",
]
