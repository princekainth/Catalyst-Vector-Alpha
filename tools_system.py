# tools_system.py — Local System Control Adapter
from __future__ import annotations

import os
import shutil
import psutil
import socket
from typing import Any, Dict, Optional
from tools_base import standardize_response, SafeSubprocess

def system_get_disk_usage(path: str = "/") -> dict:
    """Get disk usage for a specific path."""
    if ".." in path:
        return standardize_response("error", error="Path traversal not allowed.")
    
    # Simple validation: path must exist
    if not os.path.exists(path):
        return standardize_response("error", error="Specified path does not exist.")
        
    try:
        usage = shutil.disk_usage(path)
        data = {
            "path": path,
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent_used": round((usage.used / usage.total) * 100, 2)
        }
        return standardize_response("ok", data=data)
    except Exception as e:
        return standardize_response("error", error=str(e))

def system_get_memory_usage() -> dict:
    """Get system memory usage."""
    try:
        mem = psutil.virtual_memory()
        data = {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent
        }
        return standardize_response("ok", data=data)
    except Exception as e:
        return standardize_response("error", error=str(e))

def system_get_cpu_load() -> dict:
    """Get system CPU load averages (1m, 5m, 15m)."""
    try:
        if hasattr(os, "getloadavg"):
            load = os.getloadavg()
            data = {
                "load_1m": round(load[0], 2),
                "load_5m": round(load[1], 2),
                "load_15m": round(load[2], 2)
            }
            return standardize_response("ok", data=data)
        else:
            # Fallback for systems without getloadavg
            cpu_percent = psutil.cpu_percent(interval=1)
            return standardize_response("ok", data={"cpu_percent": cpu_percent}, note="os.getloadavg not available, returned cpu_percent instead.")
    except Exception as e:
        return standardize_response("error", error=str(e))

def system_check_port(host: str = "127.0.0.1", port: int = 80, timeout: int = 2) -> dict:
    """Check if a local port is open."""
    if host not in ["127.0.0.1", "localhost"]:
        return standardize_response("error", error="Restriction: Only 127.0.0.1 or localhost allowed.")
        
    if not (1 <= port <= 65535):
        return standardize_response("error", error="Invalid port number.")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            is_open = (result == 0)
            return standardize_response("ok", data={"host": host, "port": port, "open": is_open})
    except Exception as e:
        return standardize_response("error", error=str(e))

def system_tail_log_file(path: str, lines: int = 100) -> dict:
    """Tail a log file from an allowed directory."""
    # Safety Check: Allowlist
    allowed_dirs = [
        os.path.abspath("./logs"),
        os.path.abspath("./.cva/logs"),
        os.path.abspath("/tmp/cva-demo-logs")
    ]
    
    target_path = os.path.abspath(path)
    
    is_allowed = False
    for d in allowed_dirs:
        if target_path.startswith(d):
            is_allowed = True
            break
            
    if not is_allowed:
        return standardize_response("error", error="Access denied. Path is not in the log allowlist.")
        
    if ".." in path:
        return standardize_response("error", error="Path traversal detected.")
        
    if not (1 <= lines <= 1000):
        return standardize_response("error", error="Line count must be between 1 and 1000.")

    if not os.path.exists(target_path):
        return standardize_response("error", error="Log file not found.")

    try:
        # Use SafeSubprocess to run tail
        success, stdout, stderr = SafeSubprocess.run(["tail", "-n", str(lines), target_path])
        if success:
            return standardize_response("ok", data={"path": path, "lines": lines, "content": stdout})
        else:
            return standardize_response("error", error=stderr)
    except Exception as e:
        return standardize_response("error", error=str(e))

def system_restart_allowed_service(service_name: str) -> dict:
    """Restart a service if it is in the CVA_ALLOWED_SERVICES list."""
    allowed_env = os.getenv("CVA_ALLOWED_SERVICES", "")
    allowed_services = [s.strip() for s in allowed_env.split(",") if s.strip()]
    
    if service_name not in allowed_services:
        return standardize_response("error", error=f"Service '{service_name}' is not authorized in CVA_ALLOWED_SERVICES.")

    if not SafeSubprocess.check_available("systemctl"):
        # For demo purposes, we might want to simulate success if systemctl is missing but we're in a test env
        return standardize_response("error", error="systemctl is not available on this system.")

    try:
        # Structured execution, no shell=True
        success, stdout, stderr = SafeSubprocess.run(["sudo", "systemctl", "restart", service_name])
        if success:
            return standardize_response("ok", data={"service": service_name, "status": "restarted", "output": stdout})
        else:
            return standardize_response("error", error=stderr)
    except Exception as e:
        return standardize_response("error", error=str(e))

def top_processes_tool(limit: int = 10, sort_by: str = "cpu") -> dict:
    """Get top processes by CPU or memory usage."""
    # Input validation
    limit = max(1, min(int(limit), 100))
    valid_sorts = {"cpu", "memory"}
    sort_by = sort_by.lower() if sort_by.lower() in valid_sorts else "cpu"
    
    try:
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "username", "cpu_percent", "memory_percent"]):
            try:
                procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort
        if sort_by == "memory":
            procs.sort(key=lambda x: x.get("memory_percent") or 0, reverse=True)
        else:
            procs.sort(key=lambda x: x.get("cpu_percent") or 0, reverse=True)
            
        return standardize_response("ok", data={"processes": procs[:limit]})
    except Exception as e:
        return standardize_response("error", error=str(e))

def measure_responsiveness_tool() -> dict:
    """Measure system responsiveness by timing a simple command."""
    import time
    start = time.time()
    success, _, _ = SafeSubprocess.run(["python3", "-c", "print(1)"])
    elapsed = time.time() - start
    return standardize_response("ok", data={"elapsed_seconds": round(elapsed, 4), "success": success})

def get_system_resource_usage_tool() -> dict:
    """Get comprehensive system resource usage."""
    cpu = system_get_cpu_load()
    mem = system_get_memory_usage()
    disk = system_get_disk_usage()
    return standardize_response("ok", data={
        "cpu": cpu.get("data"),
        "memory": mem.get("data"),
        "disk": disk.get("data")
    })
