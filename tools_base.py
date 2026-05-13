# tools_base.py — Base utilities for all CVA tools
from __future__ import annotations

import os
import time
import shutil
import logging
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger("CatalystLogger")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
                check=False,
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
