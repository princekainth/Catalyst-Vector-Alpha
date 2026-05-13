"""
Self-Healing Monitor for Catalyst Vector Alpha

This service monitors the health of all registered tools and listens for
recurring failures that might indicate bit-rot or evolutionary defects.
It triggers REPAIR missions for the Evolution Agent when tools are broken.
"""

import time
import threading
import logging
import traceback
from typing import Optional, Any, Dict, List
from datetime import datetime

class SelfHealingMonitor:
    def __init__(
        self,
        tool_registry: Any,
        evolution_agent: Any,
        message_bus: Optional[Any] = None,
        log_sink: Optional[Any] = None,
        check_interval: int = 600,  # 10 minutes by default
    ):
        self.tool_registry = tool_registry
        self.evolution_agent = evolution_agent
        self.message_bus = message_bus
        self.log_sink = log_sink or logging.getLogger("SelfHealingMonitor")
        self.check_interval = check_interval
        
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check_ts = 0.0
        
        # Track tools we've already sent for repair to avoid spamming
        self.repair_sent: Dict[str, float] = {}

    def start(self):
        """Start the monitor loop."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.log_sink.info("[SelfHealing] Started self-healing monitor loop")

    def stop(self):
        """Stop the monitor loop."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.log_sink.info("[SelfHealing] Stopped self-healing monitor loop")

    def _run_loop(self):
        """Background loop for health checks."""
        while self.running:
            try:
                self.perform_health_scan()
            except Exception as e:
                self.log_sink.error("[SelfHealing] Error in health scan: %s", e)
                traceback.print_exc()
            
            time.sleep(self.check_interval)

    def perform_health_scan(self):
        """Scan all tools for failures or broken circuit breakers."""
        self.log_sink.info("[SelfHealing] 🩹 Starting proactive tool health scan...")
        
        # 1. Check ToolRegistry circuit breakers
        if hasattr(self.tool_registry, "_broken_until"):
            with self.tool_registry._failure_lock:
                broken_tools = list(self.tool_registry._broken_until.keys())
            
            for tool_name in broken_tools:
                self._report_broken_tool(tool_name, "Circuit breaker tripped due to repeated failures.")

        # 2. Check for recent task failures in logs (if possible)
        # For now, we rely on the ToolRegistry pushing gaps, but we can also
        # proactively "ping" critical tools.
        
        critical_tools = ["disk_usage", "web_search", "write_sandbox_file", "execute_terminal_command"]
        for tool in critical_tools:
            if self.tool_registry.has_tool(tool):
                self._ping_tool(tool)

        self._last_check_ts = time.time()

    def _ping_tool(self, tool_name: str):
        """Verify a tool can at least be invoked without a crash/syntax error."""
        try:
            # We use validate_args with empty dict to see if it even resolves the tool
            # or if the validator/normalizer crashes.
            is_valid = self.tool_registry.validate_args(tool_name, {})
            # If it returns False, it just means args are missing, which is a GOOD sign (tool exists).
            # If it raises an exception, the tool is definitely broken.
        except Exception as e:
            self.log_sink.warning("[SelfHealing] ❌ Tool '%s' crashed during validation: %s", tool_name, e)
            self._report_broken_tool(tool_name, f"Tool crashed during basic validation: {str(e)}")

    def _report_broken_tool(self, tool_name: str, reason: str):
        """Send a repair mission to the Evolution Agent."""
        now = time.time()
        # Cooldown: Don't re-report the same tool for 1 hour
        if tool_name in self.repair_sent and (now - self.repair_sent[tool_name]) < 3600:
            return

        self.log_sink.info("[SelfHealing] 🔧 Reporting broken tool '%s' for repair: %s", tool_name, reason)
        
        if self.evolution_agent and hasattr(self.evolution_agent, "record_capability_gap"):
            self.evolution_agent.record_capability_gap(
                description=f"REPAIR: Tool '{tool_name}' is broken. {reason}",
                context=f"Proactive health check detected failure. Stack trace analysis required.",
                attempted_tool=tool_name,
                failure_reason="broken_tool",
                source_agent="SelfHealingMonitor"
            )
            self.repair_sent[tool_name] = now

# Export
__all__ = ["SelfHealingMonitor"]
