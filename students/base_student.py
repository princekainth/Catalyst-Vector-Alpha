"""
Base Student Agent - All domain agents inherit from this.

Students:
- Run in their own thread (never block teachers)
- Have their own event loop
- Report results to shared memory
- Teachers can read reports anytime
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import datetime


class BaseStudent(ABC):
    """
    Base class for all student agents (domain-specific workers).
    
    Students are autonomous - they:
    1. Start in their own thread
    2. Run their own loop
    3. Do their specialized work
    4. Report back via shared memory
    5. Never block the main teacher orchestration
    """
    
    def __init__(
        self,
        name: str,
        shared_memory,          # ChromaDB or memory interface
        tool_registry,          # Access to tools
        cycle_time: int = 30,   # Seconds between work cycles
    ):
        self.name = name
        self.memory = shared_memory
        self.tools = tool_registry
        self.cycle_time = cycle_time
        
        # State
        self.running = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        self._last_result: Optional[dict] = None
        self._last_run: Optional[datetime] = None
        self._error_count = 0
        self._success_count = 0
        
        # Stats
        self.stats = {
            "started_at": None,
            "cycles_completed": 0,
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
        }
    
    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────
    
    def start(self):
        """Start the student in its own thread."""
        if self.running:
            print(f"[{self.name}] Already running")
            return
        
        self.running = True
        self.stats["started_at"] = datetime.now().isoformat()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"Student-{self.name}",
            daemon=True  # Dies when main program exits
        )
        self._thread.start()
        print(f"[{self.name}] 🎓 Student started")
    
    def stop(self):
        """Stop the student gracefully."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        print(f"[{self.name}] Student stopped")
    
    def pause(self):
        """Pause the student (skip work cycles)."""
        self.paused = True
        print(f"[{self.name}] Student paused")
    
    def resume(self):
        """Resume the student."""
        self.paused = False
        print(f"[{self.name}] Student resumed")
    
    # ─────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────
    
    def _loop(self):
        """Main loop - runs in separate thread."""
        print(f"[{self.name}] Entering work loop (cycle={self.cycle_time}s)")
        
        while self.running:
            try:
                # Skip if paused
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Do the work
                self._last_run = datetime.now()
                result = self.work()
                self._last_result = result
                self.stats["cycles_completed"] += 1
                
                # Report to teachers via shared memory
                if result:
                    self._report_to_teachers(result)
                
                # Track success
                if result and result.get("success"):
                    self._success_count += 1
                    self.stats["successful_actions"] += 1
                elif result:
                    self._error_count += 1
                    self.stats["failed_actions"] += 1
                
            except Exception as e:
                self._error_count += 1
                self.stats["failed_actions"] += 1
                print(f"[{self.name}] ❌ Error in work cycle: {e}")
                # Don't crash - keep running
            
            # Wait for next cycle
            time.sleep(self.cycle_time)
        
        print(f"[{self.name}] Exited work loop")
    
    # ─────────────────────────────────────────────────────────────
    # Abstract Methods - Students must implement
    # ─────────────────────────────────────────────────────────────
    
    @abstractmethod
    def work(self) -> Optional[dict]:
        """
        Do the student's specialized work.
        
        Returns:
            dict with at least:
            - success: bool
            - action: str (what was done)
            - details: dict (any extra info)
        
        Example:
            return {
                "success": True,
                "action": "fixed_pod",
                "details": {"pod": "postgres-xxx", "fix": "added env var"}
            }
        """
        pass
    
    @abstractmethod
    def get_status(self) -> dict:
        """
        Return current status for teachers to check.
        
        Returns:
            dict with domain-specific status info
        """
        pass
    
    # ─────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────
    
    def _report_to_teachers(self, result: dict):
        """Store result in shared memory so teachers can see."""
        try:
            if not self.memory:
                return
            
            report = {
                "student": self.name,
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "stats": self.stats.copy(),
            }
            
            # Support multiple memory interfaces
            # 1. MemeticKernel style (add_memory with memory_type, content)
            if hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    memory_type="StudentReport",
                    content=report,
                    source_agent=self.name,
                )
            # 2. SharedMemory style (add_memory with agent_name, text, category)
            elif hasattr(self.memory, "add_memory"):
                self.memory.add_memory(
                    agent_name=self.name,
                    text=str(report),
                    category="outcome",
                    metadata={"success": result.get("success", False)}
                )
            # 3. Generic store/add
            elif hasattr(self.memory, "store"):
                self.memory.store(content=str(report))
        except Exception as e:
            print(f"[{self.name}] Failed to report to teachers: {e}")
    
    # ─────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Safely call a tool from the registry."""
        if not self.tools:
            print(f"[{self.name}] No tool registry available")
            return None
        
        try:
            if hasattr(self.tools, "safe_call"):
                return self.tools.safe_call(tool_name, agent_id=self.name, **kwargs)
            
            # Fallbacks blocked for safety. Use safe_call exclusively.
            print(f"[{self.name}] 🛡️ SECURITY BLOCK: Direct tool call bypass attempted in BaseStudent for '{tool_name}'.")
            return None
        except Exception as e:
            print(f"[{self.name}] Tool call failed: {tool_name} - {e}")
            return None
    
    def recall_memory(self, query: str, limit: int = 5) -> list:
        """Search shared memory for past experiences."""
        if not self.memory:
            return []
        
        try:
            if hasattr(self.memory, "search"):
                return self.memory.search(query, limit=limit)
            elif hasattr(self.memory, "query"):
                return self.memory.query(query, n_results=limit)
        except Exception as e:
            print(f"[{self.name}] Memory recall failed: {e}")
            return []
    
    def __repr__(self):
        status = "running" if self.running else "stopped"
        if self.paused:
            status = "paused"
        return f"<{self.name} [{status}] cycles={self.stats['cycles_completed']} success={self._success_count} errors={self._error_count}>"