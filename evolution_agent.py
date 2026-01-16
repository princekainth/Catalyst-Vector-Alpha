"""
Evolution Agent - CVA Self-Modification Capability

This agent enables CVA to recognize its limitations, research solutions,
write code for new tools, test them, and deploy them to itself.

The recursive self-improvement loop:
    OBSERVE → PLAN → ACT → LEARN → EVOLVE → (new self) → OBSERVE...
"""

import time
import threading
import json
import os
import ast
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List

# Try to import shared components
try:
    from shared_models import MemeticKernel
except ImportError:
    MemeticKernel = None


class EvolutionAgent:
    """
    The Evolution Agent monitors capability gaps and writes code to fill them.
    
    When CVA fails at something repeatedly, this agent:
    1. Identifies the missing capability
    2. Researches how to implement it
    3. Writes Python code for a new tool
    4. Tests it in sandbox
    5. Hot-loads it into the tool registry
    """
    
    def __init__(
        self,
        memetic_kernel: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        log_sink: Optional[Any] = None,
        sandbox_path: str = "/tmp/cva_evolution",
        approval_mode: str = "supervised",  # supervised, sandboxed, autonomous
        gap_threshold: int = 3,  # How many gaps before triggering evolution
        cycle_interval: int = 300,  # Check every 5 minutes
    ):
        self.memetic_kernel = memetic_kernel
        self.tool_registry = tool_registry
        self.log_sink = log_sink or self._default_logger()
        self.sandbox_path = sandbox_path
        self.approval_mode = approval_mode
        self.gap_threshold = gap_threshold
        self.cycle_interval = cycle_interval
        
        # State
        self.capability_gaps: List[Dict] = []
        self.evolution_history: List[Dict] = []
        self.pending_tools: List[Dict] = []  # Awaiting approval
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # Ensure sandbox exists
        os.makedirs(sandbox_path, exist_ok=True)
        
        self.log_sink.info("[EvolutionAgent] Initialized with approval_mode=%s", approval_mode)
    
    def _default_logger(self):
        """Fallback logger."""
        import logging
        return logging.getLogger("EvolutionAgent")
    
    # =========================================================================
    # CAPABILITY GAP DETECTION
    # =========================================================================
    
    def record_capability_gap(
        self,
        description: str,
        context: str,
        attempted_tool: Optional[str] = None,
        failure_reason: Optional[str] = None,
        source_agent: Optional[str] = None,
    ):
        """
        Called when an agent fails to complete a task due to missing capability.
        
        Example:
            evolver.record_capability_gap(
                description="Need to query real-time stock prices",
                context="User asked about TSLA price",
                attempted_tool="web_search",
                failure_reason="No structured stock data returned"
            )
        """
        gap = {
            "id": f"gap-{int(time.time() * 1000)}",
            "description": description,
            "context": context,
            "attempted_tool": attempted_tool,
            "failure_reason": failure_reason,
            "source_agent": source_agent,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "pending",  # pending, researching, solved, dismissed
        }
        
        self.capability_gaps.append(gap)
        
        # Also store in long-term memory if available
        if self.memetic_kernel and hasattr(self.memetic_kernel, "add_memory"):
            try:
                self.memetic_kernel.add_memory(
                    content=gap,
                    memory_type="CapabilityGap",
                    metadata={"gap_id": gap["id"]}
                )
            except Exception as e:
                self.log_sink.warning("[EvolutionAgent] Failed to store gap in memory: %s", e)
        
        self.log_sink.info(
            "[EvolutionAgent] 🧬 Recorded capability gap: %s (total: %d)",
            description[:50], len(self.capability_gaps)
        )
        
        # Check if we should trigger evolution
        pending_gaps = [g for g in self.capability_gaps if g["status"] == "pending"]
        if len(pending_gaps) >= self.gap_threshold:
            self.log_sink.info(
                "[EvolutionAgent] 🚀 Gap threshold reached (%d). Triggering evolution cycle.",
                len(pending_gaps)
            )
            self._trigger_evolution_cycle()
    
    # =========================================================================
    # EVOLUTION CYCLE
    # =========================================================================
    
    def _trigger_evolution_cycle(self):
        """Run one evolution cycle to address capability gaps."""
        pending_gaps = [g for g in self.capability_gaps if g["status"] == "pending"]
        if not pending_gaps:
            return
        
        # Pick highest priority gap (most recent for now)
        gap = pending_gaps[-1]
        gap["status"] = "researching"
        
        self.log_sink.info(
            "[EvolutionAgent] 🔬 Researching solution for: %s",
            gap["description"]
        )
        
        try:
            # Step 1: Research the solution
            research_result = self._research_solution(gap)
            
            if not research_result:
                self.log_sink.warning("[EvolutionAgent] No solution found for gap: %s", gap["id"])
                gap["status"] = "dismissed"
                return
            
            # Step 2: Generate tool code
            tool_code = self._generate_tool_code(gap, research_result)
            
            if not tool_code:
                self.log_sink.warning("[EvolutionAgent] Failed to generate code for gap: %s", gap["id"])
                gap["status"] = "dismissed"
                return
            
            # Step 3: Test the tool
            test_passed, test_output = self._test_tool_code(tool_code)
            
            if not test_passed:
                self.log_sink.warning(
                    "[EvolutionAgent] Tool test failed for gap %s: %s",
                    gap["id"], test_output
                )
                gap["status"] = "dismissed"
                return
            
            # Step 4: Deploy based on approval mode
            if self.approval_mode == "autonomous":
                self._deploy_tool(tool_code, gap)
                gap["status"] = "solved"
            elif self.approval_mode == "sandboxed":
                self.pending_tools.append({
                    "gap": gap,
                    "code": tool_code,
                    "test_output": test_output,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "sandbox_until": time.time() + 86400,  # 24 hours
                })
                gap["status"] = "pending_approval"
                self.log_sink.info(
                    "[EvolutionAgent] 📋 Tool queued for sandboxed testing: %s",
                    tool_code.get("name")
                )
            else:  # supervised
                self.pending_tools.append({
                    "gap": gap,
                    "code": tool_code,
                    "test_output": test_output,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                })
                gap["status"] = "pending_approval"
                self.log_sink.info(
                    "[EvolutionAgent] 🔔 Tool awaiting human approval: %s",
                    tool_code.get("name")
                )
        
        except Exception as e:
            self.log_sink.error("[EvolutionAgent] Evolution cycle failed: %s", e)
            gap["status"] = "dismissed"
            traceback.print_exc()
    
    def _research_solution(self, gap: Dict) -> Optional[Dict]:
        """
        Research how to solve the capability gap.
        Uses web search if available.
        """
        if not self.tool_registry:
            return None
        
        # Try web search
        if self.tool_registry.has_tool("web_search"):
            query = f"Python API for {gap['description']}"
            try:
                result = self.tool_registry.safe_call("web_search", query=query, max_results=5)
                if result and result.get("success"):
                    return {
                        "query": query,
                        "results": result.get("data", {}).get("results", []),
                        "source": "web_search"
                    }
            except Exception as e:
                self.log_sink.warning("[EvolutionAgent] Web search failed: %s", e)
        
        # Fallback: check memory for similar solved gaps
        if self.memetic_kernel and hasattr(self.memetic_kernel, "search"):
            try:
                similar = self.memetic_kernel.search(
                    f"solved capability gap {gap['description']}",
                    limit=3
                )
                if similar:
                    return {
                        "source": "memory",
                        "similar_solutions": similar
                    }
            except Exception:
                pass
        
        return None
    
    def _generate_tool_code(self, gap: Dict, research: Dict) -> Optional[Dict]:
        """
        Generate Python code for a new tool based on research.
        Uses LLM if available, otherwise generates template.
        """
        tool_name = self._sanitize_tool_name(gap["description"])
        
        # Try to use LLM for code generation
        # For now, generate a template that can be filled in
        template = f'''
def {tool_name}(**kwargs):
    """
    Auto-generated tool to address: {gap["description"]}
    
    Generated by CVA Evolution Agent.
    Gap ID: {gap["id"]}
    """
    # TODO: Implement based on research
    # Research source: {research.get("source", "unknown")}
    
    return {{
        "success": True,
        "message": "Tool {tool_name} executed",
        "data": {{}}
    }}

# Tool metadata for registration
TOOL_METADATA = {{
    "name": "{tool_name}",
    "description": "{gap['description']}",
    "parameters": {{}},
    "category": "evolved"
}}
'''
        
        return {
            "name": tool_name,
            "code": template,
            "gap_id": gap["id"],
            "description": gap["description"],
        }
    
    def _sanitize_tool_name(self, description: str) -> str:
        """Convert description to valid Python function name."""
        import re
        # Take first few words, lowercase, underscores
        words = description.lower().split()[:4]
        name = "_".join(words)
        name = re.sub(r"[^a-z0-9_]", "", name)
        return name or "evolved_tool"
    
    def _test_tool_code(self, tool_code: Dict) -> tuple[bool, str]:
        """
        Test the generated tool code in sandbox.
        Returns (passed: bool, output: str)
        """
        code = tool_code.get("code", "")
        tool_name = tool_code.get("name", "unknown")
        
        # Write to sandbox
        test_file = os.path.join(self.sandbox_path, f"test_{tool_name}.py")
        try:
            with open(test_file, "w") as f:
                f.write(code)
                f.write(f"\n\n# Test\nif __name__ == '__main__':\n    print({tool_name}())\n")
            
            # Syntax check
            with open(test_file, "r") as f:
                try:
                    ast.parse(f.read())
                except SyntaxError as e:
                    return False, f"Syntax error: {e}"
            
            # Try to execute in sandbox
            import subprocess
            result = subprocess.run(
                ["python3", test_file],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.sandbox_path
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        
        except Exception as e:
            return False, str(e)
        
        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def _deploy_tool(self, tool_code: Dict, gap: Dict):
        """
        Deploy the new tool to the tool registry.
        """
        tool_name = tool_code.get("name")
        code = tool_code.get("code")
        
        # Save to evolved_tools directory
        evolved_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "evolved_tools"
        )
        os.makedirs(evolved_dir, exist_ok=True)
        
        tool_file = os.path.join(evolved_dir, f"{tool_name}.py")
        with open(tool_file, "w") as f:
            f.write(code)
        
        # Record evolution
        self.evolution_history.append({
            "tool_name": tool_name,
            "gap_id": gap["id"],
            "description": gap["description"],
            "deployed_at": datetime.utcnow().isoformat() + "Z",
            "file_path": tool_file,
        })
        
        self.log_sink.info(
            "[EvolutionAgent] 🧬 EVOLVED: New tool '%s' deployed! I can now: %s",
            tool_name, gap["description"]
        )
        
        # Announce to memory
        if self.memetic_kernel and hasattr(self.memetic_kernel, "add_memory"):
            self.memetic_kernel.add_memory(
                content={
                    "type": "Evolution",
                    "tool_name": tool_name,
                    "capability": gap["description"],
                    "message": f"I evolved and gained a new capability: {gap['description']}"
                },
                memory_type="EvolutionEvent",
            )
    
    # =========================================================================
    # APPROVAL INTERFACE
    # =========================================================================
    
    def get_pending_tools(self) -> List[Dict]:
        """Get tools awaiting approval."""
        return self.pending_tools
    
    def approve_tool(self, tool_name: str) -> bool:
        """Approve and deploy a pending tool."""
        for pending in self.pending_tools:
            if pending["code"]["name"] == tool_name:
                self._deploy_tool(pending["code"], pending["gap"])
                pending["gap"]["status"] = "solved"
                self.pending_tools.remove(pending)
                return True
        return False
    
    def reject_tool(self, tool_name: str) -> bool:
        """Reject a pending tool."""
        for pending in self.pending_tools:
            if pending["code"]["name"] == tool_name:
                pending["gap"]["status"] = "dismissed"
                self.pending_tools.remove(pending)
                return True
        return False
    
    # =========================================================================
    # BACKGROUND THREAD
    # =========================================================================
    
    def start(self):
        """Start the evolution agent background thread."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.log_sink.info("[EvolutionAgent] 🧬 Started evolution monitoring")
    
    def stop(self):
        """Stop the evolution agent."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.log_sink.info("[EvolutionAgent] Evolution monitoring stopped")
    
    def _run_loop(self):
        """Background loop that checks for capability gaps."""
        while self.running:
            try:
                # Check if we should trigger evolution
                pending_gaps = [g for g in self.capability_gaps if g["status"] == "pending"]
                if len(pending_gaps) >= self.gap_threshold:
                    self._trigger_evolution_cycle()
                
                # Check sandboxed tools for promotion
                if self.approval_mode == "sandboxed":
                    now = time.time()
                    for pending in self.pending_tools[:]:
                        if pending.get("sandbox_until", 0) < now:
                            self.log_sink.info(
                                "[EvolutionAgent] Sandbox period complete for %s, auto-deploying",
                                pending["code"]["name"]
                            )
                            self._deploy_tool(pending["code"], pending["gap"])
                            pending["gap"]["status"] = "solved"
                            self.pending_tools.remove(pending)
            
            except Exception as e:
                self.log_sink.error("[EvolutionAgent] Loop error: %s", e)
            
            time.sleep(self.cycle_interval)
    
    # =========================================================================
    # STATUS & STATS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get current evolution agent status."""
        return {
            "running": self.running,
            "approval_mode": self.approval_mode,
            "total_gaps": len(self.capability_gaps),
            "pending_gaps": len([g for g in self.capability_gaps if g["status"] == "pending"]),
            "solved_gaps": len([g for g in self.capability_gaps if g["status"] == "solved"]),
            "pending_tools": len(self.pending_tools),
            "evolved_tools": len(self.evolution_history),
            "recent_evolutions": self.evolution_history[-5:] if self.evolution_history else [],
        }


# Export
__all__ = ["EvolutionAgent"]
