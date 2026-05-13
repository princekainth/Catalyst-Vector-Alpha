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
import inspect
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    from database import cva_db
except ImportError:
    cva_db = None

try:
    from config import config
except ImportError:
    config = None

# Try to import shared components
try:
    from shared_models import MemeticKernel
except ImportError:
    MemeticKernel = None

try:
    from cva_runtime.control_plane.evolution_recorder import EvolutionRecorder
except ImportError:
    EvolutionRecorder = None


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
        gap_threshold: Optional[int] = None,
        cycle_interval: Optional[int] = None,
        llm: Optional[Any] = None,
        db: Optional[Any] = None,
    ):
        self.memetic_kernel = memetic_kernel
        self.tool_registry = tool_registry
        self.llm = llm
        self.db = db or cva_db
        self.log_sink = log_sink or self._default_logger()
        self.sandbox_path = sandbox_path
        self.approval_mode = approval_mode
        self.gap_threshold = gap_threshold or (config.EVOLUTION_GAP_THRESHOLD if config else 3)
        self.cycle_interval = cycle_interval or (config.EVOLUTION_CYCLE_INTERVAL if config else 300)
        
        # State
        self.capability_gaps: List[Dict] = []
        self.evolution_history: List[Dict] = []
        self.pending_tools: List[Dict] = []  # Awaiting approval
        
        # Load from DB if available
        if self.db:
            try:
                state = self.db.load_evolution_state()
                self.capability_gaps = state.get("capability_gaps", [])
                self.evolution_history = state.get("evolution_history", [])
                self.pending_tools = state.get("pending_tools", [])
                if self.capability_gaps or self.evolution_history:
                    self.log_sink.info("[EvolutionAgent] Restored state: %d gaps, %d history items.", 
                                       len(self.capability_gaps), len(self.evolution_history))
            except Exception as e:
                self.log_sink.warning("[EvolutionAgent] Failed to load state from DB: %s", e)

        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # Ensure sandbox exists
        os.makedirs(sandbox_path, exist_ok=True)

        # Replayable eval harness
        self.recorder = EvolutionRecorder() if EvolutionRecorder else None
        
        self.log_sink.info("[EvolutionAgent] Initialized with approval_mode=%s, recorder=%s", approval_mode, bool(self.recorder))
    
    def _default_logger(self):
        """Fallback logger."""
        import logging
        return logging.getLogger("EvolutionAgent")
    
    def _persist_state(self):
        """Persist current gaps, history and pending tools to DB."""
        if not self.db:
            return
        try:
            self.db.save_evolution_state(self.capability_gaps, self.evolution_history, self.pending_tools)
        except Exception as e:
            self.log_sink.error("[EvolutionAgent] Failed to persist state: %s", e)
    
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
        
        # Persist state
        self._persist_state()

        # Also store in long-term memory if available
        if self.memetic_kernel and hasattr(self.memetic_kernel, "add_memory"):
            try:
                self.memetic_kernel.add_memory(
                    content=gap,
                    memory_type="CapabilityGap",
                )
            except Exception as e:
                self.log_sink.warning("[EvolutionAgent] Failed to store gap in memory: %s", e)
        
        self.log_sink.info(
            "[EvolutionAgent] 🧬 Recorded capability gap: %s (total: %d)",
            description[:50], len(self.capability_gaps),
            extra={"event_type": "CURIOSITY_GAP", "source": "EvolutionAgent", "description": f"Gap: {description[:100]}..."}
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
        
        # --- Recorder: start run ---
        tool_name_hint = self._sanitize_tool_name(gap["description"])
        run_id = None
        if self.recorder:
            try:
                run_id = self.recorder.start_run(tool_name_hint, gap.get("context", gap["description"]))
            except Exception:
                pass  # recorder failures must never block evolution

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
                if self.recorder and run_id:
                    self.recorder.finalize_run(run_id, "dismissed", "No research results")
                return
            
            # --- Recorder: research ---
            if self.recorder and run_id:
                try:
                    self.recorder.record_research(run_id, research_result)
                except Exception:
                    pass

            # Step 2 & 3: Generate and Test (Iterative Loop)
            tool_code = None
            errors = []
            max_retries = 3
            test_passed = False
            test_output = ""
            
            for attempt in range(max_retries):
                self.log_sink.info(f"[EvolutionAgent] 🧬 Generating code (Attempt {attempt+1}/{max_retries})...")
                
                # Check if this is a REPAIR mission
                is_repair = gap["description"].startswith("REPAIR:")
                existing_code = ""
                if is_repair and self.tool_registry:
                    tool_name = gap.get("attempted_tool")
                    if tool_name and self.tool_registry.has_tool(tool_name):
                        try:
                            tool_obj = self.tool_registry.get_tool(tool_name)
                            if tool_obj and hasattr(tool_obj, "func"):
                                existing_code = inspect.getsource(tool_obj.func)
                                self.log_sink.info(f"[EvolutionAgent] 🛠 Found existing code for repair: {tool_name}")
                        except Exception as e:
                            self.log_sink.warning(f"[EvolutionAgent] Could not retrieve source for {tool_name}: {e}")

                tool_code = self._generate_tool_code(gap, research_result, previous_errors=errors, existing_code=existing_code)
                
                if not tool_code:
                    errors.append("LLM failed to generate valid Python output.")
                    continue
                
                # --- Recorder: generated code ---
                if self.recorder and run_id:
                    try:
                        self.recorder.record_generated_code(run_id, tool_code.get("code", ""), attempt + 1)
                    except Exception:
                        pass

                test_passed, test_output = self._test_tool_code(tool_code)

                # --- Recorder: test result ---
                if self.recorder and run_id:
                    try:
                        self.recorder.record_test_result(run_id, attempt + 1, test_passed, test_output)
                    except Exception:
                        pass
                
                if test_passed:
                    self.log_sink.info(f"[EvolutionAgent] ✅ Tool verification passed on attempt {attempt+1}")
                    break
                else:
                    self.log_sink.warning(f"[EvolutionAgent] Tool test failed (Attempt {attempt+1}): {test_output}")
                    errors.append(test_output)
            
            if not tool_code or (errors and not test_passed):
                self.log_sink.error(f"[EvolutionAgent] Failed to evolve tool after {max_retries} attempts. Giving up.")
                gap["status"] = "dismissed"
                if self.recorder and run_id:
                    self.recorder.finalize_run(run_id, "dismissed", f"Failed after {max_retries} attempts")
                return
            
            # Step 4: Deploy to QUARANTINE (all modes go through quarantine now)
            self._deploy_tool(tool_code, gap, test_output=test_output)
            gap["status"] = "quarantined"

            # --- Recorder: finalize ---
            if self.recorder and run_id:
                try:
                    self.recorder.finalize_run(run_id, "quarantined", f"Tool '{tool_name_hint}' quarantined successfully")
                except Exception:
                    pass
        
        except Exception as e:
            self.log_sink.error("[EvolutionAgent] Evolution cycle failed: %s", e)
            gap["status"] = "dismissed"
            if self.recorder and run_id:
                self.recorder.finalize_run(run_id, "dismissed", f"Exception: {e}")
            traceback.print_exc()
        finally:
            self._persist_state()
    
    def _research_solution(self, gap: Dict) -> Optional[Dict]:
        """
        Research how to solve the capability gap.
        Uses web search if available.
        """
        if not self.tool_registry:
            return None
        
        # Clean up the description to get a better search query
        import re
        # Remove common prefixes like "I need a tool to", "Create a function that", etc.
        clean_desc = re.sub(r"(?i)^(i need|create|make|write|generate)\s+(a|an)\s+(tool|function|script)\s+(to|that|for)\s+", "", gap['description']).strip()
        
        # If the clean description is too short, keep the original (might be a specific command)
        if len(clean_desc) < 5:
            clean_desc = gap['description']

        # Try web search
        if self.tool_registry.has_tool("web_search"):
            query = f"python code to {clean_desc}"
            try:
                self.log_sink.info(f"[EvolutionAgent] 🔍 Searching: {query}")
                result = self.tool_registry.safe_call("web_search", query=query)
                
                # Even if "success" is False, we might have some data or we can just proceed to LLM with the intent
                search_results = []
                if result and isinstance(result.get("data"), dict):
                     search_results = result.get("data", {}).get("results", [])
                
                return {
                    "query": query,
                    "results": search_results,
                    "source": "web_search",
                    "intent": clean_desc
                }
            except Exception as e:
                self.log_sink.warning("[EvolutionAgent] Web search failed: %s", e)
                # Fallback: return just the intent so the LLM can try without search results
                return {"source": "fallback", "intent": clean_desc, "results": []}
        
        return {"source": "none", "intent": clean_desc, "results": []}
    
    def _generate_tool_code(self, gap: Dict, research: Dict, previous_errors: List[str] = None, existing_code: str = "") -> Optional[Dict]:
        """
        Generate Python code for a new tool or repair an existing one.
        """
        is_repair = gap["description"].startswith("REPAIR:")
        tool_name = self._sanitize_tool_name(gap["description"])
        if is_repair and gap.get("attempted_tool"):
            tool_name = gap["attempted_tool"]

        intent = research.get("intent", gap["description"])
        context_snippets = "\n".join([str(r) for r in research.get("results", [])[:3]])
        
        error_context = ""
        if previous_errors:
            error_context = "\n\nPREVIOUS VERSIONS FAILED WITH ERRORS:\n" + "\n".join([f"- {e}" for e in previous_errors]) + "\n\nPLEASE FIX THE CODE TO AVOID THESE ERRORS."
        
        repair_context = ""
        if is_repair and existing_code:
            repair_context = f"""
### EXISTING CODE TO REPAIR:
```python
{existing_code}
```
FAILURE REASON/CONTEXT:
{gap.get('context', 'Unknown error')}
{gap.get('failure_reason', '')}

INSTRUCTIONS FOR REPAIR:
1. Preserve the existing functionality but fix the bug.
2. If it's a syntax error, fix the formatting.
3. If it's a logic error (e.g., missing imports, wrong API endpoint), update the logic.
4. Keep the function signature compatible if possible.
"""

        prompt = f"""
You are an expert Python tool generator and debugger for an AI agent.
Your task is to write a complete, self-contained Python function that creates or repairs a tool to: "{intent}".

{repair_context}

CONTEXT FROM WEB SEARCH:
{context_snippets}
{error_context}

REQUIREMENTS:
1. Function Name: `{tool_name}`
2. Arguments: Use `**kwargs` or specific named arguments with type hints.
3. Return: Must return a dictionary with keys: `{{"success": bool, "message": str, "data": dict}}`.
4. Libraries: You can use `requests`, `json`, `datetime`, `random`, `math`, `bs4`, `urllib`.
5. Error Handling: Wrap EVERYTHING in try/except blocks. Return `success: False` on error.
6. Documentation: specific docstring describing what it does.
7. CRITICAL: PRIORITIZE FREE / NO-AUTH APIs. Use Open-Meteo for weather.
8. IMPORTS: All imports MUST be inside the function definition. Do not use top-level imports.

OUTPUT FORMAT:
Return ONLY the raw python code. 
Do NOT use Markdown code blocks (no ```python ... ```). 
Do NOT include any explanations or text outside the code.
Include the `TOOL_METADATA` dictionary at the end.
The dictionary MUST contain "name", "description", "parameters", and "category".
"""
        
        try:
            if self.llm and hasattr(self.llm, "generate_text"):
                self.log_sink.info("[EvolutionAgent] Using configured LLM integration for code generation")
                code = self.llm.generate_text(prompt, temperature=0.2)
            else:
                self.log_sink.info("[EvolutionAgent] No LLM provided; falling back to hardcoded mistral-nemo")
                import ollama
                response = ollama.chat(model='mistral-nemo', messages=[{'role': 'user', 'content': prompt}])
                code = response['message']['content']
            
            # Clean up markdown code blocks if the LLM adds them
            code = code.replace("```python", "").replace("```", "").strip()
            
            # Validate basic structure
            if "def " not in code or "TOOL_METADATA" not in code:
                raise ValueError("Generated code missing function definition or metadata")
                
            return {
                "name": tool_name,
                "code": code,
                "gap_id": gap["id"],
                "description": gap["description"],
            }
            
        except Exception as e:
            self.log_sink.error(f"[EvolutionAgent] LLM code generation failed: {e}")
            # Fallback to the old stub if LLM fails, so at least we don't crash the cycle
            return {
                "name": tool_name,
                "code": f"# LLM Generation Failed: {e}\n# Check logs.",
                "gap_id": gap["id"],
                "description": gap["description"],
            }

    def _sanitize_tool_name(self, description: str) -> str:
        """Convert description to valid Python function name."""
        import re
        # Remove common technical/log prefixes
        clean = re.sub(r"(?i)^(missing tool for task:|i need|create|make|write|generate)\s+", "", description)
        clean = re.sub(r"(?i)^(a|an)\s+(tool|function|script)\s+(to|that|for)\s+", "", clean)
        
        # Take the most meaningful part
        words = clean.lower().split()
        # Filter out very common words
        stopwords = {"the", "a", "an", "for", "me", "with", "and", "in", "at", "it", "must", "include"}
        keywords = [w for w in words if w not in stopwords][:4]
        
        name = "_".join(keywords)
        name = re.sub(r"[^a-z0-9_]", "", name)
        
        # Ensure it starts with a letter
        if not name or not name[0].isalpha():
            name = "tool_" + name
            
        return name or "evolved_tool"
    
    def _test_tool_code(self, tool_code: Dict) -> tuple[bool, str]:
        # Try to detect actual function name from code
        import re
        code = tool_code.get("code", "")
        tool_name = tool_code.get("name", "unknown")
        m = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", code)
        actual_name = m.group(1) if m else tool_name
        
        # Write to sandbox
        test_file = os.path.join(self.sandbox_path, f"test_{tool_name}.py")
        try:
            with open(test_file, "w") as f:
                f.write(code)
                # Call the detected function name but safely extract sample args if any
                f.write(f'''
# Test Runner
if __name__ == '__main__':
    import json
    
    # Check if TOOL_METADATA dictates any specific parameters (flexible extraction)
    test_kwargs = {{}}
    try:
        if "parameters" in TOOL_METADATA:
            params = TOOL_METADATA["parameters"]
            
            # Case 1: parameters is a flat dict with "required" list (OpenAI style)
            if isinstance(params.get("required"), list):
                req_names = params["required"]
            else:
                # Case 2: parameters matches keys, and "required": True is inside the spec
                req_names = []
                for p_name, p_spec in params.items():
                    if isinstance(p_spec, dict) and p_spec.get("required") is True:
                        req_names.append(p_name)
                    elif p_name == "properties": # Case 3: nested in properties
                        for sub_name, sub_spec in p_spec.items():
                             if isinstance(sub_spec, dict) and sub_spec.get("required") is True:
                                 req_names.append(sub_name)

            for r in req_names:
                r_lower = r.lower()
                if any(x in r_lower for x in ["url", "host", "domain"]):
                    test_kwargs[r] = "scanme.nmap.org"
                elif "ip" in r_lower:
                    test_kwargs[r] = "45.33.32.156"
                elif "path" in r_lower or "file" in r_lower:
                    test_kwargs[r] = "/tmp/test_file.txt"
                else:
                    test_kwargs[r] = "test_value"
    except Exception:
        pass
        
    try:
        result = {actual_name}(**test_kwargs)
        print(json.dumps(result))
    except TypeError as e:
        # Fallback if signature mismatch
        print(json.dumps({{"success": False, "message": f"Sandbox Test Signature Error: {{e}}" }}))
''')
            
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
    
    def _deploy_tool(self, tool_code: Dict, gap: Dict, test_output: str = ""):
        """
        Deploy a new tool to the QUARANTINE directory with a provenance manifest.

        Tools are NOT immediately available to agents. They must be promoted
        from quarantine/ to active/ via ``promote_tool_from_quarantine()``.
        In ``autonomous`` approval mode with ``allow_evolution_deploy`` the
        promotion happens automatically after sandbox tests pass.
        """
        tool_name = tool_code.get("name")
        code = tool_code.get("code")

        # --- Derive directories ---
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "evolved_tools",
        )
        quarantine_dir = os.path.join(base_dir, "quarantine")
        os.makedirs(quarantine_dir, exist_ok=True)

        # --- Write tool source ---
        tool_file = os.path.join(quarantine_dir, f"{tool_name}.py")
        with open(tool_file, "w") as f:
            f.write(code)

        # --- Detect required capabilities from code ---
        detected_caps = []
        code_lower = (code or "").lower()
        if any(kw in code_lower for kw in ["requests.get", "requests.post", "urllib", "http"]):
            detected_caps.append("net_outbound")
        if any(kw in code_lower for kw in ["open(", "os.write", "pathlib"]):
            detected_caps.append("file_write")
        if any(kw in code_lower for kw in ["subprocess", "os.system", "os.popen"]):
            detected_caps.append("shell_write")
        risk = "caution" if detected_caps else "safe"
        if "shell_write" in detected_caps:
            risk = "destructive"

        # --- Collect source URLs from research ---
        source_urls = []
        research = tool_code.get("_research") or {}
        for r in research.get("results", []):
            if isinstance(r, dict) and r.get("url"):
                source_urls.append(r["url"])
            elif isinstance(r, str) and r.startswith("http"):
                source_urls.append(r)

        # --- Build manifest ---
        manifest = {
            "tool_name": tool_name,
            "description": gap.get("description", ""),
            "gap_id": gap.get("id"),
            "source_directive": gap.get("context", ""),
            "source_urls": source_urls[:10],
            "risk_profile": risk,
            "detected_capabilities": detected_caps,
            "test_results": {
                "passed": True,
                "output": (test_output or "")[:2000],
            },
            "author_agent": gap.get("source_agent", "EvolutionAgent"),
            "approval_status": "quarantined",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "promoted_at": None,
        }
        manifest_file = os.path.join(quarantine_dir, f"{tool_name}.manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # --- Record in evolution history ---
        self.evolution_history.append({
            "tool_name": tool_name,
            "gap_id": gap["id"],
            "description": gap["description"],
            "quarantined_at": datetime.utcnow().isoformat() + "Z",
            "file_path": tool_file,
            "manifest_path": manifest_file,
            "status": "quarantined",
        })

        self.log_sink.info(
            "[EvolutionAgent] 🧬 QUARANTINED: Tool '%s' staged for review. Risk=%s, Caps=%s",
            tool_name, risk, detected_caps,
        )
        self._persist_state()

        # --- Auto-promote if allowed ---
        allow_auto = (
            self.approval_mode == "autonomous"
            and os.environ.get("CVA_ALLOW_EVOLUTION_DEPLOY", "0") == "1"
        )
        if allow_auto:
            self.log_sink.info("[EvolutionAgent] 🚀 Auto-promoting '%s' (autonomous + allow_evolution_deploy=1)", tool_name)
            self.promote_tool_from_quarantine(tool_name)
        else:
            self.log_sink.info(
                "[EvolutionAgent] 📋 Tool '%s' awaiting promotion (approval_mode=%s). "
                "Call promote_tool_from_quarantine('%s') or approve via API.",
                tool_name, self.approval_mode, tool_name,
            )

        # Announce to memory
        if self.memetic_kernel and hasattr(self.memetic_kernel, "add_memory"):
            self.memetic_kernel.add_memory(
                content={
                    "type": "Evolution",
                    "tool_name": tool_name,
                    "capability": gap["description"],
                    "status": "quarantined",
                    "risk": risk,
                    "message": f"I evolved and staged a new capability: {gap['description']}",
                },
                memory_type="EvolutionEvent",
            )

    # ------------------------------------------------------------------
    # QUARANTINE → ACTIVE PROMOTION
    # ------------------------------------------------------------------

    def promote_tool_from_quarantine(self, tool_name: str) -> bool:
        """Move a quarantined tool into active/ and reload the registry."""
        import shutil

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "evolved_tools",
        )
        q_dir = os.path.join(base_dir, "quarantine")
        a_dir = os.path.join(base_dir, "active")
        os.makedirs(a_dir, exist_ok=True)

        src_py = os.path.join(q_dir, f"{tool_name}.py")
        src_mf = os.path.join(q_dir, f"{tool_name}.manifest.json")

        if not os.path.isfile(src_py):
            self.log_sink.error("[EvolutionAgent] Cannot promote '%s': file not found in quarantine.", tool_name)
            return False

        # Update manifest
        if os.path.isfile(src_mf):
            try:
                with open(src_mf, "r") as f:
                    mf = json.load(f)
                mf["approval_status"] = "promoted"
                mf["promoted_at"] = datetime.utcnow().isoformat() + "Z"
                with open(src_mf, "w") as f:
                    json.dump(mf, f, indent=2)
            except Exception as exc:
                self.log_sink.warning("[EvolutionAgent] Manifest update failed: %s", exc)

        # Move files
        shutil.move(src_py, os.path.join(a_dir, f"{tool_name}.py"))
        if os.path.isfile(src_mf):
            shutil.move(src_mf, os.path.join(a_dir, f"{tool_name}.manifest.json"))

        self.log_sink.info(
            "[EvolutionAgent] ✅ PROMOTED: Tool '%s' moved to active/. Reloading registry.",
            tool_name,
        )

        # Reload the registry so the tool becomes callable
        if self.tool_registry and hasattr(self.tool_registry, "load_evolved_tools"):
            self.tool_registry.load_evolved_tools()

        # Update history entry
        for entry in reversed(self.evolution_history):
            if entry.get("tool_name") == tool_name:
                entry["status"] = "promoted"
                entry["promoted_at"] = datetime.utcnow().isoformat() + "Z"
                break
        
        self._persist_state()
        return True
    
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
                self._persist_state()
                return True
        return False
    
    def reject_tool(self, tool_name: str) -> bool:
        """Reject a pending tool."""
        for pending in self.pending_tools:
            if pending["code"]["name"] == tool_name:
                pending["gap"]["status"] = "dismissed"
                self.pending_tools.remove(pending)
                self._persist_state()
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
