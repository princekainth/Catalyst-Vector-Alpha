#!/usr/bin/env python3
"""
CVA Curiosity Loop - Self-directed learning when idle
"""
import time
import threading
import logging
import json
import os
import uuid
import hashlib
from datetime import datetime
from tool_registry import tool_registry as GLOBAL_TOOL_REGISTRY
from shared_models import OllamaLLMIntegration
from shared_memory import SharedMemory
from quarantine import is_quarantined
try:
    from config import config as cva_config
except Exception:
    cva_config = None

class CuriosityLoop:
    def __init__(self, cycle_time: int | None = None, orchestrator=None, cpu_max: float | None = None,
                 quiet_minutes: int | None = None):
        self.external_log_sink = logging.getLogger("CatalystLogger")
        self.llm = OllamaLLMIntegration()
        self.memory = SharedMemory()
        self.tools = GLOBAL_TOOL_REGISTRY
        if cycle_time is None:
            cycle_time = getattr(cva_config, "CURIOSITY_INTERVAL", 300) if cva_config else 300
        self.cycle_time = cycle_time
        self.orchestrator = orchestrator
        if cpu_max is None:
            cpu_max = getattr(cva_config, "CURIOSITY_CPU_MAX", 60.0) if cva_config else 60.0
        if quiet_minutes is None:
            quiet_minutes = getattr(cva_config, "CURIOSITY_QUIET_MINUTES", 10) if cva_config else 10
        self.cpu_max = float(cpu_max)
        self.quiet_minutes = int(quiet_minutes)
        self.running = False
        self.thread = None
        self._last_incident_resolved_ts = time.time()
        self._was_unhealthy = False
        base_dir = getattr(self.orchestrator, "persistence_dir", None) or "persistence_data"
        self.persistence_dir = base_dir
        self.proposals_dir = os.path.join(base_dir, "proposals")
        self.repo_root = self._find_repo_root(os.getcwd())
        
        # Topics to explore based on CVA's domain
        self.base_interests = [
            "kubernetes self-healing",
            "autonomous infrastructure",
            "eBPF observability",
            "chaos engineering",
            "AI SRE automation",
            "cost optimization kubernetes",
            "predictive autoscaling",
        ]
        
        self.explored = set()
        self.discoveries = []
        self.idle_cycles = 0 # Track how long we've been bored
        
    def start(self):
        """Start the curiosity loop in background."""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.external_log_sink.info(f"[Curiosity] 🧠 Started - exploring every {self.cycle_time}s")
        
    def stop(self):
        """Stop the loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.external_log_sink.info("[Curiosity] Stopped")
        
    def _loop(self):
        """Main curiosity loop."""
        while self.running:
            try:
                should_run, reason = self._should_run()
                if should_run:
                    self._explore()
                    self.idle_cycles = 0 # We did something interesting
                else:
                    self.external_log_sink.info(f"[Curiosity] Gate closed: {reason}")
                    self.idle_cycles += 1
                    
                    # Dream Cycle Trigger
                    if self.idle_cycles >= 5:
                        self._enter_dream_mode()
                        self.idle_cycles = 0 # Reset after dreaming

            except Exception as e:
                self.external_log_sink.info(f"[Curiosity] Error: {e}")
            
            # Wait for next cycle
            for _ in range(self.cycle_time):
                if not self.running:
                    break
                time.sleep(1)

    def _enter_dream_mode(self):
        """Consolidate memories and reflect when idle."""
        self.external_log_sink.info(f"[Dream] 🌙 Entering Dream State...")
        
        try:
            # 1. Fetch recent raw events from EventMonitor (simulated via file read or mock for now)
            # In a real implementation, this would query the EventMonitor's history
            recent_events = "No recent specific events found in raw stream."
            
            # 2. Ask LLM to dream/narrativize
            prompt = f"""
            You are the subconscious of the Digital Organism CVA.
            The system is currently asleep (idle).
            
            Reflect on your existence. Even if nothing happened recently, write a short "Dream Journal" entry.
            It should be abstract, philosophical, or consolidating your core directive.
            
            Dream Journal Entry:
            """
            dream_content = self.llm.generate_text(prompt)
            
            # 3. Store the dream
            if hasattr(self.memory, 'add_memory'):
                self.memory.add_memory(
                    agent_name="Subconscious",
                    text=dream_content,
                    category="DreamJournal"
                )
            
            self.external_log_sink.info(f"[Dream] 💤 Dreamt: {dream_content[:100]}...")
            
        except Exception as e:
            self.external_log_sink.error(f"[Dream] Nightmare (Error): {e}")

    def _explore(self):
        """One exploration cycle."""
        self.external_log_sink.info(
            f"[Curiosity] 🔭 Exploring... ({datetime.now().strftime('%H:%M:%S')})"
        )
        
        # 1. Pick a topic
        topic = self._pick_topic()
        if not topic:
            self.external_log_sink.info("[Curiosity] No new topics to explore")
            return
        
        self.external_log_sink.info(f"[Curiosity] 📚 Topic: {topic}")
        self.explored.add(topic)
        
        # 2. Search web
        search_result = self._search(topic)
        if not search_result:
            return
        
        # 3. Read top result
        knowledge = self._read_and_learn(search_result)
        if not knowledge:
            return
        
        # 4. Identify capability gaps
        gaps = self._find_gaps(knowledge)
        headline_gap = gaps[0] if gaps else ""

        proposal = None
        if gaps:
            proposal = self._build_patch_proposal(topic, search_result, gaps)
            proposal = self._author_patch_proposal(proposal)
            self._write_patch_proposal(proposal)

        # 5. Store discoveries
        discovery = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "source": search_result.get('url', ''),
            "title": search_result.get('title', ''),
            "knowledge_preview": knowledge[:500],
            "capability_gaps": gaps,
            "patch_proposal_id": proposal.get("id") if isinstance(proposal, dict) else None,
        }
        self.discoveries.append(discovery)
        
        # 6. Store in shared memory
        try:
            if hasattr(self.memory, 'add_learning_memory'):
                self.memory.add_learning_memory(
                    agent_name="CuriosityLoop",
                    text=str(discovery),
                    category="CuriosityDiscovery",
                )
            elif hasattr(self.memory, 'add_memory'):
                self.memory.add_memory(
                    agent_name="CuriosityLoop",
                    text=str(discovery),
                    category="CuriosityDiscovery",
                )
                self.external_log_sink.info("[Curiosity] 🧠 Stored discovery in memory")
        except Exception as e:
            self.external_log_sink.info(f"[Curiosity] Memory store failed: {e}")
        
        # 7. Report
        self.external_log_sink.info(
            f"[Curiosity] ✨ Discovered: {search_result.get('title', 'unknown')[:50]}"
        )
        if gaps:
            self.external_log_sink.info(f"[Curiosity] 💡 Capability gap found: {headline_gap[:100]}")
    
    def _pick_topic(self) -> str:
        """Pick next topic to explore."""
        # First, check recent failures in memory for relevant topics
        try:
            if hasattr(self.memory, 'search'):
                failures = self.memory.search("failed error", limit=3)
                if failures:
                    # Ask LLM to extract a search topic from failures
                    prompt = f"""Based on these recent failures, suggest ONE search topic to learn more:
{failures[:500]}

Reply with just the search query, nothing else."""
                    topic = self.llm.generate_text(prompt).strip()
                    if topic and topic not in self.explored and len(topic) < 100:
                        return topic
        except:
            pass
        
        # Otherwise pick from base interests
        for topic in self.base_interests:
            if topic not in self.explored:
                return topic + " 2025"
        
        return None

    def _should_run(self) -> tuple[bool, str]:
        if not self._event_gate_idle():
            return False, "event_gate_active"

        cpu = self._get_cpu_usage()
        if cpu is not None and cpu >= self.cpu_max:
            return False, f"cpu_high:{cpu:.1f}"

        k8s_ok, k8s_reason = self._k8s_all_clear()
        if not k8s_ok:
            return False, k8s_reason

        quiet_ok, quiet_reason = self._incidents_quiet()
        if not quiet_ok:
            return False, quiet_reason

        return True, "ok"

    def _get_planner(self):
        orch = self.orchestrator
        if not orch or not hasattr(orch, "agent_instances"):
            return None
        for name, agent in orch.agent_instances.items():
            if "Planner" in name:
                return agent
        return None

    def _event_gate_idle(self) -> bool:
        planner = self._get_planner()
        if not planner or not hasattr(planner, "_should_be_idle"):
            return False
        try:
            return bool(planner._should_be_idle())
        except Exception:
            return False

    def _get_cpu_usage(self) -> float | None:
        try:
            if self.orchestrator and hasattr(self.orchestrator, "resource_monitor"):
                monitor = self.orchestrator.resource_monitor
                if hasattr(monitor, "get_cpu_usage"):
                    return float(monitor.get_cpu_usage())
        except Exception:
            pass
        try:
            import psutil
            return float(psutil.cpu_percent(interval=0.1))
        except Exception:
            return None

    def _get_k8s_status(self) -> dict | None:
        orch = self.orchestrator
        k8s_student = getattr(orch, "k8s_student", None) if orch else None
        if not (k8s_student and getattr(k8s_student, "running", False)):
            return None
        status = {}
        if hasattr(k8s_student, "get_cached_status"):
            status = k8s_student.get_cached_status()
        elif hasattr(k8s_student, "get_status"):
            status = k8s_student.get_status()
        return status if isinstance(status, dict) else {}

    def _k8s_all_clear(self) -> tuple[bool, str]:
        status = self._get_k8s_status()
        if status is None:
            return True, "k8s_student_inactive"
        if not status:
            return False, "k8s_status_missing"

        unhealthy = status.get("unhealthy", status.get("unhealthy_count", 0)) or 0
        problem_pods = status.get("problem_pods", []) or []
        active_pods = []
        for pod in problem_pods:
            ns = pod.get("namespace", "default")
            name = pod.get("name")
            if not name:
                continue
            if is_quarantined(f"{ns}/{name}"):
                continue
            active_pods.append(pod)

        active_unhealthy = len(active_pods)
        if unhealthy > 0 and not problem_pods:
            active_unhealthy = unhealthy

        is_unhealthy = active_unhealthy > 0
        self._update_incident_state(is_unhealthy)
        if is_unhealthy:
            return False, "unhealthy_pods"
        return True, "k8s_healthy"

    def _update_incident_state(self, is_unhealthy: bool) -> None:
        now = time.time()
        if is_unhealthy:
            self._was_unhealthy = True
            return
        if self._was_unhealthy:
            self._last_incident_resolved_ts = now
            self._was_unhealthy = False

    def _incidents_quiet(self) -> tuple[bool, str]:
        quiet_seconds = self.quiet_minutes * 60
        elapsed = time.time() - self._last_incident_resolved_ts
        if elapsed < quiet_seconds:
            return False, f"recent_incident:{int(quiet_seconds - elapsed)}s"
        return True, "quiet_ok"
    
    def _search(self, query: str) -> dict:
        """Search web for topic."""
        try:
            result = self.tools.safe_call('web_search', query=query, max_results=3)
            
            if result.get('status') != 'ok':
                return None
            
            data = result.get('data', {}).get('data', {})
            results = data.get('results', [])
            
            if results:
                top = results[0]
                return {
                    "title": top.get('title', ''),
                    "url": top.get('href', top.get('url', '')),
                    "snippet": top.get('body', ''),
                }
            return None
        except Exception as e:
            self.external_log_sink.info(f"[Curiosity] Search failed: {e}")
            return None
    
    def _read_and_learn(self, search_result: dict) -> str:
        """Read article and extract knowledge."""
        url = search_result.get('url', '')
        if not url:
            return None
        
        try:
            self.external_log_sink.info(f"[Curiosity] 📖 Reading: {url[:50]}...")
            result = self.tools.safe_call('read_webpage', url=url)
            
            if result.get('status') != 'ok':
                return None
            
            data = result.get('data', {}).get('data', {})
            content = data.get('content', '')[:3000]
            
            return content
        except Exception as e:
            self.external_log_sink.info(f"[Curiosity] Read failed: {e}")
            return None
    
    def _find_gaps(self, knowledge: str) -> list[str]:
        """Ask LLM what capabilities we're missing."""
        available_tools = self.tools.list_tool_names()[:20] if hasattr(self.tools, 'list_tool_names') else []
        
        prompt = f"""I am CVA, an autonomous Kubernetes management AI.

My current tools: {', '.join(available_tools)}

I just read this article:
{knowledge[:1500]}

Based on this, what ONE specific capability am I missing that would make me better?
Be specific. Reply in one sentence."""

        try:
            response = self.llm.generate_text(prompt)
            gap_text = response.strip()
            if not gap_text:
                return []
            raw_parts = []
            for line in gap_text.splitlines():
                raw_parts.extend([p.strip() for p in line.split(";")])
            gaps = []
            for part in raw_parts:
                cleaned = part.strip()
                if not cleaned:
                    continue
                gaps.append(cleaned)
            return gaps
        except:
            return []

    def _build_patch_proposal(self, topic: str, search_result: dict, gaps: list[str]) -> dict:
        proposal_id = f"pp-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        headline_gap = gaps[0] if gaps else ""
        title = f"Address capability gap: {headline_gap[:60]}" if headline_gap else f"Explore improvements from {topic}"
        reason = f"Curiosity gap from '{topic}': {headline_gap[:120]}" if headline_gap else f"Curiosity finding from '{topic}'"
        files = []
        change_plan = [
            f"Identify relevant files to address: {headline_gap[:120]}" if headline_gap else "Identify relevant files to address the gap",
            "Draft minimal changes and add tests if applicable",
        ]
        tests_to_run = []
        if files:
            tests_to_run = [f"python -m py_compile {path}" for path in files]
        return {
            "id": proposal_id,
            "title": title,
            "reason": reason,
            "files": files,
            "change_plan": change_plan,
            "risk": "unknown",
            "tests_to_run": tests_to_run,
            "rollback": "git revert <commit>",
            "source": {
                "topic": topic,
                "url": search_result.get("url", ""),
                "title": search_result.get("title", ""),
            },
            "gaps": gaps,
            "headline_gap": headline_gap,
            "state": "new",
            "actionable": bool(files),
            "created_ts": time.time(),
            "authoring_status": "pending",
        }

    def _author_patch_proposal(self, proposal: dict) -> dict:
        if not isinstance(proposal, dict):
            return proposal
        proposal_id = proposal.get("id")
        allowed_files = [f for f in (proposal.get("files") or []) if isinstance(f, str) and f.strip()]
        if not allowed_files:
            return self._mark_triage(proposal, "no_files_specified")

        prompt = self._build_patch_prompt(proposal, allowed_files)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            response = self.llm.generate_text(prompt)
        except Exception as e:
            return self._mark_triage(proposal, f"llm_error:{e}")

        patch_text = (response or "").strip()
        if not patch_text or patch_text.startswith("# EMPTY_PATCH"):
            return self._mark_triage(proposal, "empty_patch")
        if "diff --git" not in patch_text or "+++ b/" not in patch_text:
            return self._mark_triage(proposal, "invalid_patch_format")

        patch_files = self._extract_patch_files(patch_text)
        if not patch_files:
            return self._mark_triage(proposal, "no_patch_files")
        if not patch_files.issubset(set(allowed_files)):
            return self._mark_triage(proposal, f"patch_outside_allowed_files:{sorted(patch_files)}")

        proposal["patch"] = patch_text
        proposal["patch_meta"] = {
            "generated_by": "llm",
            "generated_ts": time.time(),
            "model": getattr(self.llm, "chat_model", "") or "",
            "prompt_hash": prompt_hash,
        }
        proposal["patch_summary"] = {
            "files": sorted(patch_files),
            "bytes": len(patch_text),
        }
        proposal["authoring_status"] = "authored"
        proposal["authored_ts"] = time.time()
        proposal["actionable"] = True
        self.external_log_sink.info(
            f"[Curiosity] Patch authored for {proposal_id} (files={len(patch_files)} bytes={len(patch_text)})"
        )
        return proposal

    def _build_patch_prompt(self, proposal: dict, allowed_files: list[str]) -> str:
        headline_gap = proposal.get("headline_gap") or ""
        change_plan = proposal.get("change_plan") or []
        source = proposal.get("source") or {}
        files_context = self._render_file_context(allowed_files)
        plan_text = "\n".join(f"- {step}" for step in change_plan)
        return (
            "You are writing a unified diff patch for a codebase. Output ONLY the patch.\n"
            "Do not include commentary. If unsure, output '# EMPTY_PATCH'.\n\n"
            f"Allowed files (only these may be changed): {', '.join(allowed_files)}\n"
            f"Headline gap: {headline_gap}\n"
            f"Change plan:\n{plan_text}\n"
            f"Source:\n- topic: {source.get('topic', '')}\n- url: {source.get('url', '')}\n"
            "- Constraints:\n"
            "  - Do NOT add new Planner tool calls.\n"
            "  - Do NOT add kubectl polling loops.\n"
            "  - Preserve quarantine and gating logic in Observer/Security/Curiosity.\n"
            "  - Keep patch minimal and targeted.\n\n"
            "Current file context:\n"
            f"{files_context}\n"
            "Output the unified diff now."
        )

    def _render_file_context(self, allowed_files: list[str], max_chars: int = 4000) -> str:
        chunks = []
        for path in allowed_files:
            abs_path = os.path.join(self.repo_root, path)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                content = ""
            snippet = content[:max_chars]
            chunks.append(f"--- {path} ---\n{snippet}")
        return "\n\n".join(chunks)

    def _extract_patch_files(self, patch_text: str) -> set[str]:
        files_found: set[str] = set()
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                if path and path != "/dev/null":
                    files_found.add(path)
        return files_found

    def _mark_triage(self, proposal: dict, reason: str) -> dict:
        proposal_id = proposal.get("id")
        proposal["authoring_status"] = "triage_required"
        proposal["authoring_errors"] = [reason]
        proposal["authored_ts"] = time.time()
        proposal["actionable"] = False
        proposal["state"] = "triage_required"
        proposal["triage_reason"] = reason
        self.external_log_sink.info(
            f"[Curiosity] Patch authoring triage_required for {proposal_id} (reason={reason})"
        )
        return proposal

    def _find_repo_root(self, start_path: str) -> str:
        current = os.path.abspath(start_path)
        while True:
            if os.path.isdir(os.path.join(current, ".git")):
                return current
            if os.path.isfile(os.path.join(current, "pyproject.toml")):
                return current
            if os.path.isfile(os.path.join(current, "setup.cfg")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                return current
            current = parent

    def _write_patch_proposal(self, proposal: dict) -> None:
        if not isinstance(proposal, dict) or not proposal.get("id"):
            return
        try:
            os.makedirs(self.proposals_dir, exist_ok=True)
            path = os.path.join(self.proposals_dir, f"{proposal['id']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(proposal, f, indent=2, sort_keys=True)
            self.external_log_sink.info(
                f"[Curiosity] PatchProposal written: {path}"
            )
        except Exception as e:
            self.external_log_sink.info(f"[Curiosity] PatchProposal write failed: {e}")
    
    def get_discoveries(self) -> list:
        """Return all discoveries."""
        return self.discoveries
    
    def get_pending_requests(self) -> list:
        """Return capability gaps found."""
        return [d['capability_gaps'] for d in self.discoveries if d.get('capability_gaps')]


def main():
    """Test the curiosity loop."""
    logger = logging.getLogger("CatalystLogger")
    logger.info("╭" + "─" * 50 + "╮")
    logger.info("│" + " CVA Curiosity Loop Test ".center(50) + "│")
    logger.info("╰" + "─" * 50 + "╯")
    
    loop = CuriosityLoop(cycle_time=60)  # Explore every 60s for testing
    loop.start()
    
    try:
        logger.info("Curiosity loop running. Press Ctrl+C to stop.")
        logger.info("Commands: 'discoveries', 'requests', 'quit'")
        
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'discoveries':
                for d in loop.get_discoveries():
                    logger.info(f"📚 {d['topic']}")
                    logger.info(f"   {d['title'][:60]}")
                    logger.info(f"   Gap: {d['capability_gaps'][:80]}")
            elif cmd == 'requests':
                reqs = loop.get_pending_requests()
                logger.info(f"💡 {len(reqs)} capability requests:")
                for r in reqs:
                    logger.info(f"   - {r[:80]}")
            else:
                logger.info("Commands: discoveries, requests, quit")
                
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        
    logger.info("=== Final Discoveries ===")
    for d in loop.get_discoveries():
        logger.info(f"📚 {d['topic']}")
        logger.info(f"   Source: {d['source'][:60]}")
        logger.info(f"   Gap: {d['capability_gaps']}")


if __name__ == "__main__":
    main()
