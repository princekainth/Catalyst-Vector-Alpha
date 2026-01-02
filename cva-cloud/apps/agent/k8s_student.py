"""
K8s Student Agent - Autonomous Kubernetes remediation.

This student:
- Watches k8s cluster for issues
- Fixes problems autonomously
- Reports back to teachers via shared memory
- Never blocks main orchestration
"""

import subprocess
import json
import time
import re
from typing import Optional
from datetime import datetime

from base_student import BaseStudent
from reasoning_engine import get_reasoning_engine, ReasoningTrace
from quarantine import is_quarantined, set_quarantine


class K8sStudent(BaseStudent):
    """
    Autonomous K8s remediation agent.
    
    Handles:
    - CrashLoopBackOff (missing env vars, OOM)
    - CreateContainerConfigError (missing ConfigMap/Secret)
    - ImagePullBackOff (bad image tags)
    - Pending pods (scheduling issues)
    - OOMKilled (resource limits)
    """
    
    def __init__(
        self,
        shared_memory,
        tool_registry,
        cycle_time: int = 30,
        namespace: str = "all",
    ):
        super().__init__(
            name="K8sStudent",
            shared_memory=shared_memory,
            tool_registry=tool_registry,
            cycle_time=cycle_time,
        )
        self.namespace = namespace
        self.reasoning = get_reasoning_engine()
        
        # Track what we've already tried to fix
        self._remediated_pods: dict[str, float] = {}  # pod_key -> timestamp
        self._cooldown_seconds = 300  # Don't retry same pod for 5 min
        self._failed_attempts: dict[str, int] = {}  # deployment_key -> fail count
        self._max_attempts = 3  # Stop after 3 failures
        self._permanently_skip: set[str] = set()  # Pods we gave up on
        self._last_status: Optional[dict] = None
        
        # Severity order for prioritization
        self.SEVERITY = {
            "OOMKilled": 1,
            "CrashLoopBackOff": 1,
            "BackOff": 1,
            "Failed": 2,
            "CreateContainerConfigError": 2,
            "ImagePullBackOff": 3,
            "ErrImagePull": 3,
            "NotReady": 3,
            "Pending": 4,
        }
    
    # ─────────────────────────────────────────────────────────────
    # Main Work Loop
    # ─────────────────────────────────────────────────────────────
    
    def work(self) -> Optional[dict]:
        """
        Main work cycle:
        1. Get failing pods
        2. Prioritize by severity
        3. Fix each one
        4. Return summary
        """
        # 1. Get problem pods
        items = self._get_pods_snapshot()
        problem_pods = self._parse_problem_pods(items)
        all_pods = self._parse_all_pods(items)
        healthy = sum(1 for p in all_pods if p.get("ready"))
        unhealthy = len(all_pods) - healthy
        self._last_status = {
            "timestamp": time.time(),
            "total_pods": len(all_pods),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "unhealthy_count": unhealthy,
            "problem_pods": problem_pods,
            "namespace": self.namespace,
            "status": "ok",
        }
        
        if not problem_pods:
            return {"success": True, "action": "patrol", "details": {"message": "All pods healthy"}}
        
        # 2. Sort by severity
        sorted_pods = sorted(
            problem_pods,
            key=lambda p: self.SEVERITY.get(p.get("reason", ""), 5)
        )
        
        # 3. Fix each (up to 5 per cycle)
        fixed = []
        failed = []
        skipped = []
        
        for pod in sorted_pods[:5]:
            pod_key = f"{pod['namespace']}/{pod['name']}"
            
            # Check cooldown
            if self._in_cooldown(pod_key):
                skipped.append(pod_key)
                continue
            
            # Check if pod still exists
            if not self._pod_exists(pod['name'], pod['namespace']):
                skipped.append(f"{pod_key} (gone)")
                continue
            
            # Try to fix
            result = self._remediate_pod(pod)
            
            if result.get("planned"):
                fixed.append(pod_key)
                self._mark_remediated(pod_key)
            elif result.get("success"):
                fixed.append(pod_key)
                self._mark_remediated(pod_key)
            else:
                failed.append(pod_key)
        
        # 4. Return summary
        return {
            "success": len(fixed) > 0 or len(problem_pods) == 0,
            "action": "remediate",
            "details": {
                "total_problems": len(problem_pods),
                "fixed": fixed,
                "failed": failed,
                "skipped": skipped,
            }
        }
    
    def get_status(self) -> dict:
        """Return current K8s cluster status."""
        try:
            cached = self.get_cached_status()
            if cached:
                return cached

            items = self._get_pods_snapshot()
            pods = self._parse_all_pods(items)
            healthy = sum(1 for p in pods if p.get("ready"))
            unhealthy = len(pods) - healthy
            problem_pods = self._parse_problem_pods(items)
            status = {
                "total_pods": len(pods),
                "healthy": healthy,
                "unhealthy": unhealthy,
                "unhealthy_count": unhealthy,
                "problem_pods": problem_pods,
                "namespace": self.namespace,
                "status": "ok",
                "timestamp": time.time(),
            }
            self._last_status = status
            
            status.update({
                "remediated_this_session": self._success_count,
                "in_cooldown": len(self._remediated_pods),
            })
            return status
        except Exception as e:
            return {"error": str(e)}

    def get_cached_status(self, max_age: int = 120) -> dict:
        """Return cached status if it's fresh."""
        status = getattr(self, "_last_status", None)
        if not status:
            return {}
        ts = status.get("timestamp")
        if not ts or (time.time() - ts) > max_age:
            return {}
        return status
    
    # ─────────────────────────────────────────────────────────────
    # Pod Detection
    # ─────────────────────────────────────────────────────────────
    
    def _get_pods_snapshot(self) -> list:
        """Fetch raw pod items once per cycle."""
        try:
            cmd = ["kubectl", "get", "pods", "-A", "-o", "json"]
            if self.namespace != "all":
                cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            return data.get("items", []) or []
        except Exception as e:
            print(f"[{self.name}] Error getting pod snapshot: {e}")
            return []

    def _get_problem_pods(self, items: Optional[list] = None) -> list:
        """Get list of pods with issues."""
        try:
            if items is None:
                items = self._get_pods_snapshot()
            return self._parse_problem_pods(items)
        except Exception as e:
            print(f"[{self.name}] Error getting problem pods: {e}")
            return []

    def _parse_problem_pods(self, items: list) -> list:
        problems = []

        for item in items or []:
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            phase = status.get("phase", "")

            # Check container statuses
            container_statuses = status.get("containerStatuses", [])

            for cs in container_statuses:
                waiting = (cs.get("state", {}) or {}).get("waiting", {})
                reason = waiting.get("reason", "")

                # Also check terminated for OOMKilled
                terminated = (cs.get("state", {}) or {}).get("terminated", {})
                if terminated.get("reason") == "OOMKilled":
                    reason = "OOMKilled"

                if reason in self.SEVERITY:  # Only check actual current issues, not old restart counts
                    problems.append({
                        "name": metadata.get("name"),
                        "namespace": metadata.get("namespace", "default"),
                        "reason": reason or "CrashLoopBackOff",
                        "restarts": cs.get("restartCount", 0),
                        "message": waiting.get("message", ""),
                        "container": cs.get("name"),
                        "lastState": cs.get("lastState", {}) or {},
                    })

            # Detect Running but not Ready (probe failures or config issues)
            if phase == "Running":
                not_ready = any(not cs.get("ready", False) for cs in container_statuses)
                if not_ready:
                    cond_msg = ""
                    try:
                        for cond in status.get("conditions", []) or []:
                            if cond.get("type") in {"Ready", "ContainersReady"} and cond.get("status") == "False":
                                reason = cond.get("reason") or ""
                                message = cond.get("message") or ""
                                if reason or message:
                                    cond_msg = f"{reason}: {message}".strip(": ")
                                    break
                    except Exception:
                        cond_msg = ""
                    problems.append({
                        "name": metadata.get("name"),
                        "namespace": metadata.get("namespace", "default"),
                        "reason": "NotReady",
                        "restarts": sum(cs.get("restartCount", 0) for cs in container_statuses),
                        "message": cond_msg,
                        "container": container_statuses[0].get("name") if container_statuses else None,
                    })

            # Check for pending
            if phase == "Pending":
                problems.append({
                    "name": metadata.get("name"),
                    "namespace": metadata.get("namespace", "default"),
                    "reason": "Pending",
                    "restarts": 0,
                    "message": "",
                })

        # Dedupe by pod name
        seen = set()
        unique = []
        for p in problems:
            key = f"{p['namespace']}/{p['name']}"
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique
    
    def _get_all_pods(self, items: Optional[list] = None) -> list:
        """Get all pods with ready status."""
        try:
            if items is None:
                items = self._get_pods_snapshot()
            return self._parse_all_pods(items)
        except Exception:
            return []

    def _parse_all_pods(self, items: list) -> list:
        pods = []

        for item in items or []:
            metadata = item.get("metadata", {})
            status = item.get("status", {})

            ready = all(
                cs.get("ready", False)
                for cs in status.get("containerStatuses", [])
            )

            pods.append({
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "ready": ready,
                "phase": status.get("phase"),
            })

        return pods
    
    def _pod_exists(self, name: str, namespace: str) -> bool:
        """Check if pod still exists."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pod", name, "-n", namespace],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    # ─────────────────────────────────────────────────────────────
    # Remediation
    # ─────────────────────────────────────────────────────────────
    
    def _remediate_pod(self, pod: dict) -> dict:
        """Fix a single pod with full reasoning trace."""
        name = pod["name"]
        namespace = pod["namespace"]
        reason = pod["reason"]
        pod_key = f"{namespace}/{name}"
        deployment_name = self._get_deployment_name(name, namespace)
        track_key = f"{namespace}/{deployment_name}" if deployment_name else pod_key

        # Skip if quarantined globally
        if is_quarantined(pod_key):
            print(f"[{self.name}] Skipping quarantined pod {pod_key}")
            return {"success": False, "skipped": True, "reason": "quarantined"}

        # Skip if we've permanently given up
        if track_key in self._permanently_skip:
            print(f"[{self.name}] ⏭️ Permanently skipping {track_key}")
            return {"success": False, "skipped": True, "reason": "permanently_skipped"}

        # Skip if too many failures
        attempts = self._failed_attempts.get(track_key, 0)
        if attempts >= self._max_attempts:
            print(f"[{self.name}] ❌ Giving up on {track_key} after {attempts} failed attempts")
            self._permanently_skip.add(track_key)
            return {"success": False, "skipped": True, "reason": "max_attempts_exceeded"}

        # Start reasoning trace
        trace = self.reasoning.start_trace(
            agent_name=self.name,
            trigger=f"Pod {namespace}/{name} has {reason}",
            goal=f"Restore pod {name} to healthy state"
        )
        result = {"success": False, "error": "Unknown failure", "trace_id": trace.trace_id}

        try:
            # Step 1: Observe
            trace.observe(
                f"Detected {reason} on pod {namespace}/{name}",
                evidence=[f"Restarts: {pod.get('restarts', 0)}", f"Message: {pod.get('message', 'none')[:100]}"]
            )

            # Step 2: Get more context
            logs = self._get_pod_logs(name, namespace, tail=50)
            log_preview = logs[:200] if logs else "No logs available"
            trace.observe(f"Retrieved pod logs", evidence=[log_preview])
            events = self._get_pod_events(name, namespace, limit=5)
            events_preview = events[:400] if events else "No events available"
            trace.observe("Retrieved pod events", evidence=[events_preview])

            # Check for unfixable scenarios
            unfixable, unfixable_reason = self._is_unfixable(pod, logs)
            if unfixable:
                print(f"[{self.name}] 🚫 Pod {track_key} is unfixable: {unfixable_reason}")
                self._permanently_skip.add(track_key)
                set_quarantine(
                    pod_key,
                    status="UNFIXABLE",
                    until_ts=time.time() + (6 * 60 * 60),
                    reason=unfixable_reason,
                    source=self.name,
                )
                trace.verify(f"Pod is unfixable: {unfixable_reason}", success=False)
                self.reasoning.complete_trace(trace, "UNFIXABLE")
                return {"success": False, "unfixable": True, "reason": unfixable_reason, "trace_id": trace.trace_id}

            # Step 3: Analyze
            trace.analyze(
                f"Error type {reason} typically caused by: " + self._get_typical_causes(reason),
                confidence=0.8
            )

            # Step 4: Check memory for similar issues
            if hasattr(self, 'shared_memory') and self.shared_memory:
                similar = self.shared_memory.query_memory(f"fix {reason} kubernetes", n_results=2)
                if similar:
                    trace.recall(
                        f"Found {len(similar)} similar past incidents",
                        evidence=[m.get('text', '')[:80] for m in similar[:2]]
                    )

            # Step 5: Decide on action
            action_plan = self._decide_action_llm(reason, logs, pod)
            trace.decide(
                f"Will attempt: {action_plan['action']}",
                confidence=action_plan.get('confidence', 0.7),
                evidence=[action_plan.get('rationale', 'Standard remediation')]
            )

            # Step 6: Plan (no execution in agent loop)
            recommended_actions = self._build_recommended_actions(action_plan, pod, logs)
            self.reasoning.complete_trace(trace, "PLANNED")
            result = {
                "success": True,
                "planned": True,
                "action": action_plan.get("action"),
                "action_plan": action_plan,
                "recommended_actions": recommended_actions,
                "trace_id": trace.trace_id,
                "trace": trace.to_dict(),
            }

        except Exception as e:
            trace.verify(f"Exception: {str(e)[:100]}", success=False)
            self.reasoning.complete_trace(trace, f"ERROR: {str(e)[:50]}")
            result = {"success": False, "error": str(e)}

        # Track failure
        if not result.get("success"):
            self._failed_attempts[track_key] = self._failed_attempts.get(track_key, 0) + 1
            print(f"[{self.name}] 📊 Failure #{self._failed_attempts[track_key]} for {track_key}")
        else:
            # Reset on success
            self._failed_attempts.pop(track_key, None)

        return result

    def analyze(self, pod: dict) -> dict:
        """Return reasoning trace and planned action without executing."""
        plan = self._remediate_pod(pod)
        return {
            "trace": plan.get("trace", {}),
            "action_plan": plan.get("action_plan", {}),
            "recommended_actions": plan.get("recommended_actions", []),
            "action": plan.get("action"),
        }

    def execute_fix(self, incident_id: str, action_type: str, config: dict) -> dict:
        """Execute an approved fix based on action_type and config."""
        pod_name = (config or {}).get("pod_name") or (config or {}).get("name")
        namespace = (config or {}).get("namespace") or "default"
        message = (config or {}).get("message") or ""
        issue_type = (config or {}).get("issue_type") or (config or {}).get("reason") or ""
        pod = {"name": pod_name, "namespace": namespace, "message": message, "reason": issue_type}

        if not pod_name:
            return {"success": False, "error": "Missing pod name", "incident_id": incident_id}

        if action_type in ("fix_image_tag", "update_image_url"):
            image = (config or {}).get("image")
            if image:
                dep_name = self._get_deployment_owner(pod_name, namespace)
                if not dep_name:
                    return {"success": False, "error": "No deployment owner found", "incident_id": incident_id}
                return self._set_image(dep_name, namespace, image)
            return self._fix_image_pull(pod)

        if action_type in ("create_missing_config", "create_missing_secret"):
            cm_name = (config or {}).get("configmap")
            sec_name = (config or {}).get("secret")
            if cm_name:
                return self._create_configmap(cm_name, namespace)
            if sec_name:
                return self._create_secret(sec_name, namespace)
            return self._fix_config_error(pod)

        if action_type in ("add_missing_env_var",):
            dep_name = self._get_deployment_owner(pod_name, namespace)
            if not dep_name:
                return {"success": False, "error": "No deployment owner found", "incident_id": incident_id}
            var_name = (config or {}).get("env_name") or (config or {}).get("var_name") or "MISSING_ENV"
            var_value = (config or {}).get("env_value") or "changeme"
            return self._add_env_var(dep_name, namespace, var_name, var_value)

        if action_type in ("increase_memory_limits", "adjust_resources"):
            dep_name = self._get_deployment_owner(pod_name, namespace)
            if not dep_name:
                return {"success": False, "error": "No deployment owner found", "incident_id": incident_id}
            request = (config or {}).get("requests") or "256Mi"
            limit = (config or {}).get("memory") or "512Mi"
            return self._increase_memory(dep_name, namespace, request, limit)

        if action_type in ("restart_deployment", "generic_remediation"):
            dep_name = self._get_deployment_owner(pod_name, namespace)
            if not dep_name:
                return {"success": False, "error": "No deployment owner found", "incident_id": incident_id}
            try:
                result = subprocess.run(
                    ["kubectl", "rollout", "restart", f"deployment/{dep_name}", "-n", namespace],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return {"success": True, "action": "restart_deployment", "output": result.stdout}
                return {"success": False, "error": result.stderr}
            except Exception as e:
                return {"success": False, "error": str(e)}

        if action_type == "delete_pod":
            try:
                result = subprocess.run(
                    ["kubectl", "delete", "pod", pod_name, "-n", namespace],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return {"success": True, "action": "delete_pod", "output": result.stdout}
                return {"success": False, "error": result.stderr}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": f"Unsupported action: {action_type}"}

    def _get_typical_causes(self, reason: str) -> str:
        """Return typical causes for error types."""
        causes = {
            "CrashLoopBackOff": "missing env vars, OOM, startup crash, missing deps",
            "OOMKilled": "memory limit too low, memory leak, large payload",
            "CreateContainerConfigError": "missing ConfigMap, missing Secret, bad mount",
            "ImagePullBackOff": "wrong image tag, private registry, network issue",
            "ErrImagePull": "wrong image tag, private registry, network issue",
            "NotReady": "readiness/liveness probe failure or dependency not ready",
            "Pending": "insufficient resources, node selector, taints",
        }
        return causes.get(reason, "unknown cause")

    def _decide_action_llm(self, reason: str, logs: str, pod: dict) -> dict:
        """
        Use LLM to intelligently decide remediation action.
        Replaces hardcoded if/elif logic.
        """

        restarts = pod.get('restarts', 0)
        container = pod.get('container', 'unknown')
        message = pod.get('message', '')[:200]
        namespace = pod.get('namespace', 'default')
        name = pod.get('name', 'unknown')

        prompt = f"""You are a Kubernetes SRE expert diagnosing pod failures.

CONTEXT:
Pod: {namespace}/{name}
Error: {reason}
Restarts: {restarts}
Container: {container}
Error Message: {message}
Recent Logs: {logs[:400] if logs else 'No logs available'}

AVAILABLE ACTIONS:
1. fix_image_tag - Image name/tag is wrong or doesn't exist
2. increase_memory_limits - Pod needs more memory (OOMKilled)
3. add_missing_env_var - Missing required environment variables
4. restart_deployment - General restart to clear transient issues
5. create_missing_config - Missing ConfigMap or Secret
6. check_dependencies - Waiting for database/service dependency
7. check_security_context - Permission/security context issues
8. delete_pod - Pod is unfixable (bare pod with code bugs)
9. scale_deployment - Scale issues or resource contention

DECISION RULES:
- ImagePullBackOff/ErrImagePull → fix_image_tag
- OOMKilled or "137" in logs → increase_memory_limits
- "secret"/"password"/"env" in logs → add_missing_env_var
- ConfigMap/Secret errors → create_missing_config
- "connection refused" in logs → check_dependencies
- No logs + instant crash + no deployment → delete_pod
- CrashLoopBackOff with deployment → restart_deployment

Respond with ONLY valid JSON (no markdown, no backticks):
{{"action": "action_name", "confidence": 0.85, "rationale": "one sentence why"}}"""

        try:
            from llm import OllamaLLMIntegration
            llm = OllamaLLMIntegration()
            response = llm.generate_text(prompt=prompt, temperature=0.1, max_tokens=150, json_mode=True)

            response = response.strip()
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(l for l in lines if not l.startswith('```'))

            decision = json.loads(response.strip())

            required_keys = {'action', 'confidence', 'rationale'}
            if not all(k in decision for k in required_keys):
                raise ValueError("Missing required keys in LLM response")

            valid_actions = {
                'fix_image_tag', 'increase_memory_limits', 'add_missing_env_var',
                'restart_deployment', 'create_missing_config', 'check_dependencies',
                'check_security_context', 'delete_pod', 'scale_deployment',
                'generic_remediation'
            }

            if decision['action'] not in valid_actions:
                print(f"[{self.name}] ⚠️  LLM suggested unknown action: {decision['action']}, using generic")
                decision['action'] = 'generic_remediation'

            decision['confidence'] = max(0.0, min(1.0, float(decision['confidence'])))

            print(f"[{self.name}] 🧠 LLM Decision: {decision['action']} ({decision['confidence']:.0%}) - {decision['rationale']}")

            return decision

        except Exception as e:
            print(f"[{self.name}] ⚠️  LLM decision failed: {e}, falling back to rules")
            return self._decide_action_fallback(reason, logs, pod)

    def _decide_action_fallback(self, reason: str, logs: str, pod: dict) -> dict:
        """Fallback rule-based decision if LLM fails."""
        logs_lower = logs.lower() if logs else ""

        if reason in ["ImagePullBackOff", "ErrImagePull"]:
            return {
                "action": "fix_image_tag",
                "confidence": 0.7,
                "rationale": "Image pull error - likely wrong tag or registry"
            }

        if reason == "OOMKilled" or "137" in logs_lower or "oomkilled" in logs_lower:
            return {
                "action": "increase_memory_limits",
                "confidence": 0.85,
                "rationale": "Pod killed by OOM - needs more memory"
            }

        if reason == "CreateContainerConfigError":
            return {
                "action": "create_missing_config",
                "confidence": 0.9,
                "rationale": "ConfigMap or Secret is missing"
            }

        if reason == "CrashLoopBackOff":
            deployment_name = self._get_deployment_name(pod.get('name', ''), pod.get('namespace', 'default'))

            if not deployment_name and not logs:
                return {
                    "action": "delete_pod",
                    "confidence": 0.6,
                    "rationale": "Bare pod with no logs - likely code bug"
                }

            if "env" in logs_lower or "secret" in logs_lower:
                return {
                    "action": "add_missing_env_var",
                    "confidence": 0.7,
                    "rationale": "Logs suggest missing environment variable"
                }

            return {
                "action": "restart_deployment",
                "confidence": 0.3,
                "rationale": "No clear cause - trying restart"
            }

        return {
            "action": "generic_remediation",
            "confidence": 0.4,
            "rationale": f"Unknown error type: {reason}"
        }

    def _build_recommended_actions(self, action_plan: dict, pod: dict, logs: str) -> list[dict]:
        if not isinstance(action_plan, dict):
            return []
        action = action_plan.get("action")
        if not action:
            return []
        rationale = action_plan.get("rationale") or ""

        if action == "increase_memory_limits":
            return [{
                "action": "adjust_resources",
                "memory": "512Mi",
                "requests": "256Mi",
                "reason": rationale or "Increase memory limits",
            }]

        if action == "fix_image_tag":
            image = None
            msg = pod.get("message", "") or ""
            try:
                m = re.search(r'"([^"]+)"', msg)
                if m:
                    image = m.group(1)
            except Exception:
                image = None
            rec = {"action": "update_image_url", "reason": rationale or "Image pull error"}
            if image:
                rec["image"] = image
            return [rec]

        return [{
            "action": action,
            "reason": rationale,
        }]

    def _is_unfixable(self, pod: dict, logs: str) -> tuple[bool, str]:
        """Check if this pod is fundamentally unfixable."""
        logs_lower = logs.lower()
        message_lower = (pod.get("message", "") or "").lower()
        restarts = pod.get("restarts", 0)

        # OOM signal should be treated as fixable
        if "signal 9" in logs_lower:
            return False, ""

        # Container designed to exit
        if "exit 0" in logs_lower and "fail" not in logs_lower:
            return True, "Container completed successfully (Job?)"

        # No logs + multiple restarts suggests instant crash
        if (not logs or logs.strip() == "" or "no logs available" in logs_lower) and restarts >= 3:
            return True, "Container crashes instantly with no logs - likely code bug"

        # Explicit exit with error - likely code bug
        if "exit 1" in logs_lower and "stress" not in logs_lower:
            return True, "Container exits with error - likely code bug"

        # Init container stuck
        if pod.get("reason") == "Init:Error":
            return True, "Init container failed - needs code fix"

        if pod.get("reason") == "NotReady":
            if "readiness probe failed" in message_lower or "liveness probe failed" in message_lower:
                return True, "Probe failure indicates misconfiguration"

        return False, ""
    
    def _fix_crashloop(self, pod: dict) -> dict:
        """Fix CrashLoopBackOff - usually missing env vars."""
        name, namespace = pod["name"], pod["namespace"]
        
        # Get logs to diagnose
        logs = self._get_pod_logs(name, namespace)
        
        # Get deployment owner
        dep_name = self._get_deployment_owner(name, namespace)
        if not dep_name:
            return {"success": False, "error": "No deployment owner found"}
        
        # Check for common patterns
        if "POSTGRES_PASSWORD" in logs or "password" in logs.lower():
            return self._add_env_var(dep_name, namespace, "POSTGRES_PASSWORD", "changeme")
        elif "DATABASE_URL" in logs:
            return self._add_env_var(dep_name, namespace, "DATABASE_URL", "postgres://localhost:5432/db")
        elif "exit code 137" in logs.lower() or "oomkilled" in logs.lower():
            return self._increase_memory(dep_name, namespace, "512Mi", "1Gi")
        else:
            # Try increasing memory as default
            return self._increase_memory(dep_name, namespace, "256Mi", "512Mi")
    
    def _fix_config_error(self, pod: dict) -> dict:
        """Fix CreateContainerConfigError - missing ConfigMap or Secret."""
        name, namespace = pod["name"], pod["namespace"]
        message = pod.get("message", "")
        
        # Extract missing resource name
        cm_match = re.search(r'configmap "([^"]+)"', message, re.IGNORECASE)
        sec_match = re.search(r'secret "([^"]+)"', message, re.IGNORECASE)
        
        if cm_match:
            cm_name = cm_match.group(1)
            return self._create_configmap(cm_name, namespace)
        elif sec_match:
            sec_name = sec_match.group(1)
            return self._create_secret(sec_name, namespace)
        else:
            return {"success": False, "error": "Could not identify missing config"}
    
    def _fix_image_pull(self, pod: dict) -> dict:
        """Fix ImagePullBackOff - usually typo in image tag."""
        name, namespace = pod["name"], pod["namespace"]
        
        dep_name = self._get_deployment_owner(name, namespace)
        if not dep_name:
            return {"success": False, "error": "No deployment owner"}
        
        # Get current image
        try:
            result = subprocess.run(
                ["kubectl", "get", "deployment", dep_name, "-n", namespace, "-o", 
                 "jsonpath={.spec.template.spec.containers[0].image}"],
                capture_output=True, text=True, timeout=10
            )
            current_image = result.stdout.strip()
            
            # Try common fixes
            fixed_image = current_image
            if current_image.endswith("ttt"):  # nginx:latesttt -> nginx:latest
                fixed_image = current_image[:-2]
            elif ":latest" not in current_image and ":" not in current_image:
                fixed_image = f"{current_image}:latest"
            
            if fixed_image != current_image:
                return self._set_image(dep_name, namespace, fixed_image)
            
            return {"success": False, "error": "Could not determine correct image"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _fix_oom(self, pod: dict) -> dict:
        """Fix OOMKilled - increase memory limits."""
        name, namespace = pod["name"], pod["namespace"]
        
        dep_name = self._get_deployment_owner(name, namespace)
        if not dep_name:
            return {"success": False, "error": "No deployment owner"}
        
        return self._increase_memory(dep_name, namespace, "512Mi", "1Gi")
    
    def _fix_pending(self, pod: dict) -> dict:
        """Fix Pending - usually resource constraints. Just report for now."""
        return {"success": False, "error": "Pending pods require manual intervention"}
    
    # ─────────────────────────────────────────────────────────────
    # Fix Actions
    # ─────────────────────────────────────────────────────────────
    
    def _add_env_var(self, dep_name: str, namespace: str, var_name: str, var_value: str) -> dict:
        """Add environment variable to deployment."""
        try:
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": dep_name.split("-")[0],  # Best guess at container name
                                "env": [{"name": var_name, "value": var_value}]
                            }]
                        }
                    }
                }
            }
            
            result = subprocess.run(
                ["kubectl", "patch", "deployment", dep_name, "-n", namespace,
                 "--type", "strategic", "-p", json.dumps(patch)],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                print(f"[{self.name}] ✅ Added {var_name} to {dep_name}")
                return {"success": True, "action": f"Added env {var_name}"}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _increase_memory(self, dep_name: str, namespace: str, request: str, limit: str) -> dict:
        """Increase memory limits for deployment."""
        try:
            patch = {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": dep_name.split("-")[0],
                                "resources": {
                                    "requests": {"memory": request},
                                    "limits": {"memory": limit}
                                }
                            }]
                        }
                    }
                }
            }
            
            result = subprocess.run(
                ["kubectl", "patch", "deployment", dep_name, "-n", namespace,
                 "--type", "strategic", "-p", json.dumps(patch)],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                print(f"[{self.name}] ✅ Increased memory for {dep_name} to {limit}")
                return {"success": True, "action": f"Increased memory to {limit}"}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_configmap(self, name: str, namespace: str) -> dict:
        """Create a placeholder ConfigMap."""
        try:
            result = subprocess.run(
                ["kubectl", "create", "configmap", name, "-n", namespace,
                 "--from-literal=placeholder=true"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                print(f"[{self.name}] ✅ Created ConfigMap {name}")
                return {"success": True, "action": f"Created ConfigMap {name}"}
            elif "already exists" in result.stderr:
                return {"success": True, "action": f"ConfigMap {name} already exists"}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _create_secret(self, name: str, namespace: str) -> dict:
        """Create a placeholder Secret."""
        try:
            result = subprocess.run(
                ["kubectl", "create", "secret", "generic", name, "-n", namespace,
                 "--from-literal=placeholder=true"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                print(f"[{self.name}] ✅ Created Secret {name}")
                return {"success": True, "action": f"Created Secret {name}"}
            elif "already exists" in result.stderr:
                return {"success": True, "action": f"Secret {name} already exists"}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _set_image(self, dep_name: str, namespace: str, image: str) -> dict:
        """Set container image for deployment."""
        try:
            container_name = dep_name.split("-")[0]
            result = subprocess.run(
                ["kubectl", "set", "image", f"deployment/{dep_name}",
                 f"{container_name}={image}", "-n", namespace],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                print(f"[{self.name}] ✅ Set image to {image}")
                return {"success": True, "action": f"Set image to {image}"}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    
    def _get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str:
        """Get pod logs."""
        try:
            result = subprocess.run(
                ["kubectl", "logs", name, "-n", namespace, f"--tail={tail}"],
                capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        except:
            return ""

    def _get_pod_events(self, name: str, namespace: str, limit: int = 5) -> str:
        """Get recent events for a pod."""
        try:
            import json
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "events",
                    "-n",
                    namespace,
                    "--field-selector",
                    f"involvedObject.name={name}",
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return ""
            data = json.loads(result.stdout or "{}")
            items = data.get("items", []) or []
            if not items:
                return ""

            def _event_ts(item: dict) -> str:
                return (
                    item.get("lastTimestamp")
                    or item.get("eventTime")
                    or item.get("firstTimestamp")
                    or ""
                )

            items = sorted(items, key=_event_ts)
            lines = []
            for item in items[-limit:]:
                etype = item.get("type", "")
                reason = item.get("reason", "")
                message = item.get("message", "")
                line = f"{etype} {reason}: {message}".strip()
                lines.append(line[:200])
            return "\n".join(lines)
        except Exception:
            return ""
    
    def _get_deployment_owner(self, pod_name: str, namespace: str) -> Optional[str]:
        """Get deployment that owns this pod."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", 
                 "jsonpath={.metadata.ownerReferences[0].name}"],
                capture_output=True, text=True, timeout=10
            )
            rs_name = result.stdout.strip()
            
            if rs_name:
                # Get deployment from replicaset
                result = subprocess.run(
                    ["kubectl", "get", "rs", rs_name, "-n", namespace, "-o",
                     "jsonpath={.metadata.ownerReferences[0].name}"],
                    capture_output=True, text=True, timeout=10
                )
                return result.stdout.strip() or None
            return None
        except:
            return None

    def _get_deployment_name(self, pod_name: str, namespace: str) -> Optional[str]:
        """Get deployment name for tracking across pod restarts."""
        return self._get_deployment_owner(pod_name, namespace)
    
    def _in_cooldown(self, pod_key: str) -> bool:
        """Check if pod is in cooldown period."""
        if pod_key not in self._remediated_pods:
            return False
        last_time = self._remediated_pods[pod_key]
        return (time.time() - last_time) < self._cooldown_seconds
    
    def _mark_remediated(self, pod_key: str):
        """Mark pod as recently remediated."""
        self._remediated_pods[pod_key] = time.time()
        
        # Cleanup old entries
        now = time.time()
        self._remediated_pods = {
            k: v for k, v in self._remediated_pods.items()
            if (now - v) < self._cooldown_seconds * 2
        }

    # ─────────────────────────────────────────────────────────────
    # Web Search & Learning
    # ─────────────────────────────────────────────────────────────
    
    def _search_web_for_fix(self, reason: str, namespace: str, pod_name: str) -> Optional[dict]:
        """Search web for fix when local remediation fails."""
        if not self.tools:
            return None
        
        query = f"kubernetes {reason} fix solution"
        print(f"[{self.name}] 🔍 Searching: {query}")
        
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
                }
            return None
        except Exception as e:
            print(f"[{self.name}] Web search failed: {e}")
            return None
    
    def _apply_web_fix(self, url: str, reason: str, pod_name: str, namespace: str) -> Optional[dict]:
        """Fetch article, extract fix with LLM, apply it."""
        if not self.tools:
            return None
        
        try:
            # 1. Fetch article
            print(f"[{self.name}] 📖 Reading: {url[:50]}...")
            result = self.tools.safe_call('read_webpage', url=url)
            
            if not isinstance(result, dict) or result.get('status') != 'ok':
                return None
            
            outer_data = result.get('data', {})
            inner_data = outer_data.get('data', outer_data) if isinstance(outer_data, dict) else {}
            content = inner_data.get('content', '')[:3000]
            
            if not content:
                return None
            
            # 2. Ask LLM to extract kubectl fix
            from llm import OllamaLLMIntegration
            llm = OllamaLLMIntegration()
            
            # Get actual deployment name from Kubernetes
            dep_name = self._get_deployment_owner(pod_name, namespace)
            if not dep_name:
                # No deployment owner - can't fix with kubectl rollout/set
                print(f"[{self.name}] ⚠️ No deployment owner for {pod_name} - cannot auto-fix")
                return {"status": "no_deployment", "pod": pod_name}
            
            # Choose fix strategy based on error type
            if reason == "OOMKilled":
                fix_hint = f"kubectl set resources deployment/{dep_name} -n {namespace} --limits=memory=512Mi --requests=memory=256Mi"
            elif reason == "CrashLoopBackOff":
                fix_hint = f"kubectl rollout restart deployment/{dep_name} -n {namespace}"
            elif reason == "ImagePullBackOff":
                fix_hint = "NO_FIX (image issues need manual correction)"
            elif reason == "CreateContainerConfigError":
                fix_hint = "NO_FIX (missing configmap/secret needs manual creation)"
            else:
                fix_hint = f"kubectl rollout restart deployment/{dep_name} -n {namespace}"
            
            prompt = f"""Fix Kubernetes {reason} issue.

DEPLOYMENT: {dep_name}
NAMESPACE: {namespace}
ERROR TYPE: {reason}

RECOMMENDED FIX FOR {reason}:
{fix_hint}

If the recommended fix looks correct, return it exactly.
Otherwise say NO_FIX.

Reply ONLY the command or NO_FIX."""

            response = llm.generate_text(prompt)
            fix_command = response.strip() if response else ""
            
            print(f"[{self.name}] 🤖 LLM suggested: {fix_command[:80]}")
            
            if not fix_command or "NO_FIX" in fix_command or len(fix_command) < 10:
                return {"status": "no_fix_found"}
            
            # 3. Safety check - only allow certain kubectl commands
            allowed_prefixes = ['kubectl set', 'kubectl patch', 'kubectl scale', 'kubectl rollout']
            if not any(fix_command.startswith(p) for p in allowed_prefixes):
                print(f"[{self.name}] ⚠️ Command not in allowed list, skipping")
                return {"status": "unsafe_command", "command": fix_command}
            
            # 4. Apply the fix
            print(f"[{self.name}] 🔨 Applying: {fix_command}")
            import subprocess
            result = subprocess.run(fix_command, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"[{self.name}] ✅ Web fix applied successfully!")
                return {"status": "applied", "command": fix_command, "output": result.stdout}
            else:
                print(f"[{self.name}] ❌ Command failed: {result.stderr[:100]}")
                return {"status": "failed", "command": fix_command, "error": result.stderr}
            
        except Exception as e:
            print(f"[{self.name}] Web fix failed: {e}")
            return None

    def _pod_still_broken(self, pod_name: str, namespace: str) -> bool:
        """Check if pod is still in a broken state."""
        try:
            import json
            result = subprocess.run(
                ["kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return True
            pod_json = json.loads(result.stdout or "{}")
            status = pod_json.get("status", {}) or {}
            phase = status.get("phase", "")
            container_statuses = status.get("containerStatuses", []) or []
            if not container_statuses or phase != "Running":
                return True
            for cs in container_statuses:
                if not cs.get("ready", False):
                    return True
                state = cs.get("state", {}) or {}
                if "waiting" in state or "terminated" in state:
                    return True
            return False
        except Exception:
            return True  # Assume broken if we can't check
