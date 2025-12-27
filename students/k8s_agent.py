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

from .base_student import BaseStudent


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
        
        # Track what we've already tried to fix
        self._remediated_pods: dict[str, float] = {}  # pod_key -> timestamp
        self._cooldown_seconds = 300  # Don't retry same pod for 5 min
        
        # Severity order for prioritization
        self.SEVERITY = {
            "OOMKilled": 1,
            "CrashLoopBackOff": 1,
            "BackOff": 1,
            "Failed": 2,
            "CreateContainerConfigError": 2,
            "ImagePullBackOff": 3,
            "ErrImagePull": 3,
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
        problem_pods = self._get_problem_pods()
        
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
            
            if result.get("success"):
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
            pods = self._get_all_pods()
            healthy = sum(1 for p in pods if p.get("ready"))
            unhealthy = len(pods) - healthy
            
            return {
                "total_pods": len(pods),
                "healthy": healthy,
                "unhealthy": unhealthy,
                "remediated_this_session": self._success_count,
                "in_cooldown": len(self._remediated_pods),
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ─────────────────────────────────────────────────────────────
    # Pod Detection
    # ─────────────────────────────────────────────────────────────
    
    def _get_problem_pods(self) -> list:
        """Get list of pods with issues."""
        try:
            cmd = ["kubectl", "get", "pods", "-A", "-o", "json"]
            if self.namespace != "all":
                cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return []
            
            data = json.loads(result.stdout)
            problems = []
            
            for item in data.get("items", []):
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
            
        except Exception as e:
            print(f"[{self.name}] Error getting problem pods: {e}")
            return []
    
    def _get_all_pods(self) -> list:
        """Get all pods with ready status."""
        try:
            cmd = ["kubectl", "get", "pods", "-A", "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return []
            
            data = json.loads(result.stdout)
            pods = []
            
            for item in data.get("items", []):
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
        except:
            return []
    
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
        """Fix a single pod using the proven microsoft_autonomous_remediation tool."""
        name = pod["name"]
        namespace = pod["namespace"]
        reason = pod["reason"]
        
        print(f"[{self.name}] 🔧 Fixing {namespace}/{name} ({reason})")
        
        try:
            # Use the battle-tested tool from main CVA
            if self.tools and hasattr(self.tools, 'safe_call'):
                result = self.tools.safe_call(
                    "microsoft_autonomous_remediation",
                    pod_name=name,
                    namespace=namespace,
                    recommended_actions=[]
                )
                
                # Check result
                if isinstance(result, dict):
                    data = result.get("data", result)
                    outcome = data.get("outcome", "UNKNOWN")
                    actions = data.get("actions_taken", data.get("actions", []))
                    
                    if outcome in ("SUCCESS", "PARTIAL_SUCCESS"):
                        print(f"[{self.name}] ✅ Fixed {namespace}/{name}: {actions}")
                        return {"success": True, "action": f"remediated_{reason}", "details": data}
                    else:
                        print(f"[{self.name}] ⚠️ Outcome {outcome} for {namespace}/{name}")
                        
                        # Try web search as fallback
                        web_fix = self._search_web_for_fix(reason, namespace, name)
                        if web_fix:
                            print(f"[{self.name}] 🌐 Found: {web_fix.get('title', '')[:50]}")
                            self._apply_web_fix(web_fix['url'], reason, name, namespace)
                        
                        return {"success": False, "error": f"Outcome: {outcome}"}
                        
                return {"success": False, "error": "Invalid result from tool"}
            else:
                return {"success": False, "error": "No tool registry available"}
                
        except Exception as e:
            print(f"[{self.name}] ❌ Error fixing {namespace}/{name}: {e}")
            return {"success": False, "error": str(e)}
    
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
            from shared_models import OllamaLLMIntegration
            llm = OllamaLLMIntegration()
            
            # Extract deployment name from pod
            dep_name = pod_name.rsplit('-', 2)[0] if '-' in pod_name else pod_name
            
            prompt = f"""Fix Kubernetes {reason} issue.

DEPLOYMENT: {dep_name}
NAMESPACE: {namespace}

Command MUST use deployment name: {dep_name}

Example: kubectl rollout restart deployment/{dep_name} -n {namespace}

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