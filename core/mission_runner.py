# core/mission_runner.py
from __future__ import annotations
from core.mission_policy import select_next_mission
from typing import Optional, Dict, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

class MissionRunner:
    """
    Minimal runner:
      1) observe (Prometheus)
      2) decide (policy eval)
      3) propose/act (K8s), respecting approval + cooldown/budget
    """
    def __init__(self, metrics, actions, policy, mem_kernel, logger: Optional[logging.Logger] = None):
        self.metrics = metrics
        self.actions = actions
        self.policy = policy
        self.mem = mem_kernel
        self.log = logger or logging.getLogger("MissionRunner")

    def scale_on_cpu_threshold(
        self,
        namespace: str,
        deployment: str,
        threshold: float,
        replicas: int,
        instance: Optional[str] = None,
        approval: str = "human",
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """Scale deployment based on cluster CPU with timeout and error handling."""
        from tools import kubernetes_pod_metrics_tool, _get_deploy
        import signal
        
        now = time.time()
        mission_name = "scale_on_cpu_threshold"
        
        try:
            # Wrap in timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_scaling_logic, 
                                        namespace, deployment, threshold, replicas)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    return {
                        "mission": mission_name,
                        "status": "error",
                        "error": f"Operation timed out after {timeout_seconds}s",
                        "timestamp": now,
                    }
        
        except Exception as e:
            return {
                "mission": mission_name,
                "status": "error",
                "error": f"Unexpected failure: {str(e)}",
                "timestamp": now,
            }
            
    def _execute_scaling_logic(self, namespace, deployment, threshold, replicas):
        """Internal logic with proper error handling."""
        from tools import kubernetes_pod_metrics_tool, _get_deploy
        
        now = time.time()
        mission_name = "scale_on_cpu_threshold"
        
        try:
            # 1) Get ALL pod metrics
            pods_result = kubernetes_pod_metrics_tool(limit=999)
            if pods_result.get('status') != 'ok':
                return {
                    "mission": mission_name,
                    "status": "error",
                    "error": "Failed to fetch pod metrics",
                    "timestamp": now,
                }
        except Exception as e:
            return {
                "mission": mission_name,
                "status": "error",
                "error": f"Metrics fetch failed: {e}",
                "timestamp": now,
            }
        
        try:
            # 2) Calculate total cluster CPU
            all_pods = pods_result['data']['pods']
            total_cluster_cpu = sum(p.get('cpu_mcores', 0) for p in all_pods)
            
            # 3) Get current deployment replicas
            current_deploy = _get_deploy(namespace, deployment)
            current_replicas = int(current_deploy['spec'].get('replicas', 1))
            
        except Exception as e:
            return {
                "mission": mission_name,
                "status": "error",
                "error": f"Failed to get deployment info: {e}",
                "timestamp": now,
            }
        
        # 4) Decision logic
        if total_cluster_cpu <= threshold:
            return {
                "mission": mission_name,
                "status": "no_action_needed",
                "observed": {
                    "total_cluster_cpu_mcores": total_cluster_cpu,
                    "current_replicas": current_replicas,
                },
                "reason": f"Cluster CPU {total_cluster_cpu} mcores below threshold {threshold}",
                "timestamp": now,
            }
        
        # 5) Prepare scaling
        new_replicas = current_replicas + 1
        rule = {
            "id": f"scale:{namespace}:{deployment}",
            "metric": "cluster_cpu_mcores",
            "op": ">",
            "threshold": float(threshold),
            "approval": "human",
            "cooldown_seconds": 60,
            "change_budget_per_hour": 2,
            "action": "k8s_scale",
            "params": {"namespace": namespace, "name": deployment, "replicas": new_replicas},
        }
        
        try:
            decision = self.policy.eval_threshold_rule(rule, metric_value=total_cluster_cpu)
        except Exception as e:
            return {
                "mission": mission_name,
                "status": "error",
                "error": f"Policy evaluation failed: {e}",
                "timestamp": now,
            }

        if not decision["allow"]:
            return {
                "mission": mission_name,
                "status": "denied",
                "observed": {"total_cluster_cpu_mcores": total_cluster_cpu},
                "decision": decision,
                "timestamp": now,
            }

        if decision["needs_approval"]:
            if self.mem:
                self.mem.add_memory("HumanInputAcknowledged", 
                                {"response": "Awaiting approval", "context": rule})
            return {
                "mission": mission_name,
                "status": "awaiting_approval",
                "observed": {"total_cluster_cpu_mcores": total_cluster_cpu, 
                            "proposed_replicas": new_replicas},
                "decision": decision,
                "timestamp": now,
            }

        # 6) Execute scaling
        try:
            outcome = self.actions.scale_deployment(namespace=namespace, 
                                                name=deployment, 
                                                replicas=new_replicas)
            self.policy.record_action(rule_id=rule["id"])
            if self.mem:
                self.mem.add_memory("ActionExecuted", {"action": "scale", "outcome": outcome})
            
            return {
                "mission": mission_name,
                "status": "executed",
                "observed": {"total_cluster_cpu_mcores": total_cluster_cpu},
                "action": {"from_replicas": current_replicas, "to_replicas": new_replicas},
                "outcome": outcome,
                "timestamp": now,
            }
            
        except Exception as e:
            return {
                "mission": mission_name,
                "status": "error",
                "error": f"Scale execution failed: {e}",
                "timestamp": now,
            }

    def decide_and_run(self) -> Dict[str, Any]:
        """
        Choose a mission using epsilon-greedy (from mission_policy)
        and execute if we have a local executor. Otherwise, return a
        no-op stub so the Planner can still log the choice.
        """
        mission = select_next_mission(self.mem)
        now = time.time()

        # Dispatch table grows as you add executors
        if mission == "scale_on_cpu_threshold":
            # Threshold in millicores: 10000 = 10 CPU cores
            return self.scale_on_cpu_threshold(
                namespace="default",
                deployment="nginx",
                threshold=10000,  # Scale when cluster exceeds 10 CPU cores
                replicas=3,
                approval="auto",
            )

        # No local executor for this mission yet — safe no-op
        self.log.debug(f"[MissionRunner] no executor for mission '{mission}', skipping")
        return {
            "mission": mission,
            "status": "no_executor",
            "timestamp": now,
        }

