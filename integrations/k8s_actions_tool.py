from __future__ import annotations
import time, logging
from typing import Dict, Any, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import json  # Add this import at the top

class K8sActions:
    """
    Safe K8s actions with dry-run default.
    Auth: in-cluster or local kubeconfig.
    """
    def __init__(self, logger: Optional[logging.Logger] = None, dry_run: bool = True):
        self.log = logger or logging.getLogger("K8sActions")
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.apps = client.AppsV1Api()
        self.dry_run = dry_run
        self.safety_limits = {
            "max_replicas": 20,
            "allowed_namespaces": ["default", "development", "staging"],  # Never production
            "max_operations_per_minute": 10
        }

    def restart_deployment(self, namespace: str, name: str) -> Dict[str, Any]:
        body = {"spec": {"template": {"metadata": {"annotations": {"ai.catalyst/restarted-at": str(int(time.time()))}}}}}
        outcome = {"action": "k8s_restart", "namespace": namespace, "deployment": name, "dry_run": self.dry_run}
        try:
            if self.dry_run:
                self.log.info(f"[DRY-RUN] restart {namespace}/{name}")
            else:
                self.apps.patch_namespaced_deployment(name=name, namespace=namespace, body=body)
            outcome["status"] = "ok"
        except ApiException as e:
            self.log.error(f"restart_deployment failed: {e}")
            outcome.update({"status": "error", "error": str(e)})
        return outcome

    def scale_deployment(self, namespace: str, name: str, replicas: int) -> Dict[str, Any]:
        """
        Scale a Kubernetes deployment with production safety controls.
        
        Args:
            namespace: Target Kubernetes namespace
            name: Deployment name
            replicas: Desired replica count
            
        Returns:
            Dict with status, action details, and any errors
        """
        outcome = {
            "action": "k8s_scale", 
            "namespace": namespace, 
            "deployment": name, 
            "replicas": replicas, 
            "dry_run": self.dry_run,
            "timestamp": time.time()
        }
        
        try:
            # Input validation
            if not isinstance(replicas, int) or replicas < 0:
                outcome.update({"status": "error", "error": f"Invalid replica count: {replicas}. Must be non-negative integer."})
                return outcome
                
            if replicas > self.safety_limits.get("max_replicas", 20):
                outcome.update({"status": "error", "error": f"Replica count {replicas} exceeds safety limit of {self.safety_limits['max_replicas']}"})
                return outcome
                
            # Namespace safety check
            allowed_namespaces = self.safety_limits.get("allowed_namespaces", ["default", "development", "staging"])
            if namespace not in allowed_namespaces:
                outcome.update({"status": "error", "error": f"Namespace '{namespace}' not in allowed list: {allowed_namespaces}"})
                return outcome
                
            # Check if deployment exists first
            try:
                current_deployment = self.apps.read_namespaced_deployment(name=name, namespace=namespace)
                current_replicas = current_deployment.spec.replicas or 0
                outcome["current_replicas"] = current_replicas
                
                # Skip if no change needed
                if current_replicas == replicas:
                    outcome.update({
                        "status": "skipped", 
                        "reason": f"Deployment already has {replicas} replicas"
                    })
                    return outcome
                    
            except ApiException as e:
                if e.status == 404:
                    outcome.update({"status": "error", "error": f"Deployment '{name}' not found in namespace '{namespace}'"})
                    return outcome
                else:
                    # Re-raise other API exceptions
                    raise e
            
            # Rate limiting check (if implemented)
            if hasattr(self, '_check_rate_limit') and not self._check_rate_limit():
                outcome.update({"status": "error", "error": "Rate limit exceeded. Too many operations in short period."})
                return outcome
            
            # Prepare the scaling operation
            body = {"spec": {"replicas": replicas}}
            
            if self.dry_run:
                self.log.info(f"[DRY-RUN] scale {namespace}/{name} -> {replicas} (current: {current_replicas})")
                outcome.update({
                    "status": "ok",
                    "message": f"DRY-RUN: Would scale {name} from {current_replicas} to {replicas} replicas"
                })
            else:
                # Perform actual scaling
                self.log.info(f"Scaling {namespace}/{name} from {current_replicas} to {replicas} replicas")
                
                # Use the scale subresource for atomic scaling
                scale_response = self.apps.patch_namespaced_deployment_scale(
                    name=name, 
                    namespace=namespace, 
                    body=body
                )
                
                # Verify the operation was accepted
                new_desired_replicas = scale_response.spec.replicas
                if new_desired_replicas != replicas:
                    outcome.update({
                        "status": "warning",
                        "message": f"Scaling initiated but Kubernetes set replicas to {new_desired_replicas} instead of {replicas}",
                        "actual_replicas": new_desired_replicas
                    })
                else:
                    outcome.update({
                        "status": "ok",
                        "message": f"Successfully scaled {name} from {current_replicas} to {replicas} replicas"
                    })
                
                self.log.info(f"Scale operation completed for {namespace}/{name}")
                
        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.reason}"
            if hasattr(e, 'body') and e.body:
                try:
                    error_detail = json.loads(e.body)
                    if 'message' in error_detail:
                        error_msg = f"Kubernetes API error: {error_detail['message']}"
                except (json.JSONDecodeError, KeyError):
                    pass
                    
            self.log.error(f"scale_deployment failed: {error_msg}")
            outcome.update({"status": "error", "error": error_msg, "api_status": e.status})
            
        except Exception as e:
            error_msg = f"Unexpected error during scaling: {str(e)}"
            self.log.error(error_msg)
            outcome.update({"status": "error", "error": error_msg})
        
        return outcome
    
    def update_resource_limits(self, namespace: str, name: str, cpu_limit: str, memory_limit: str, container_name: str = None) -> Dict[str, Any]:
        """
        Updates CPU and Memory limits for containers in a deployment.
        Fetches current deployment to avoid overwriting existing configurations.
        """
        outcome = {
            "action": "k8s_update_resources",
            "namespace": namespace,
            "deployment": name,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "dry_run": self.dry_run,
            "timestamp": time.time()
        }
        
        try:
            # Validate resource format
            if not self._validate_resource_format(cpu_limit, memory_limit):
                outcome.update({"status": "error", "error": "Invalid CPU or memory format"})
                return outcome
                
            # Safety checks
            allowed_namespaces = self.safety_limits.get("allowed_namespaces", ["default", "development", "staging"])
            if namespace not in allowed_namespaces:
                outcome.update({"status": "error", "error": f"Namespace '{namespace}' not in allowed list"})
                return outcome
            
            # Fetch current deployment
            current_deployment = self.apps.read_namespaced_deployment(name=name, namespace=namespace)
            containers = current_deployment.spec.template.spec.containers
            
            if not containers:
                outcome.update({"status": "error", "error": "No containers found in deployment"})
                return outcome
            
            # Determine target container
            target_container = containers[0]  # Default to first container
            if container_name:
                target_container = next((c for c in containers if c.name == container_name), None)
                if not target_container:
                    outcome.update({"status": "error", "error": f"Container '{container_name}' not found"})
                    return outcome
            
            outcome["target_container"] = target_container.name
            
            # Store current limits
            current_resources = target_container.resources
            current_limits = current_resources.limits if current_resources else {}
            outcome["previous_limits"] = dict(current_limits) if current_limits else {}
            
            # Find container index for patching
            container_index = next(i for i, c in enumerate(containers) if c.name == target_container.name)
            
            if self.dry_run:
                self.log.info(f"[DRY-RUN] Update resource limits for {namespace}/{name} container '{target_container.name}' to cpu={cpu_limit}, memory={memory_limit}")
                outcome.update({
                    "status": "ok",
                    "message": f"DRY-RUN: Would update container '{target_container.name}' resource limits"
                })
            else:
                # Use strategic merge patch (simpler than JSON patch)
                patch_body = {
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": target_container.name,
                                        "resources": {
                                            "limits": {
                                                "cpu": cpu_limit,
                                                "memory": memory_limit
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
                
                self.apps.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body=patch_body
                )
                
                self.log.info(f"Updated resource limits for {namespace}/{name} container '{target_container.name}'")
                outcome.update({
                    "status": "ok",
                    "message": f"Successfully updated resource limits for container '{target_container.name}'"
                })
                
        except ApiException as e:
            if e.status == 404:
                error_msg = f"Deployment '{name}' not found in namespace '{namespace}'"
            else:
                error_msg = f"Kubernetes API error: {e.reason}"
            
            self.log.error(f"update_resource_limits failed: {error_msg}")
            outcome.update({"status": "error", "error": error_msg, "api_status": e.status})
            
        except Exception as e:
            error_msg = f"Unexpected error updating resource limits: {str(e)}"
            self.log.error(error_msg)
            outcome.update({"status": "error", "error": error_msg})
        
        return outcome

    def _validate_resource_format(self, cpu_limit: str, memory_limit: str) -> bool:
        """Validate Kubernetes resource format"""
        import re
        
        # CPU: number with optional 'm' suffix (e.g., "100m", "1", "1.5")
        cpu_pattern = r'^\d+(\.\d+)?m?$'
        # Memory: number with K/M/G/T suffixes (e.g., "256Mi", "1Gi")  
        memory_pattern = r'^\d+(\.\d+)?[KMGT]i?$'
        
        return bool(re.match(cpu_pattern, cpu_limit) and re.match(memory_pattern, memory_limit))
    
    def _check_rate_limit(self) -> bool:
        """Simple rate limiting - implement based on your needs"""
        # For now, always allow operations
        # In production, track operations per minute
        return True