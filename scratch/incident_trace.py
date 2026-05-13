
import os
import json
import time
import sys

# Ensure PYTHONPATH includes current directory
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def run_incident_test():
    registry = ToolRegistry()
    namespace = "cva-test"
    
    print("\n--- Phase 3: Incident Trace ---")
    
    # 1. Detection
    print("\n[1] Detection: Finding broken pods...")
    status_res = registry.safe_call("get_pod_status", agent_id="observer", namespace=namespace)
    
    problem_pods = status_res.get("data", {}).get("problem_pods", [])
    broken_pod = None
    if problem_pods:
        broken_pod = problem_pods[0]["name"]
        print(f"(!) Detected broken pod: {broken_pod}")
            
    if not broken_pod:
        print("FAIL: No broken pod found.")
        return

    # 2. Observation
    print(f"\n[2] Observation: Gathering logs for {broken_pod}...")
    logs_res = registry.safe_call("k8s_get_pod_logs", agent_id="observer", pod_name=broken_pod, namespace=namespace, tail=10)
    print(f"Logs Status: {logs_res.get('status')}")

    # 3. Remediation Proposal
    print("\n[4] Reasoning: Proposing fix (rollout restart)...")
    remed_res = registry.safe_call("k8s_rollout_restart", agent_id="worker", deployment="crashloop-test", namespace=namespace)
    
    if remed_res.get("code") == "approval_required":
        trace_id = remed_res.get("trace_id")
        approval_info = remed_res.get("approval") or remed_res.get("details", {}).get("approval", {})
        args_hash = approval_info.get("args_hash")
        
        print(f"✓ Correctly hit approval gate. TraceID: {trace_id}")
        
        # 4. Approval
        token, ttl = issue_approval_token(trace_id=trace_id, tool="k8s_rollout_restart", args_hash=args_hash, agent_id="worker")
        print(f"Issued Approval Token: {token}")
        
        # 5. Execution (PASSING TRACE_ID)
        print("\n[6] Execution: Running remediation with token and matching trace_id...")
        exec_res = registry.safe_call(
            "k8s_rollout_restart", 
            agent_id="worker", 
            deployment="crashloop-test", 
            namespace=namespace,
            approval_token=token,
            trace_id=trace_id
        )
        print(f"Execution Result: {exec_res.get('status')}")
        if exec_res.get("status") == "ok":
             print("✓ Success: Rollout restart triggered.")
        else:
             print(f"Execution Error: {exec_res}")
    else:
        print(f"FAIL: Expected approval_required, got {remed_res}")

    # 6. Verification
    print("\n[7] Verification: Checking audit log...")
    time.sleep(1)
    audit_path = "./.cva/audit/actions.jsonl"
    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            lines = f.readlines()
            for line in lines[-10:]:
                record = json.loads(line)
                if record.get("tool") == "k8s_rollout_restart":
                     print(f"AUDIT: {record.get('tool')} | Decision: {record.get('decision')} | Trace: {record.get('trace_id')[:10]}... | Status: {record.get('result_status')}")

if __name__ == "__main__":
    run_incident_test()
