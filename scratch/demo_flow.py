
import os
import json
import time
import sys
import subprocess

# Ensure PYTHONPATH includes current directory
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def run_demo():
    print("🚀 STARTING CVA CLOUD DEMO FLOW")
    
    # 1. Setup
    print("\n[1] SETUP: Preparing K8s environment...")
    subprocess.run(["kubectl", "create", "namespace", "cva-demo"], capture_output=True)
    
    deployment_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crashloop-demo
  namespace: cva-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: crashloop-demo
  template:
    metadata:
      labels:
        app: crashloop-demo
    spec:
      containers:
      - name: main
        image: busybox
        command: ["/bin/sh", "-c", "echo 'DEMO_CRASH'; sleep 5; exit 1"]
"""
    subprocess.run(["kubectl", "apply", "-f", "-"], input=deployment_yaml.encode())
    print("✓ Workload 'crashloop-demo' deployed in 'cva-demo'.")

    # 2. Wait for failure
    print("\n[2] DETECTION: Waiting for pod failure...")
    time.sleep(10)
    
    registry = ToolRegistry()
    status_res = registry.safe_call("get_pod_status", agent_id="demo_observer", namespace="cva-demo")
    problem_pods = status_res.get("data", {}).get("problem_pods", [])
    
    if not problem_pods:
        print("(!) Retrying detection...")
        time.sleep(10)
        status_res = registry.safe_call("get_pod_status", agent_id="demo_observer", namespace="cva-demo")
        problem_pods = status_res.get("data", {}).get("problem_pods", [])

    if not problem_pods:
        print("FAIL: Pod did not fail in time.")
        return

    broken_pod = problem_pods[0]["name"]
    print(f"(!) INCIDENT DETECTED: Pod {broken_pod} is in {problem_pods[0].get('issues', ['Error'])}")

    # NEW: Report incident to IncidentStore via API
    try:
        import requests
        requests.post("http://localhost:5000/api/incidents/report", json={
            "incident_type": "CrashLoopBackOff",
            "severity": "CRITICAL",
            "namespace": "cva-demo",
            "workload": "crashloop-demo",
            "pod": broken_pod,
            "evidence": f"Pod {broken_pod} is failing with issues: {problem_pods[0].get('issues')}",
            "classification": "Application crash detected in demo workload",
            "recommended_tool": "k8s_rollout_restart",
            "risk": "DESTRUCTIVE"
        }, timeout=5)
        print("✓ Incident reported to CVA Cloud IncidentStore.")
    except Exception as e:
        print(f"(!) Failed to report incident: {e} (Is the dashboard running?)")

    # 3. Observation
    print("\n[3] EVIDENCE: Collecting logs and description...")
    logs = registry.safe_call("k8s_get_pod_logs", agent_id="demo_observer", pod_name=broken_pod, namespace="cva-demo", tail=5)
    print(f"   > LOGS: {logs.get('data', {}).get('logs', '').strip()}")
    
    # 4. Remediation Proposal
    print("\n[4] PROPOSAL: Proposing safe remediation...")
    print("   > ACTION: k8s_rollout_restart")
    print("   > RISK:   DESTRUCTIVE")
    
    remed_res = registry.safe_call("k8s_rollout_restart", agent_id="demo_worker", deployment="crashloop-demo", namespace="cva-demo")
    
    if remed_res.get("code") == "approval_required":
        trace_id = remed_res.get("trace_id")
        approval_info = remed_res.get("approval") or remed_res.get("details", {}).get("approval", {})
        args_hash = approval_info.get("args_hash")
        
        print(f"\n[5] SECURITY: Approval gate hit (Trace: {trace_id[:10]}...)")
        print("   > Status: AWAITING_HUMAN_APPROVAL")
        
        # 5. Approval
        print("\n[6] APPROVAL: Human operator issues approval token...")
        token, _ = issue_approval_token(trace_id=trace_id, tool="k8s_rollout_restart", args_hash=args_hash, agent_id="demo_worker")
        print(f"   > TOKEN: {token[:15]}...")
        
        # 6. Execution
        print("\n[7] EXECUTION: Running remediation with token...")
        exec_res = registry.safe_call(
            "k8s_rollout_restart", 
            agent_id="demo_worker", 
            deployment="crashloop-demo", 
            namespace="cva-demo",
            approval_token=token,
            trace_id=trace_id
        )
        print(f"   > RESULT: {exec_res.get('status').upper()}")
        
        # 7. Verification
        print("\n[8] VERIFICATION: Confirming cluster state...")
        time.sleep(5)
        new_pods = subprocess.run(["kubectl", "get", "pods", "-n", "cva-demo"], capture_output=True, text=True).stdout
        print(f"Current Pods:\n{new_pods}")
        
        # 8. Audit Summary
        print("\n[9] AUDIT: Redacted Action Log Summary (Lifecycle for current trace)")
        audit_path = "./.cva/audit/actions.jsonl"
        if os.path.exists(audit_path):
            with open(audit_path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('trace_id') == trace_id:
                            ts = record.get('timestamp', 0)
                            status = record.get('result_status', 'unknown')
                            tool = record.get('tool', 'unknown')
                            print(f"   > {ts} | {tool:25} | {status:20} | Trace: {trace_id[:10]}...")
                    except Exception:
                        continue

    print("\n✅ DEMO COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    run_demo()
