
import os
import json
import time
import sys

# Ensure PYTHONPATH includes current directory
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def run_redaction_test():
    registry = ToolRegistry()
    namespace = "cva-test"
    deployment = "crashloop-test"
    
    print("\n--- Phase 3: Redaction and Environment Patch Test ---")
    
    # 1. Proposal
    print("\n[1] Proposal: Patching environment with sensitive value...")
    remed_res = registry.safe_call(
        "k8s_patch_deployment_env", 
        agent_id="worker", 
        deployment=deployment, 
        namespace=namespace,
        env_name="DB_PASSWORD",
        env_value="SUPER_SECRET_123"
    )
    
    if remed_res.get("code") == "approval_required":
        trace_id = remed_res.get("trace_id")
        approval_info = remed_res.get("approval") or remed_res.get("details", {}).get("approval", {})
        args_hash = approval_info.get("args_hash")
        
        print(f"✓ Hit approval gate. TraceID: {trace_id}")
        
        # 2. Approval
        token, ttl = issue_approval_token(trace_id=trace_id, tool="k8s_patch_deployment_env", args_hash=args_hash, agent_id="worker")
        
        # 3. Execution
        print("[2] Execution: Applying patch...")
        exec_res = registry.safe_call(
            "k8s_patch_deployment_env", 
            agent_id="worker", 
            deployment=deployment, 
            namespace=namespace,
            env_name="DB_PASSWORD",
            env_value="SUPER_SECRET_123",
            approval_token=token,
            trace_id=trace_id
        )
        print(f"Execution Result: {exec_res.get('status')}")

    # 4. Verification
    print("\n[3] Verification: Checking audit log for redaction...")
    time.sleep(1)
    audit_path = "./.cva/audit/actions.jsonl"
    found_redacted = False
    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "k8s_patch_deployment_env" in line and "SUPER_SECRET_123" in line:
                    print("FAIL: Sensitive value found in plain text in audit log!")
                    return
                if "k8s_patch_deployment_env" in line and "***REDACTED***" in line:
                    found_redacted = True
                    
    if found_redacted:
        print("✓ Success: Sensitive values are redacted in audit log.")
    else:
        print("FAIL: Could not confirm redaction in audit log.")

if __name__ == "__main__":
    run_redaction_test()
