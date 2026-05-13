import os
import sys
import json
import logging

# Add project root to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token
from cva_runtime.control_plane.audit_log import _audit_path

def show_proof():
    # Setup logging to capture CatalystLogger
    logger = logging.getLogger("CatalystLogger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)

    registry = ToolRegistry()
    os.environ["CVA_ALLOWED_SERVICES"] = "cva-demo-service"
    
    print("=== 1. REGISTERED SYSTEM TOOLS ===")
    tools = [t for t in registry.list_tool_names() if t.startswith("system_")]
    for t in tools:
        print(f"- {t}")

    print("\n=== 2. TOOL PROFILES (SYSTEM_READ/WRITE) ===")
    from cva_runtime.control_plane.capabilities import get_tool_profile
    for t in tools:
        prof = get_tool_profile(t)
        if prof:
            print(f"{t}: Risk={prof.risk.value}, Caps={[c.value for c in prof.required_caps]}")
        else:
            print(f"{t}: No profile defined (default Deny)")

    print("\n=== 3. APPROVAL_REQUIRED RESPONSE ===")
    res_req = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="cva-demo-service")
    print(json.dumps(res_req, indent=2))

    print("\n=== 4. SUCCESSFUL EXECUTION (AFTER APPROVAL) ===")
    trace_id = res_req.get("trace_id")
    approval_info = res_req.get("approval", {})
    args_hash = approval_info.get("args_hash")
    token, _ = issue_approval_token(trace_id=trace_id, tool="system_restart_allowed_service", args_hash=args_hash)
    
    res_exec = registry.safe_call("system_restart_allowed_service", 
                                agent_id="worker", 
                                service_name="cva-demo-service", 
                                approval_token=token, 
                                trace_id=trace_id)
    print(json.dumps(res_exec, indent=2))

    print("\n=== 5. AUDIT LOG ENTRIES (WITH REDACTION) ===")
    # Trigger a call with a redacted key
    registry.safe_call("k8s_patch_deployment_env", 
                     agent_id="worker", 
                     deployment="test", 
                     env_name="DB_PASSWORD", 
                     env_value="super-secret-password-123")
    
    path = _audit_path()
    with open(path, "r") as f:
        lines = f.readlines()
        # Find the last POLICY_DECISION line for the patch
        for line in reversed(lines):
            data = json.loads(line)
            if data.get("tool") == "k8s_patch_deployment_env" and data.get("decision") == "POLICY_DECISION":
                print(f"Audit Record (Hashed Args): {line.strip()}")
                break
    
    # Also show the Standard Log redaction
    print("\nStandard Log Redaction Check:")
    registry.safe_call("system_get_disk_usage", agent_id="worker", path="/", token="hidden-secret-key")

if __name__ == "__main__":
    show_proof()
