import os
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def get_responses():
    registry = ToolRegistry()
    os.environ["CVA_ALLOWED_SERVICES"] = "cva-demo-service"
    
    # 3. approval_required response
    res_req = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="cva-demo-service")
    print("--- APPROVAL_REQUIRED RESPONSE ---")
    print(json.dumps(res_req, indent=2))
    
    # 4. successful execution response (after approval)
    trace_id = res_req.get("trace_id")
    approval_info = res_req.get("approval", {})
    args_hash = approval_info.get("args_hash")
    token, _ = issue_approval_token(trace_id=trace_id, tool="system_restart_allowed_service", args_hash=args_hash)
    
    res_exec = registry.safe_call("system_restart_allowed_service", 
                                agent_id="worker", 
                                service_name="cva-demo-service", 
                                approval_token=token, 
                                trace_id=trace_id)
    print("\n--- EXECUTION RESPONSE (WITH TOKEN) ---")
    print(json.dumps(res_exec, indent=2))

if __name__ == "__main__":
    get_responses()
