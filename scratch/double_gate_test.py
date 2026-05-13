import os
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def test_double_gate():
    registry = ToolRegistry()
    os.environ["CVA_ALLOWED_SERVICES"] = "cva-demo-service"
    
    print("🚀 LAYER 1: NON-ALLOWLISTED SERVICE (nginx)")
    
    # 1. Attempt without approval
    print("\n[Step 1] Attempting 'nginx' without approval...")
    res1 = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="nginx")
    print(f"Result: status={res1.get('status')}, code={res1.get('code')}, error={res1.get('error')}")
    
    # 2. Issue token
    trace_id1 = res1.get("trace_id")
    args_hash1 = res1.get("approval", {}).get("args_hash")
    token1, _ = issue_approval_token(trace_id=trace_id1, tool="system_restart_allowed_service", args_hash=args_hash1)
    print(f"Token issued for trace {trace_id1}")
    
    # 3. Attempt with approval
    print("\n[Step 2] Attempting 'nginx' WITH approval token...")
    res2 = registry.safe_call("system_restart_allowed_service", 
                             agent_id="worker", 
                             service_name="nginx", 
                             approval_token=token1, 
                             trace_id=trace_id1)
    print(f"Result: status={res2.get('status')}, error={res2.get('error')}")
    
    # Manual cooldown reset for demo
    tool_obj = registry.get_tool("system_restart_allowed_service")
    if tool_obj:
        setattr(tool_obj, "_last_called_ts", 0.0)
    
    if res2.get("status") == "error" and "not authorized" in str(res2.get("error")).lower():
        print("✓ SUCCESS: Tool-level allowlist correctly blocked 'nginx' after approval.")
    else:
        print("FAIL: Tool did not enforce allowlist after approval.")

    print("\n🚀 LAYER 2: ALLOWLISTED SERVICE (cva-demo-service)")
    
    # 4. Attempt without approval
    print("\n[Step 3] Attempting 'cva-demo-service' without approval...")
    res3 = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="cva-demo-service")
    print(f"Result: status={res3.get('status')}, code={res3.get('code')}")
    
    # 5. Issue token
    trace_id3 = res3.get("trace_id")
    args_hash3 = res3.get("approval", {}).get("args_hash")
    token3, _ = issue_approval_token(trace_id=trace_id3, tool="system_restart_allowed_service", args_hash=args_hash3)
    print(f"Token issued for trace {trace_id3}")
    
    # 6. Attempt with approval
    print("\n[Step 4] Attempting 'cva-demo-service' WITH approval token...")
    res4 = registry.safe_call("system_restart_allowed_service", 
                             agent_id="worker", 
                             service_name="cva-demo-service", 
                             approval_token=token3, 
                             trace_id=trace_id3)
    # Note: this might return status="error" if systemctl is missing, but it shouldn't be an allowlist error.
    print(f"Result: status={res4.get('status')}, error={res4.get('error')}")
    if res4.get("status") == "ok" or ("not authorized" not in str(res4.get("error")).lower()):
        print("✓ SUCCESS: Tool-level allowlist passed for 'cva-demo-service'.")
    else:
        print("FAIL: Tool blocked allowlisted service.")

if __name__ == "__main__":
    test_double_gate()
