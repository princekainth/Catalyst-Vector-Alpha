import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def test_token_misuse():
    registry = ToolRegistry()
    os.environ["CVA_ALLOWED_SERVICES"] = "service-a,service-b"
    
    # 1. Trigger trace A
    res_a = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="service-a")
    trace_a = res_a.get("trace_id")
    args_hash_a = res_a.get("details", {}).get("approval", {}).get("args_hash")
    
    # 2. Trigger trace B
    res_b = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="service-b")
    trace_b = res_b.get("trace_id")
    
    # 3. Issue token for A
    token_a, _ = issue_approval_token(trace_id=trace_a, tool="system_restart_allowed_service", args_hash=args_hash_a)
    
    # 4. Attempt to use token A for trace B
    print(f"Testing Token A on Trace B...")
    res_bad = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="service-b", approval_token=token_a, trace_id=trace_b)
    
    print(f"Result: {res_bad.get('status')} | Error: {res_bad.get('error')}")
    if res_bad.get("error") and "invalid approval token" in res_bad.get("error").lower():
        print("✓ SUCCESS: Token bound to trace A was rejected for trace B.")
    else:
        print("FAIL: Token bound to trace A was accepted for trace B!")

if __name__ == "__main__":
    test_token_misuse()
