
import os
import sys
sys.path.append(os.getcwd())
from tool_registry import ToolRegistry
from cva_runtime.control_plane.approvals import issue_approval_token

def verify_trace_binding():
    registry = ToolRegistry()
    trace_a = "trc_aaaaa"
    trace_b = "trc_bbbbb"
    tool = "k8s_rollout_restart"
    args_hash = "abc123hash"
    
    # 1. Issue for Trace A
    token_a, _ = issue_approval_token(trace_id=trace_a, tool=tool, args_hash=args_hash)
    print(f"Token issued for Trace A: {token_a}")
    
    # 2. Try on Trace B
    print(f"Attempting to use Token A on Trace B...")
    res = registry.safe_call(tool, deployment="test", namespace="default", trace_id=trace_b, approval_token=token_a)
    
    print(f"Result Status: {res.get('status')}")
    print(f"Result Code: {res.get('code')}")
    print(f"Result Message: {res.get('message')}")
    
    if res.get("code") == "approval_invalid" and "trace mismatch" in res.get("message", ""):
        print("✓ SUCCESS: Token correctly rejected due to trace mismatch.")
    else:
        print("FAIL: Token was not rejected for trace mismatch as expected.")

if __name__ == "__main__":
    verify_trace_binding()
