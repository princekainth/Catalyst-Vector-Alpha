import os
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

def test_audit_redaction():
    registry = ToolRegistry()
    audit_log_path = os.getenv("CVA_AUDIT_LOG_PATH", "./.cva/audit/actions.jsonl")
    
    # Tool call with sensitive looking value
    print("Executing tool with sensitive value...")
    # Using k8s_patch_deployment_env which we know should redact env_value
    # But since we're testing system tools, let's see if we can find one that takes sensitive input.
    # system_restart_allowed_service doesn't.
    # Let's use k8s_patch_deployment_env as a control.
    
    # Ensure audit log is fresh or we can find the last line
    res = registry.safe_call("k8s_patch_deployment_env", 
                           agent_id="worker", 
                           namespace="default", 
                           deployment="test", 
                           env_name="DB_PASSWORD", 
                           env_value="super-secret-123")
    
    print(f"Result status: {res.get('status')}")
    
    # Read last line of audit log
    if os.path.exists(audit_log_path):
        with open(audit_log_path, 'r') as f:
            lines = f.readlines()
            last_line = json.loads(lines[-1])
            
            # Check if 'super-secret-123' is in the log
            log_str = json.dumps(last_line)
            if "super-secret-123" in log_str:
                print("FAIL: Sensitive value found in audit log!")
            else:
                print("✓ SUCCESS: Sensitive value was redacted/hashed.")
                # Confirm it IS there in hashed form
                import hashlib
                expected_hash = hashlib.sha256("super-secret-123".encode()).hexdigest()
                if expected_hash in log_str:
                     print("✓ SUCCESS: Found SHA-256 hash of the value.")
    else:
        print("FAIL: Audit log not found.")

if __name__ == "__main__":
    test_audit_redaction()
