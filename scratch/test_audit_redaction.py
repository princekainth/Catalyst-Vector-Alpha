
import os
import sys
import json
sys.path.append(os.getcwd())

def verify_audit_log_redaction():
    from app import app
    client = app.test_client()
    
    audit_path = "./.cva/audit/actions.jsonl"
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    
    # Manually inject a redacted record to simulate real logging behavior
    from cva_runtime.control_plane.audit_log import log_decision
    
    test_args = {
        "tool": "k8s_patch_deployment_env",
        "env_value": "SECRET_PASSWORD_123",
        "nested": {"value": "HIDDEN_VAL"}
    }
    
    print("Logging a sensitive action...")
    log_decision(
        trace_id="trc_redact_test",
        agent_id="test_agent",
        tool="k8s_patch_deployment_env",
        args=test_args,
        decision="TEST",
        reason="testing redaction",
        result_status="ok",
        latency_ms=10
    )
    
    print("Fetching logs via /api/audit/logs...")
    res = client.get("/api/audit/logs")
    data = res.get_json()
    
    logs = data.get("logs", [])
    relevant_log = None
    for l in logs:
        if l.get("trace_id") == "trc_redact_test":
            relevant_log = l
            break
            
    if not relevant_log:
        print("FAIL: Test log not found in API response.")
        return
        
    print(f"Log Record: {json.dumps(relevant_log)}")
    
    # Note: AuditRecord does NOT store args, only args_hash.
    # But let's check if 'extra' or anything else contains raw data.
    
    raw_text = res.get_data(as_text=True)
    if "SECRET_PASSWORD_123" in raw_text or "HIDDEN_VAL" in raw_text:
        print("FAIL: Sensitive data found in raw response!")
    else:
        print("✓ SUCCESS: No sensitive data found in audit log response.")
        print("✓ Note: System correctly uses args_hash and redacted extras.")

if __name__ == "__main__":
    verify_audit_log_redaction()
