import os
import sys
import json
import subprocess

# Add project root to path
sys.path.append(os.getcwd())

from tool_registry import ToolRegistry

def validate_system_adapter():
    registry = ToolRegistry()
    print("🚀 STARTING SYSTEM ADAPTER VALIDATION")

    # 1. Disk Usage
    print("\n[1] system_get_disk_usage...")
    res = registry.safe_call("system_get_disk_usage", agent_id="worker", path="/")
    print(f"Result: {res.get('status')} | Data: {res.get('data')}")

    # 2. Memory Usage
    print("\n[2] system_get_memory_usage...")
    res = registry.safe_call("system_get_memory_usage", agent_id="worker")
    print(f"Result: {res.get('status')} | Data: {res.get('data')}")

    # 3. CPU Load
    print("\n[3] system_get_cpu_load...")
    res = registry.safe_call("system_get_cpu_load", agent_id="worker")
    print(f"Result: {res.get('status')} | Data: {res.get('data')}")

    # 4. Port Check (localhost)
    print("\n[4] system_check_port (localhost:80)...")
    res = registry.safe_call("system_check_port", agent_id="worker", port=80)
    print(f"Result: {res.get('status')} | Data: {res.get('data')}")

    # 5. Invalid Port Rejection
    print("\n[5] system_check_port (invalid port)...")
    res = registry.safe_call("system_check_port", agent_id="worker", port=99999)
    print(f"Result: {res.get('status')} | Error: {res.get('error')}")

    # 6. Host Restriction Check
    print("\n[6] system_check_port (external host)...")
    res = registry.safe_call("system_check_port", agent_id="worker", host="8.8.8.8", port=53)
    print(f"Result: {res.get('status')} | Error: {res.get('error')}")

    # 7. Log Tail Path Traversal Rejection
    print("\n[7] system_tail_log_file (traversal)...")
    res = registry.safe_call("system_tail_log_file", agent_id="worker", path="../../etc/passwd")
    print(f"Result: {res.get('status')} | Error: {res.get('error')}")

    # 8. Log Tail Success (mock log)
    print("\n[8] system_tail_log_file (allowed path)...")
    log_dir = "/tmp/cva-demo-logs"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/demo.log", "w") as f:
        f.write("Line 1\nLine 2\nLine 3")
    res = registry.safe_call("system_tail_log_file", agent_id="worker", path=f"{log_dir}/demo.log")
    print(f"Result: {res.get('status')} | Content: {res.get('data', {}).get('content', '').strip()}")

    # 9. Service Restart (Unauthorized)
    print("\n[9] system_restart_allowed_service (unauthorized)...")
    os.environ["CVA_ALLOWED_SERVICES"] = "cva-demo-service"
    res = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="nginx")
    print(f"Result: {res.get('status')} | Error: {res.get('error')}")

    # 10. Service Restart (Approval Required)
    print("\n[10] system_restart_allowed_service (approval_required)...")
    res = registry.safe_call("system_restart_allowed_service", agent_id="worker", service_name="cva-demo-service")
    print(f"Result: {res.get('status')} | Code: {res.get('code')} | Trace: {res.get('trace_id')}")

    if res.get("code") == "approval_required":
        print("✓ SUCCESS: Destructive system tool correctly gated.")
    else:
        print("FAIL: Destructive system tool was not gated.")

if __name__ == "__main__":
    validate_system_adapter()
