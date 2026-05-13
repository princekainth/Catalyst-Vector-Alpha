
import os
import json
import re
from typing import Dict, Any, Optional

# Mocking the environment for testing
os.environ["CVA_ALLOW_UNSAFE_TOOL_CALL"] = "0"

from tool_registry import ToolRegistry
from tools_k8s import (
    _validate_k8s_name, 
    _validate_env_name, 
    _validate_cpu_quantity, 
    _validate_memory_quantity,
    k8s_get_pod_logs,
    k8s_patch_deployment_env,
    k8s_patch_resource_limits
)
from cva_runtime.control_plane.capabilities import get_tool_profile, ToolRisk
from cva_runtime.control_plane.audit_log import _redact

def test_validation():
    print("--- Testing Validation Logic ---")
    
    # 1. k8s_name validation
    assert _validate_k8s_name("valid-name") == True
    assert _validate_k8s_name("invalid name") == False
    assert _validate_k8s_name("name; rm -rf /") == False
    print("✓ _validate_k8s_name works")

    # 2. env_name validation
    assert _validate_env_name("VALID_ENV_123") == True
    assert _validate_env_name("invalid-env") == False
    assert _validate_env_name("invalid name") == False
    print("✓ _validate_env_name works")

    # 3. CPU/Memory validation
    assert _validate_cpu_quantity("500m") == True
    assert _validate_cpu_quantity("1") == True
    assert _validate_cpu_quantity("512Mi") == False
    
    assert _validate_memory_quantity("512Mi") == True
    assert _validate_memory_quantity("1Gi") == True
    assert _validate_memory_quantity("500m") == False
    print("✓ CPU/Memory cross-type validation works")

def test_tool_logic():
    print("\n--- Testing Tool Logic (Arguments) ---")
    
    # 1. tail validation
    res = k8s_get_pod_logs(pod_name="mypod", tail=0)
    assert res["success"] == False
    assert "Tail must be int between 1 and 1000" in res["error"]
    
    res = k8s_get_pod_logs(pod_name="mypod", tail=1001)
    assert res["success"] == False
    print("✓ tail validation works")

    # 2. env_value length
    long_val = "x" * 4097
    res = k8s_patch_deployment_env(deployment="dep", env_name="NAME", env_value=long_val)
    assert res["success"] == False
    assert "Invalid env_value" in res["error"]
    print("✓ env_value length validation works")

    # 3. empty resource patch
    res = k8s_patch_resource_limits(deployment="dep")
    assert res["success"] == False
    assert "No valid resource values specified" in res["error"]
    print("✓ empty resource patch rejected")

def test_capabilities():
    print("\n--- Testing Capabilities/Risk ---")
    
    rollout_profile = get_tool_profile("k8s_rollout_restart")
    print(f"Rollout Profile Risk: {rollout_profile.risk}")
    assert rollout_profile.risk == ToolRisk.DESTRUCTIVE
    print("✓ k8s_rollout_restart is DESTRUCTIVE")

    env_profile = get_tool_profile("k8s_patch_deployment_env")
    assert env_profile.risk == ToolRisk.DESTRUCTIVE
    print("✓ k8s_patch_deployment_env is DESTRUCTIVE")

    logs_profile = get_tool_profile("k8s_get_pod_logs")
    assert logs_profile.risk == ToolRisk.SAFE
    print("✓ k8s_get_pod_logs is SAFE")

def test_audit_redaction():
    print("\n--- Testing Audit Redaction (Recursive) ---")
    
    data = {
        "deployment": "my-dep",
        "nested": {
            "env_value": "SECRET_VALUE",
            "other": "safe"
        },
        "list": [
            {"value": "ANOTHER_SECRET"},
            {"safe": 123}
        ]
    }
    
    redacted = _redact(data)
    assert redacted["nested"]["env_value"] == "***REDACTED***"
    assert redacted["list"][0]["value"] == "***REDACTED***"
    assert redacted["nested"]["other"] == "safe"
    print("✓ audit redaction is recursive and masks secrets")

def test_safe_call_enforcement():
    print("\n--- Testing ToolExecutor Enforcement ---")
    registry = ToolRegistry()
    
    # Destructive tool without approval token should fail or return approval_required
    res = registry.safe_call("k8s_rollout_restart", agent_id="worker", deployment="test-dep")
    print(f"Tool Result: {res}")
    assert res.get("status") in ["approval_required", "deny", "error", "approval_invalid"], f"Expected gated status, got {res.get('status')}"
    print(f"✓ Gated execution returned: {res.get('status')}")

if __name__ == "__main__":
    try:
        test_validation()
        test_tool_logic()
        test_capabilities()
        test_audit_redaction()
        test_safe_call_enforcement()
        print("\nALL PHASE 2 VALIDATION TESTS PASSED")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
