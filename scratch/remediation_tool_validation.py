import sys
import os
import json
import re
sys.path.append(os.getcwd())
from tools_k8s import _validate_container_image, _validate_probe_path, _validate_integer
from tool_registry import tool_registry
from cva_runtime.control_plane.capabilities import get_tool_profile, ToolRisk

def test_validators():
    print("--- Testing Primitives ---")
    
    # 1. Image Validator
    print("Testing Image Validator...")
    assert _validate_container_image("nginx:1.25") == True
    assert _validate_container_image("busybox") == True
    assert _validate_container_image("ghcr.io/org/app:v1") == True
    assert _validate_container_image("registry.example.com:5000/team/app@sha256:abc") == True
    assert _validate_container_image("nginx; rm -rf") == False
    assert _validate_container_image("nginx ") == False
    assert _validate_container_image("nginx|cat") == False
    assert _validate_container_image("nginx:1.25' OR 1=1") == False
    print("✓ Image Validator OK")

    # 2. Probe Path Validator
    print("\nTesting Probe Path Validator...")
    assert _validate_probe_path("/health") == True
    assert _validate_probe_path("health") == False
    assert _validate_probe_path("/health;rm") == False
    assert _validate_probe_path("/health path") == False
    print("✓ Probe Path Validator OK")

    # 3. Integer Validator
    print("\nTesting Integer Validator...")
    assert _validate_integer(8080, 1, 65535) is None
    assert _validate_integer("8080", 1, 65535) is None
    assert _validate_integer("8080; rm", 1, 65535) is not None
    assert _validate_integer(0, 1, 65535) is not None
    assert _validate_integer(65536, 1, 65535) is not None
    print("✓ Integer Validator OK")

def test_registry_gating():
    print("\n--- Testing Registry Gating & Schema ---")
    
    cases = [
        ("k8s_patch_deployment_image", {"deployment": "test", "container": "main", "image": "nginx:1.25"}),
        ("k8s_patch_probe", {"deployment": "test", "container": "main", "probe_type": "livenessProbe", "path": "/health", "port": 8080}),
        ("k8s_rollout_undo", {"deployment": "test"})
    ]

    for tool, args in cases:
        res = tool_registry.safe_call(tool, agent_id="worker", **args)
        status = res.get("status")
        # ToolExecutor returns status='awaiting_approval' or similar if code='approval_required'
        # Actually in this codebase's ToolExecutor.safe_call:
        # if needs_approval: return {"status": "error", "code": "approval_required", ...}
        
        code = res.get("code")
        print(f"Tool: {tool} | Status: {status} | Code: {code}")
        assert code == "approval_required", f"{tool} should be gated but got code={code}"
        print(f"✓ {tool} Gated OK")

def test_destructive_profiles():
    print("\n--- Testing Tool Profiles ---")
    
    for tool in ["k8s_patch_deployment_image", "k8s_patch_probe", "k8s_rollout_undo"]:
        profile = get_tool_profile(tool)
        assert profile is not None
        assert profile.risk == ToolRisk.DESTRUCTIVE
        print(f"✓ {tool} risk verified as DESTRUCTIVE")

if __name__ == "__main__":
    try:
        test_validators()
        test_destructive_profiles()
        test_registry_gating()
        print("\n" + "="*40)
        print("ALL REMEDIATION VALIDATION TESTS PASSED")
        print("="*40)
    except Exception as e:
        print(f"\n[FAILURE] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
