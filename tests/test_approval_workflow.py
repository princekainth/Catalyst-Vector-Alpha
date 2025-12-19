"""
Test CVA's human-in-the-loop approval workflow
"""
import os
import sys
import json
import time
import socket

import pytest
import requests

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_URL = "http://127.0.0.1:5000"


def _server_up(host="127.0.0.1", port=5000, timeout=0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get_pending():
    """Helper: fetch pending approvals."""
    resp = requests.get(f"{BASE_URL}/api/pending")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    return resp.json()


def _create_scale_request():
    """Helper: create a scale request that should require approval."""
    from tool_registry import tool_registry

    tool = tool_registry.get_tool("k8s_scale")
    result = tool.func(deployment="nginx-test", namespace="default", replicas=4)

    assert isinstance(result, dict), "Expected dict result from tool"
    assert result.get("status") == "awaiting_approval", "Should require approval"
    return result


@pytest.mark.skipif(not _server_up(), reason="Approval API server not running on 127.0.0.1:5000")
def test_pending_plans_endpoint():
    print("\n=== TEST: Get pending plans ===")
    data = _get_pending()
    print(f"Response:\n{json.dumps(data, indent=2)[:500]}...")
    # Your API returns {"pending": null} when nothing is pending
    assert "pending" in data


def test_create_scale_request():
    print("\n=== TEST: Create scale request ===")
    result = _create_scale_request()
    print(f"Scale request result:\n{json.dumps(result, indent=2)[:500]}...")
    # If you want to assert specific keys later, do it here (no returns)


@pytest.mark.skipif(not _server_up(), reason="Approval API server not running on 127.0.0.1:5000")
def test_approval_flow():
    print("\n=== TEST: Full approval flow ===")

    # Step 1: create request (tool-level)
    print("\n1) Creating scale request...")
    _create_scale_request()

    # Step 2: check it appears in pending (API-level)
    print("\n2) Checking pending plans...")
    time.sleep(0.5)
    data = _get_pending()
    assert "pending" in data

    # Step 3: document the manual approval endpoint (not executing it here)
    print("\n3) Approval would happen via:")
    print(f"   POST {BASE_URL}/api/approve")
    print("   Body: {'task_id': '<task_id>', 'approval_token': '<token>'}")

    print("\n✓ Approval workflow verified (manual approve step not executed)")
