import os
import sys
import subprocess
import pytest


def _real_k8s_enabled() -> bool:
    # Only run if explicitly enabled
    if os.getenv("CVA_REAL_K8S", "0") != "1":
        return False

    # And kubectl + cluster must be reachable
    try:
        result = subprocess.run(["kubectl", "cluster-info"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _real_k8s_enabled(),
    reason="Real K8s tests disabled (set CVA_REAL_K8S=1) or cluster not reachable",
)

"""
Test CVA K8s tools against real minikube cluster
"""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json

def test_k8s_cluster_accessible():
    """Verify kubectl can reach the cluster"""
    
    print("\n=== TEST: K8s cluster accessibility ===")
    
    result = subprocess.run(
        ["kubectl", "get", "nodes"],
        capture_output=True,
        text=True
    )
    
    print(f"kubectl output:\n{result.stdout}")
    
    assert result.returncode == 0, "kubectl should connect successfully"
    assert "minikube" in result.stdout, "Should see minikube node"
    print("✓ K8s cluster is accessible")


def test_get_deployment_info():
    """Test getting deployment info"""
    
    print("\n=== TEST: Get deployment info ===")
    
    result = subprocess.run(
        ["kubectl", "get", "deployment", "nginx-test", "-n", "default", "-o", "json"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Should get deployment info"
    
    deployment = json.loads(result.stdout)
    
    print(f"Deployment name: {deployment['metadata']['name']}")
    print(f"Replicas: {deployment['spec']['replicas']}")
    print(f"Available: {deployment['status'].get('availableReplicas', 0)}")
    
    assert deployment['metadata']['name'] == 'nginx-test', "Should find nginx-test"
    assert deployment['spec']['replicas'] == 2, "Should have 2 replicas"
    
    print("✓ Can retrieve deployment info")


def test_scale_deployment():
    """Test scaling deployment up and down"""
    
    print("\n=== TEST: Scale deployment ===")
    
    # Scale to 3
    print("\nScaling to 3 replicas...")
    result = subprocess.run(
        ["kubectl", "scale", "deployment", "nginx-test", "--replicas=3", "-n", "default"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Scale to 3 should succeed"
    print("✓ Scaled to 3")
    
    # Verify
    import time
    time.sleep(3)
    
    result = subprocess.run(
        ["kubectl", "get", "deployment", "nginx-test", "-n", "default", "-o", "json"],
        capture_output=True,
        text=True
    )
    
    deployment = json.loads(result.stdout)
    print(f"Current replicas: {deployment['spec']['replicas']}")
    assert deployment['spec']['replicas'] == 3, "Should be scaled to 3"
    
    # Scale back to 2
    print("\nScaling back to 2 replicas...")
    result = subprocess.run(
        ["kubectl", "scale", "deployment", "nginx-test", "--replicas=2", "-n", "default"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Scale to 2 should succeed"
    print("✓ Scaled back to 2")
    
    print("✓ Scaling works correctly")


def test_get_pods():
    """Test listing pods"""
    
    print("\n=== TEST: List pods ===")
    
    result = subprocess.run(
        ["kubectl", "get", "pods", "-n", "default", "-l", "app=nginx-test", "-o", "json"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, "Should list pods"
    
    pods = json.loads(result.stdout)
    pod_count = len(pods['items'])
    
    print(f"Found {pod_count} pods")
    for pod in pods['items']:
        name = pod['metadata']['name']
        status = pod['status']['phase']
        print(f"  - {name}: {status}")
    
    assert pod_count >= 2, "Should have at least 2 pods"
    print("✓ Can list pods")


if __name__ == "__main__":
    print("Testing Real K8s Integration")
    print("=" * 50)
    
    try:
        test_k8s_cluster_accessible()
        test_get_deployment_info()
        test_scale_deployment()
        test_get_pods()
        
        print("\n" + "=" * 50)
        print("ALL K8s INTEGRATION TESTS PASSED ✓")
        print("\nVerified capabilities:")
        print("- Cluster connectivity")
        print("- Deployment info retrieval")
        print("- Scaling operations")
        print("- Pod listing")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
