#!/bin/bash
set -e

echo "======================================"
echo "Running CVA Test Suite"
echo "======================================"

# Validation tests (no K8s needed)
echo -e "\n[1/5] Worker Validation Tests..."
python3 tests/test_worker_validation.py

echo -e "\n[2/5] Directive Format Tests..."
python3 tests/test_directive_format.py

echo -e "\n[3/5] Retry & Blocking Tests..."
python3 tests/test_retry_blocking.py

echo -e "\n[4/5] K8s Event Flow (Simulated)..."
python3 tests/test_k8s_event_flow.py

# K8s integration tests (requires minikube)
if kubectl get nodes &>/dev/null; then
    echo -e "\n[5/5] Real K8s Integration Tests..."
    python3 tests/test_real_k8s_integration.py
    python3 tests/test_cva_k8s_tools.py
else
    echo -e "\n[5/5] Skipping K8s tests (no cluster running)"
fi

echo -e "\n======================================"
echo "✓ ALL TESTS PASSED"
echo "======================================"
