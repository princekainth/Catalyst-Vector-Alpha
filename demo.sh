#!/bin/bash
# demo.sh - Unified CVA Demonstration Script

# Environment Setup
export CVA_DEMO_MODE=1
export CVA_DISABLE_GMAIL_AGENT=1
export PYTHONPATH=.

echo "================================================================================"
echo "   CVA ALPHA: AI SRE AGENT WITH APPROVAL-GATED REMEDIATION   "
echo "================================================================================"

echo ""
echo "--- [1/4] STEP 1: SECURITY & VALIDATION CHECK ---"
echo "Verifying that the safety model blocks malicious inputs and enforced bounds..."
python3 scratch/remediation_tool_validation.py
if [ $? -ne 0 ]; then echo "Validation failed. Aborting."; exit 1; fi

echo ""
echo "--- [2/4] STEP 2: LOCAL SYSTEM CONTROL DEMO ---"
echo "Demonstrating safe host monitoring and gated service management..."
python3 scratch/system_demo_flow.py
if [ $? -ne 0 ]; then echo "System demo failed. Aborting."; exit 1; fi

echo ""
echo "--- [3/4] STEP 3: KUBERNETES 5-INCIDENT BENCHMARK ---"
echo "Simulating real-world SRE failures and validating CVA's diagnosis logic..."
export KEEP_CVA_TESTS=0
python3 scratch/test_5_k8s_incidents.py
if [ $? -ne 0 ]; then echo "Incident benchmark failed. Aborting."; exit 1; fi

echo ""
echo "================================================================================"
echo "   DEMO COMPLETE: ALL SAFETY GATES AND DIAGNOSTIC CHECKS PASSED   "
echo "================================================================================"
