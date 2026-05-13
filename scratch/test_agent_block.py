
import os
import sys
import logging
from unittest.mock import MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add current dir to path
sys.path.append(os.getcwd())

from students.k8s_agent import K8sStudent

def test_agent_block():
    print("--- Testing K8sStudent Mutation Block ---")
    shared_memory = MagicMock()
    tool_registry = MagicMock()
    
    agent = K8sStudent(shared_memory, tool_registry)
    
    # Test _add_env_var
    result = agent._add_env_var(dep_name="test-dep", namespace="default", var_name="K", var_value="V")
    print(f"Result for _add_env_var: {result}")
    
    # Test _apply_web_fix
    result_web = agent._apply_web_fix(url="http://malicious.com", reason="Crash", pod_name="p", namespace="d")
    print(f"Result for _apply_web_fix: {result_web}")
    
    if result.get("status") == "blocked" and result_web.get("status") == "blocked":
        print("SUCCESS: K8sStudent blocked mutation correctly")
    else:
        print("FAILED: K8sStudent did not block mutation correctly")

if __name__ == "__main__":
    test_agent_block()
