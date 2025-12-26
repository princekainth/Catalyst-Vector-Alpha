"""
Students Module - Domain-specific autonomous agents.

Students are workers that:
- Run in their own threads (never block teachers)
- Handle specific domains (K8s, Email, Calendar, etc.)
- Report back via shared memory
- Learn from past experiences

Usage:
    from students import K8sStudent
    
    k8s = K8sStudent(shared_memory=memory, tool_registry=tools)
    k8s.start()  # Runs in background
    
    # Check status anytime
    print(k8s.get_status())
    
    # Stop when done
    k8s.stop()
"""

from .base_student import BaseStudent
from .k8s_agent import K8sStudent

__all__ = [
    "BaseStudent",
    "K8sStudent",
]