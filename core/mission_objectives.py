# core/mission_objectives.py
from datetime import datetime

# A dictionary of standing goals the Planner can choose from when idle.
goal_driven_tasks = {
    "security_audit": [
        "Monitor for new CVEs related to our tech stack (Python, Torch, LXML)",
        "Check our primary GitHub repositories for new security alerts",
        "Scan the local network for any new or unauthorized devices",
        f"Verify all critical security controls are functional as of {datetime.now().strftime('%Y-%m-%d')}"
    ],
    "performance_optimization": [
        "Identify resource bottlenecks affecting system performance over the last hour",
        "Analyze memory allocation patterns and propose optimizations",
        "Plan for potential capacity increases based on recent usage trends"
    ],
    "maintenance": [
        "Document any significant system changes or agent updates from the last 24 hours",
        "Validate that the system's backup and persistence mechanisms are working correctly",
        "Update the system's internal knowledge base with key findings from the latest mission cycle"
    ],
    "k8s_monitoring": [
        "Monitor Kubernetes cluster for incidents, pod failures, and unhealthy states. Alert on critical events."
    ],
    "sandbox_inspection": [
        "Use execute_terminal_command to check what operating system the sandbox is running (uname -a)",
        "Verify Python version in the sandbox using execute_terminal_command",
        "List all files in /workspace using execute_terminal_command",
        "Check available disk space in the sandbox using execute_terminal_command (df -h)"
    ]
}
