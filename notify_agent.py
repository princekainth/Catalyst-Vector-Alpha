# notify_agent.py
from agents import ProtoAgent # Import the base class from your existing agents.py
from typing import Optional, Dict, Any

class ProtoAgent_Notifier(ProtoAgent):
    """
    A specialized worker agent for sending notifications to the user.
    Its only job is to execute notification tasks given by the Planner.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eidos_spec["role"] = "notification_executor"

    def _execute_agent_specific_task(self, task_description, **kwargs):
        """Required by base class - delegates to perform_task"""
        return self.perform_task(task_description, **kwargs)
    
    def perform_task(
        self,
        task_description: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple:
        """
        Notifier-specific task execution.
        It expects a 'send_desktop_notification' task.
        """
        # Filter out spam notifications
        if "No specific intent" in task_description:
            return ("completed", None, {"summary": "Skipped generic notification"}, 0.0)
        
        # Also check message content
        message_content = (tool_args or {}).get('message', '')
        if "No specific intent" in str(message_content):
            return ("completed", None, {"summary": "Skipped generic notification"}, 0.0)
        
        print(f"[NotifierAgent] Received task: {task_description}")
        
        # This agent only has one job.
        # We'll default to 'send_desktop_notification' if not specified
        if not tool_name:
            tool_name = "send_desktop_notification"
            
        if tool_name != "send_desktop_notification":
            msg = f"NotifierAgent cannot perform tool: {tool_name}"
            return "failed", msg, {"summary": msg}, 0.0

        # Extract arguments
        title = (tool_args or {}).get("title", "CVA Notification")
        message = (tool_args or {}).get("message", task_description) # Use task desc as fallback

        # Get the tool from the registry (which will be passed in by app.py)
        registry = getattr(self, "tool_registry", None)
        if not (registry and registry.has_tool(tool_name)):
            msg = f"Tool '{tool_name}' not found in registry."
            return "failed", msg, {"summary": msg}, 0.0
        
        # --- Execute the Tool ---
        try:
            result = registry.safe_call(
                tool_name,
                title=title,
                message=message
            )
            
            if result and result.get("ok") is True:
                summary = f"Notification sent: {result.get('title', 'N/A')}"
                return "completed", None, {"summary": summary}, 1.0
            else:
                msg = result.get("result", "Notification tool failed.")
                return "failed", msg, {"summary": msg}, 0.0
                
        except Exception as e:
            msg = f"Error calling notification tool: {e}"
            return "failed", msg, {"summary": msg}, 0.0