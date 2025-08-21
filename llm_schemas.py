# llm_schemas.py
from typing import Optional, Dict, Any

class ToolProposalSchema:
    tool_name: Optional[str]
    tool_args: Dict[str, Any]

    def __init__(self, tool_name: Optional[str], tool_args: Dict[str, Any]):
        self.tool_name = tool_name
        self.tool_args = tool_args
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        tool_name = data.get("tool_name")
        tool_args = data.get("tool_args", {})
        if not isinstance(tool_args, dict):
            raise ValueError("'tool_args' must be a dictionary.")
        return cls(tool_name, tool_args)

# You might also have other schema definitions here as your project evolves.
