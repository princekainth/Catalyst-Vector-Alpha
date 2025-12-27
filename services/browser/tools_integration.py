"""Register browser tools with CVA ToolRegistry."""
from .browser_service import BrowserService

def register_browser_tools(tool_registry, browser_service: BrowserService = None):
    """Register browser tools with CVA."""
    from tool_registry import Tool
    
    service = browser_service or BrowserService()
    
    tool_registry.register_tool(Tool(
        name="web_fetch",
        func=service.web_fetch,
        description="Fetch content from URL. Returns title, text, links, code blocks.",
        parameters={"url": "string (required)", "use_cache": "bool", "extract": "bool"}
    ))
    
    tool_registry.register_tool(Tool(
        name="web_search",
        func=service.web_search,
        description="Search the web. Returns list of results with title and URL.",
        parameters={"query": "string (required)", "num_results": "int (default 5)"}
    ))
    
    tool_registry.register_tool(Tool(
        name="web_api",
        func=service.web_api,
        description="Make REST API call (GET/POST/PUT/DELETE).",
        parameters={"url": "string (required)", "method": "string", "data": "dict", "headers": "dict"}
    ))
    
    print(f"[BrowserService] Registered 3 browser tools")
    return service
