#!/usr/bin/env python3
"""
CVA Teacher Console - REAL tool execution
"""
import subprocess
from tool_registry import tool_registry as GLOBAL_TOOL_REGISTRY
from shared_models import OllamaLLMIntegration
from shared_memory import SharedMemory

class TeacherConsole:
    def __init__(self):
        self.llm = OllamaLLMIntegration()
        self.memory = SharedMemory()
        self.tools = GLOBAL_TOOL_REGISTRY
        
    def chat(self, user_input: str) -> str:
        lower = user_input.lower()
        
        # REAL web search
        if "search" in lower and "web" in lower:
            query = user_input.split("for", 1)[-1].strip() if "for" in lower else user_input
            return self._real_web_search(query)
        
        # REAL cluster check
        if any(w in lower for w in ["cluster", "pod", "k8s", "kubernetes", "status"]):
            return self._real_cluster_status(user_input)
        
        # REAL memory check
        if any(w in lower for w in ["learn", "remember", "memory", "recent"]):
            return self._real_memory_check()
        
        # REAL capability gaps based on actual failures
        if any(w in lower for w in ["request", "need", "missing", "gap"]):
            return self._real_capability_gaps()
        
        # General chat - but with context
        return self._contextual_chat(user_input)
    
    def _real_web_search(self, query: str) -> str:
        """Actually search the web."""
        print(f"  [🔍 Searching: {query}]")
        
        result = self.tools.safe_call('web_search', query=query, max_results=3)
        
        if result.get('status') != 'ok':
            return f"Search failed: {result.get('error', 'unknown')}"
        
        data = result.get('data', {}).get('data', {})
        results = data.get('results', [])
        
        if not results:
            return "No results found."
        
        # Format real results
        output = "Here's what I found:\n\n"
        for i, r in enumerate(results[:3], 1):
            title = r.get('title', 'No title')
            url = r.get('href', r.get('url', ''))
            body = r.get('body', '')[:150]
            output += f"{i}. **{title}**\n   {url}\n   {body}...\n\n"
        
        return output
    
    def _real_cluster_status(self, query: str) -> str:
        """Actually check the cluster."""
        print("  [🔎 Checking cluster...]")
        
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-A", "-o", "wide"],
                capture_output=True, text=True, timeout=10
            )
            pods_output = result.stdout if result.returncode == 0 else "kubectl failed"
            
            # Count issues
            lines = pods_output.strip().split('\n')[1:]  # Skip header
            total = len(lines)
            unhealthy = sum(1 for l in lines if any(x in l for x in ['Error', 'CrashLoop', 'Pending', 'ImagePull']))
            
            # Ask LLM to analyze
            prompt = f"""Cluster status:
{pods_output}

User asked: {query}

Summary: {total} pods, {unhealthy} unhealthy.

Give a brief, direct answer about the cluster status."""
            
            return self.llm.generate_text(prompt)
            
        except Exception as e:
            return f"Error checking cluster: {e}"
    
    def _real_memory_check(self) -> str:
        """Actually check SharedMemory."""
        print("  [🧠 Checking memory...]")
        
        try:
            # Get recent memories
            if hasattr(self.memory, 'get_recent'):
                recent = self.memory.get_recent(5)
            elif hasattr(self.memory, 'search'):
                recent = self.memory.search("", limit=5)
            else:
                recent = []
            
            if not recent:
                return "My memory is empty. I haven't learned anything yet in this session."
            
            output = f"I have {len(recent)} recent memories:\n\n"
            for i, mem in enumerate(recent, 1):
                if isinstance(mem, dict):
                    mem_type = mem.get('type', mem.get('memory_type', 'unknown'))
                    content = str(mem.get('content', mem))[:100]
                    output += f"{i}. [{mem_type}] {content}...\n"
                else:
                    output += f"{i}. {str(mem)[:100]}...\n"
            
            return output
            
        except Exception as e:
            return f"Error reading memory: {e}"
    
    def _real_capability_gaps(self) -> str:
        """Check REAL failures and gaps."""
        print("  [📊 Analyzing capability gaps...]")
        
        # Check recent failures in memory
        failures = []
        try:
            if hasattr(self.memory, 'search'):
                failures = self.memory.search("failed error", limit=5)
        except:
            pass
        
        # Check what tools exist
        available_tools = self.tools.list_tool_names() if hasattr(self.tools, 'list_tool_names') else []
        
        prompt = f"""I am a CVA Teacher agent managing Kubernetes infrastructure.

My available tools: {', '.join(available_tools[:20])}

Recent failures/issues I've seen: {failures[:3] if failures else 'None recorded yet'}

Based on REAL Kubernetes operations needs, what 2-3 tools am I actually missing?
Be specific to K8s/infrastructure. Not generic business tools."""
        
        return self.llm.generate_text(prompt)
    
    def _contextual_chat(self, user_input: str) -> str:
        """General chat with real context."""
        tools = self.tools.list_tool_names()[:15] if hasattr(self.tools, 'list_tool_names') else []
        
        prompt = f"""You are a CVA Teacher agent, part of Catalyst Vector Alpha - an autonomous AI system built by Prince Kainth. You manage Kubernetes infrastructure.

Your REAL tools: {', '.join(tools)}

You can:
- Actually search the web (say "search web for X")
- Actually check cluster (ask about pods/cluster)
- Actually check memory (ask what you learned)

User: {user_input}

Respond directly and honestly. If you need to do something, tell the user to ask you to do it."""
        
        return self.llm.generate_text(prompt)


def main():
    print("╭" + "─" * 50 + "╮")
    print("│" + " CVA Teacher Console (REAL) ".center(50) + "│")
    print("├" + "─" * 50 + "┤")
    print("│ This console executes REAL actions:".ljust(51) + "│")
    print("│   'search web for X' - actual web search".ljust(51) + "│")
    print("│   'cluster status' - actual kubectl".ljust(51) + "│")  
    print("│   'what did you learn' - actual memory".ljust(51) + "│")
    print("│   'what do you need' - real gaps analysis".ljust(51) + "│")
    print("│   'exit' - quit".ljust(51) + "│")
    print("╰" + "─" * 50 + "╯")
    print()
    
    console = TeacherConsole()
    
    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ")
            
            if user_input.lower() in ('exit', 'quit', 'q'):
                print("Goodbye.")
                break
            
            if not user_input.strip():
                continue
            
            response = console.chat(user_input)
            print(f"\033[94mTeacher:\033[0m {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
