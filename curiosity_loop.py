#!/usr/bin/env python3
"""
CVA Curiosity Loop - Self-directed learning when idle
"""
import time
import threading
from datetime import datetime
from tool_registry import tool_registry as GLOBAL_TOOL_REGISTRY
from shared_models import OllamaLLMIntegration
from shared_memory import SharedMemory

class CuriosityLoop:
    def __init__(self, cycle_time: int = 120):
        self.llm = OllamaLLMIntegration()
        self.memory = SharedMemory()
        self.tools = GLOBAL_TOOL_REGISTRY
        self.cycle_time = cycle_time
        self.running = False
        self.thread = None
        
        # Topics to explore based on CVA's domain
        self.base_interests = [
            "kubernetes self-healing",
            "autonomous infrastructure",
            "eBPF observability",
            "chaos engineering",
            "AI SRE automation",
            "cost optimization kubernetes",
            "predictive autoscaling",
        ]
        
        self.explored = set()
        self.discoveries = []
        
    def start(self):
        """Start the curiosity loop in background."""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[Curiosity] 🧠 Started - exploring every {self.cycle_time}s")
        
    def stop(self):
        """Stop the loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("[Curiosity] Stopped")
        
    def _loop(self):
        """Main curiosity loop."""
        while self.running:
            try:
                self._explore()
            except Exception as e:
                print(f"[Curiosity] Error: {e}")
            
            # Wait for next cycle
            for _ in range(self.cycle_time):
                if not self.running:
                    break
                time.sleep(1)
    
    def _explore(self):
        """One exploration cycle."""
        print(f"\n[Curiosity] 🔭 Exploring... ({datetime.now().strftime('%H:%M:%S')})")
        
        # 1. Pick a topic
        topic = self._pick_topic()
        if not topic:
            print("[Curiosity] No new topics to explore")
            return
        
        print(f"[Curiosity] 📚 Topic: {topic}")
        self.explored.add(topic)
        
        # 2. Search web
        search_result = self._search(topic)
        if not search_result:
            return
        
        # 3. Read top result
        knowledge = self._read_and_learn(search_result)
        if not knowledge:
            return
        
        # 4. Identify capability gaps
        gaps = self._find_gaps(knowledge)
        
        # 5. Store discoveries
        discovery = {
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "source": search_result.get('url', ''),
            "title": search_result.get('title', ''),
            "knowledge_preview": knowledge[:500],
            "capability_gaps": gaps,
        }
        self.discoveries.append(discovery)
        
        # 6. Store in shared memory
        try:
            if hasattr(self.memory, 'add_memory'):
                self.memory.add_memory(
                    agent_name="CuriosityLoop", text=str(discovery), category="CuriosityDiscovery",
                    # content moved to text
                    
                )
                print(f"[Curiosity] 🧠 Stored discovery in memory")
        except Exception as e:
            print(f"[Curiosity] Memory store failed: {e}")
        
        # 7. Report
        print(f"[Curiosity] ✨ Discovered: {search_result.get('title', 'unknown')[:50]}")
        if gaps:
            print(f"[Curiosity] 💡 Capability gap found: {gaps[:100]}")
    
    def _pick_topic(self) -> str:
        """Pick next topic to explore."""
        # First, check recent failures in memory for relevant topics
        try:
            if hasattr(self.memory, 'search'):
                failures = self.memory.search("failed error", limit=3)
                if failures:
                    # Ask LLM to extract a search topic from failures
                    prompt = f"""Based on these recent failures, suggest ONE search topic to learn more:
{failures[:500]}

Reply with just the search query, nothing else."""
                    topic = self.llm.generate_text(prompt).strip()
                    if topic and topic not in self.explored and len(topic) < 100:
                        return topic
        except:
            pass
        
        # Otherwise pick from base interests
        for topic in self.base_interests:
            if topic not in self.explored:
                return topic + " 2025"
        
        return None
    
    def _search(self, query: str) -> dict:
        """Search web for topic."""
        try:
            result = self.tools.safe_call('web_search', query=query, max_results=3)
            
            if result.get('status') != 'ok':
                return None
            
            data = result.get('data', {}).get('data', {})
            results = data.get('results', [])
            
            if results:
                top = results[0]
                return {
                    "title": top.get('title', ''),
                    "url": top.get('href', top.get('url', '')),
                    "snippet": top.get('body', ''),
                }
            return None
        except Exception as e:
            print(f"[Curiosity] Search failed: {e}")
            return None
    
    def _read_and_learn(self, search_result: dict) -> str:
        """Read article and extract knowledge."""
        url = search_result.get('url', '')
        if not url:
            return None
        
        try:
            print(f"[Curiosity] 📖 Reading: {url[:50]}...")
            result = self.tools.safe_call('read_webpage', url=url)
            
            if result.get('status') != 'ok':
                return None
            
            data = result.get('data', {}).get('data', {})
            content = data.get('content', '')[:3000]
            
            return content
        except Exception as e:
            print(f"[Curiosity] Read failed: {e}")
            return None
    
    def _find_gaps(self, knowledge: str) -> str:
        """Ask LLM what capabilities we're missing."""
        available_tools = self.tools.list_tool_names()[:20] if hasattr(self.tools, 'list_tool_names') else []
        
        prompt = f"""I am CVA, an autonomous Kubernetes management AI.

My current tools: {', '.join(available_tools)}

I just read this article:
{knowledge[:1500]}

Based on this, what ONE specific capability am I missing that would make me better?
Be specific. Reply in one sentence."""

        try:
            response = self.llm.generate_text(prompt)
            return response.strip()
        except:
            return ""
    
    def get_discoveries(self) -> list:
        """Return all discoveries."""
        return self.discoveries
    
    def get_pending_requests(self) -> list:
        """Return capability gaps found."""
        return [d['capability_gaps'] for d in self.discoveries if d.get('capability_gaps')]


def main():
    """Test the curiosity loop."""
    print("╭" + "─" * 50 + "╮")
    print("│" + " CVA Curiosity Loop Test ".center(50) + "│")
    print("╰" + "─" * 50 + "╯")
    
    loop = CuriosityLoop(cycle_time=60)  # Explore every 60s for testing
    loop.start()
    
    try:
        print("\nCuriosity loop running. Press Ctrl+C to stop.\n")
        print("Commands: 'discoveries', 'requests', 'quit'\n")
        
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'discoveries':
                for d in loop.get_discoveries():
                    print(f"\n📚 {d['topic']}")
                    print(f"   {d['title'][:60]}")
                    print(f"   Gap: {d['capability_gaps'][:80]}")
            elif cmd == 'requests':
                reqs = loop.get_pending_requests()
                print(f"\n💡 {len(reqs)} capability requests:")
                for r in reqs:
                    print(f"   - {r[:80]}")
            else:
                print("Commands: discoveries, requests, quit")
                
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        
    print("\n\n=== Final Discoveries ===")
    for d in loop.get_discoveries():
        print(f"\n📚 {d['topic']}")
        print(f"   Source: {d['source'][:60]}")
        print(f"   Gap: {d['capability_gaps']}")


if __name__ == "__main__":
    main()
