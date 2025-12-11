# agent_factory.py - Dynamic agent spawning and lifecycle management
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import threading
import numpy as np
from config_manager import get_config

from database import CVADatabase
from agents import ProtoAgent
from tool_registry import ToolRegistry

# PHASE 2: Semantic Tool Matching imports
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_MATCHING_ENABLED = True
except ImportError:
    SEMANTIC_MATCHING_ENABLED = False
    logging.warning("sentence-transformers not installed - semantic matching disabled")

_config = get_config()
_features = _config.get("features", {})
_semantic_matching_enabled = _features.get("semantic_tool_matching", True) and SEMANTIC_MATCHING_ENABLED

logger = logging.getLogger("CatalystLogger")

@dataclass
class AgentSpec:
    """Specification for a dynamically created agent."""
    agent_id: str
    name: str
    purpose: str
    specialized_prompt: str
    tools: List[str]
    ttl_hours: float
    created_at: datetime
    expires_at: datetime
    parent_agent: str  # Which agent spawned this

class DynamicAgent(ProtoAgent):
    """Agent created on-demand by AgentFactory."""
    
    def __init__(self, spec: AgentSpec, tool_registry: ToolRegistry, db: CVADatabase,
                 message_bus, event_monitor, external_log_sink, chroma_db_path, 
                 persistence_dir, paused_agents_file_path, world_model):
        self.spec = spec
        self.db = db
        self.is_expired = False
        
        # Create minimal eidos_spec
        eidos_spec = {
            "role": "dynamic_agent",
            "initial_intent": spec.purpose,
            "location": "autonomous"
        }
        
        # Base agent init with all required params
        super().__init__(
            name=spec.name,
            eidos_spec=eidos_spec,
            message_bus=message_bus,
            event_monitor=event_monitor,
            external_log_sink=external_log_sink,
            chroma_db_path=chroma_db_path,
            persistence_dir=persistence_dir,
            paused_agents_file_path=paused_agents_file_path,
            world_model=world_model,
            tool_registry=tool_registry
        )
        
        logger.info(f"DynamicAgent spawned: {spec.name} (expires in {spec.ttl_hours}h)")
    
    def _execute_agent_specific_task(self, task_description: str, cycle_id: Optional[str], 
                                      reporting_agents: Optional[Union[str, List]], 
                                      context_info: Optional[dict], **kwargs):
        """Execute task with specialized tools."""
        if self.check_expiration():
            return "failed", "Agent expired", {}, 0.0
        
        # Use parent's generic execution but with our specialized tools
        return "completed", None, {"summary": f"Executed: {task_description}"}, 0.8
    
    
    def check_expiration(self) -> bool:
        """Check if agent has expired."""
        if datetime.now(timezone.utc) > self.spec.expires_at:
            self.is_expired = True
            logger.info(f"Agent {self.spec.name} expired")
            return True
        return False
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with Tool-First Protocol"""
        from datetime import datetime
        
        if self.check_expiration():
            return {"success": False, "error": "Agent expired"}
        
        task_description = task.get("description", str(task))
        current_year = datetime.now().year
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # STEP 1: Try to find existing tool
        decision_prompt = f"""
    You are {self.spec.name}. Your purpose: {self.spec.purpose}

    Task: {task_description}

    Available tools: {', '.join(self.spec.tools)}
    Current date: {current_date}, Current year: {current_year}

    DECISION PROTOCOL:
    1. Can ANY of your available tools solve this task?
    - If YES: Use that tool
    - If NO: Respond with "TOOLSMITH_MODE"

    Respond with JSON:
    {{"decision": "USE_TOOL" or "TOOLSMITH_MODE", "tool": "tool_name", "args": {{}}, "reasoning": "why"}}

    If no tool fits, set decision to "TOOLSMITH_MODE" and explain what code is needed.
    """

        try:
            response = self.ollama_inference_model.generate_text(decision_prompt, temperature=0.2)
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            
            import json, re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            decision = json.loads(response)
            
            # STEP 2: Execute based on decision
            if decision.get("decision") == "TOOLSMITH_MODE":
                # Try toolsmith generation if enabled
                if getattr(self.tool_registry, "toolsmith_enabled", False):
                    try:
                        gen = self.tool_registry.safe_call("toolsmith_generate", task=task_description)
                        if isinstance(gen, dict) and gen.get("status") == "ok":
                            gen_name = gen["data"].get("tool_name")
                            if gen_name:
                                self.spec.tools.append(gen_name)
                                result = self.tool_registry.safe_call(gen_name)
                                self.db.record_dynamic_agent_task(
                                    agent_id=self.spec.agent_id,
                                    task=task,
                                    result=result
                                )
                                return {"success": True, "result": result}
                    except Exception:
                        pass
                # If toolsmith not used or failed, return helpful error
                return {
                    "success": False,
                    "error": "No matching tool available for this task",
                    "agent_tools": self.spec.tools,
                    "task": task_description,
                    "suggestion": "Spawn a new agent with a purpose that matches this task, or rephrase the task to use available tools",
                    "available_tools_hint": f"This agent has: {', '.join(self.spec.tools)}"
                }
            else:
                # Use existing tool
                tool_name = decision.get("tool")
                tool_args = decision.get("args", {})

                if tool_name in self.spec.tools:
                    result = self.tool_registry.safe_call(tool_name, **tool_args)

                    # Fallback handling for bad args
                    if isinstance(result, dict) and not result.get("ok"):
                        if "unexpected keyword argument" in str(result.get("error", "")):
                            result = self.tool_registry.safe_call(tool_name, query=task_description)
                        elif tool_name == "check_calendar":
                            time_min = f"{current_date}T00:00:00Z"
                            time_max = f"{current_date}T23:59:59Z"
                            result = self.tool_registry.safe_call(tool_name, time_min_utc=time_min, time_max_utc=time_max)
                else:
                    # Tool not available - try a smart fallback before failing
                    result = None
                    if "web_search" in self.spec.tools:
                        result = self.tool_registry.safe_call("web_search", query=task_description)
                    elif "read_webpage" in self.spec.tools:
                        # heuristically extract a URL
                        import re
                        url_match = re.search(r"https?://\\S+", task_description)
                        if url_match:
                            result = self.tool_registry.safe_call("read_webpage", url=url_match.group(0))
                    # Toolsmith fallback (sandboxed codegen)
                    if result is None and getattr(self.tool_registry, "toolsmith_enabled", False):
                        try:
                            gen = self.tool_registry.safe_call("toolsmith_generate", task=task_description)
                            if isinstance(gen, dict) and gen.get("status") == "ok":
                                gen_name = gen["data"].get("tool_name")
                                if gen_name:
                                    self.spec.tools.append(gen_name)
                                    result = self.tool_registry.safe_call(gen_name)
                        except Exception:
                            result = None
                    if result is None:
                        available = ", ".join(self.spec.tools)
                        return {
                            "success": False,
                            "error": f"Tool '{tool_name}' not available for this agent",
                            "suggestion": f"This agent has: {available}",
                            "hint": "Try a task that uses one of these tools, or spawn a new agent with different purpose"
                        }
            
            # Track in database
            self.db.record_dynamic_agent_task(
                agent_id=self.spec.agent_id,
                task=task,
                result=result
            )
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _toolsmith_mode(self, task_description: str, reasoning: str) -> dict:
        """Generate and execute custom code when no tool exists"""
        code_prompt = f"""
    Task: {task_description}
    Reason for code generation: {reasoning}

    Write Python code to solve this task. Requirements:
    - Use standard libraries (os, sys, subprocess, json, requests)
    - Return result as JSON dict
    - Handle errors gracefully

    Output ONLY the Python code, no explanations.
    """
        
        try:
            code = self.ollama_inference_model.generate_text(code_prompt, temperature=0.3)
            
            # Clean code
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0].strip()
            elif '```' in code:
                code = code.split('```')[1].split('```')[0].strip()
            
            # Execute in sandbox (simplified - add proper sandboxing)
            import subprocess
            result = subprocess.run(
                ['python3', '-c', code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {"ok": True, "output": result.stdout, "method": "toolsmith"}
            else:
                return {"ok": False, "error": result.stderr, "method": "toolsmith"}
                
        except Exception as e:
            return {"ok": False, "error": f"Toolsmith failed: {str(e)}"}

class AgentFactory:
    """Spawns and manages dynamic agents."""

    # Vague keywords that indicate a purpose is too ambiguous
    VAGUE_KEYWORDS = [
        "help", "assist", "improve", "optimize", "better", "good", "nice",
        "manage", "handle", "work", "do", "make", "get", "thing"
    ]

    # Suggested specific purposes for common vague requests
    PURPOSE_SUGGESTIONS = {
        "productivity": [
            "Monitor system CPU and memory usage every 5 minutes",
            "Track disk usage and alert when above 90%",
            "Search for productivity tips and summarize results"
        ],
        "monitoring": [
            "Monitor system performance and alert on high CPU",
            "Track Kubernetes pod health and restart failed pods",
            "Watch disk space and notify when low"
        ],
        "research": [
            "Search for AI research papers from 2025",
            "Find latest news on quantum computing",
            "Research cryptocurrency market trends"
        ],
        "default": [
            "Monitor system CPU and memory usage",
            "Search for AI research papers",
            "Track Kubernetes deployment health",
            "Research latest technology news"
        ]
    }

    def __init__(self, db: CVADatabase, tool_registry: ToolRegistry, llm):
        self.db = db
        self.tool_registry = tool_registry
        self.llm = llm
        self.active_agents: Dict[str, DynamicAgent] = {}
        self.lock = threading.Lock()

        # PHASE 2: Initialize semantic matching
        self.semantic_model = None
        self.tool_embeddings = {}
        if _semantic_matching_enabled:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
            try:
                logger.info("Loading Ollama embedding model for semantic tool matching...")
                # Use the same embedding model as the rest of CVA
                self.semantic_model = self.llm  # Use existing Ollama LLM integration
                self._precompute_tool_embeddings()
                logger.info(f"Semantic tool matching ready ({len(self.tool_embeddings)} tools embedded)")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                self.semantic_model = None

        # Initialize factory table
        self._init_db()
        logger.info("AgentFactory initialized")

    def _validate_purpose(self, purpose: str) -> Dict[str, Any]:
        """
        Validate that purpose is specific enough to create a useful agent.
        Rejects vague purposes and provides helpful suggestions.

        Returns:
            {"valid": True} if purpose is specific enough
            {"valid": False, "error": str, "suggestions": list} if too vague
        """
        purpose_lower = purpose.lower().strip()

        # Check for empty or too short
        if len(purpose_lower) < 10:
            return {
                "valid": False,
                "error": "Purpose too short - be more specific about what you need",
                "suggestions": self.PURPOSE_SUGGESTIONS["default"]
            }

        # Count vague keywords
        vague_count = sum(1 for word in self.VAGUE_KEYWORDS if word in purpose_lower.split())
        word_count = len(purpose_lower.split())

        # If >50% of words are vague, reject
        if word_count > 0 and vague_count / word_count > 0.5:
            # Find relevant suggestions based on keywords
            suggestions = self.PURPOSE_SUGGESTIONS["default"]
            for category, keywords in [
                ("productivity", ["productiv", "efficien", "time"]),
                ("monitoring", ["monitor", "watch", "track", "alert"]),
                ("research", ["research", "find", "search", "news"])
            ]:
                if any(kw in purpose_lower for kw in keywords):
                    suggestions = self.PURPOSE_SUGGESTIONS[category]
                    break

            return {
                "valid": False,
                "error": f"Purpose too vague: '{purpose}'. Be specific about WHAT to do and HOW.",
                "suggestions": suggestions
            }

        # Check for actionable verbs
        actionable_verbs = [
            "monitor", "search", "track", "tracker", "tracking", "find", "check", "analyze", "scan",
            "watch", "detect", "alert", "notify", "report", "fetch", "query",
            "research", "investigate", "audit", "verify", "validate"
        ]
        # Allow stem matches (e.g., "tracker"/"tracking" counts for "track")
        has_action = any(
            verb in purpose_lower.split() or any(w.startswith(verb) for w in purpose_lower.split())
            for verb in actionable_verbs
        )

        if not has_action:
            return {
                "valid": False,
                "error": "Purpose needs an action verb (monitor, search, track, analyze, etc.)",
                "suggestions": [
                    f"Monitor {purpose}",
                    f"Search for {purpose}",
                    f"Track {purpose} and alert on changes"
                ]
            }

        return {"valid": True}

    # ============== PHASE 2: Semantic Tool Matching ==============

    def _precompute_tool_embeddings(self):
        """Pre-compute embeddings for all registered tools."""
        if not self.semantic_model:
            return

        tool_specs = self.tool_registry.get_all_tool_specs()
        for spec in tool_specs:
            tool_name = spec['name']
            # Create rich description for better semantic matching
            description = f"{tool_name}: {spec.get('description', '')}".strip()
            try:
                embedding = self.semantic_model.generate_embedding(description)
                self.tool_embeddings[tool_name] = {
                    'embedding': embedding,
                    'description': description
                }
            except Exception as e:
                logger.warning(f"Failed to embed tool {tool_name}: {e}")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def cleanup_expired_agents(self):
        """Remove expired DynamicAgents from registry."""
        expired = []
        
        for agent_id, agent in list(self.active_agents.items()):
            if hasattr(agent, 'check_expiration') and agent.check_expiration():
                expired.append(agent_id)
        
        for agent_id in expired:
            agent = self.active_agents.pop(agent_id)
            logger.info(f"[CLEANUP] Removed expired agent: {agent_id}")
            
            # Remove from orchestrator if registered
            if self.orchestrator and hasattr(self.orchestrator, 'agent_instances'):
                self.orchestrator.agent_instances.pop(agent_id, None)
        
        return len(expired)

    def find_semantic_tools(self, purpose: str, top_k: int = 5, threshold: float = 0.35) -> List[Tuple[str, float]]:
        """
        Find most semantically relevant tools for a given purpose.

        Args:
            purpose: The task/purpose description
            top_k: Maximum number of tools to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (tool_name, similarity_score) tuples, sorted by relevance
        """
        if not self.semantic_model or not self.tool_embeddings:
            logger.warning("Semantic matching not available, returning empty list")
            return []

        try:
            # Embed the purpose
            purpose_embedding = self.semantic_model.generate_embedding(purpose)

            # Compute similarities
            scores = []
            for tool_name, data in self.tool_embeddings.items():
                similarity = self._cosine_similarity(purpose_embedding, data['embedding'])
                if similarity >= threshold:
                    scores.append((tool_name, similarity))

            # Sort by similarity descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_k
            return scores[:top_k]

        except Exception as e:
            logger.error(f"Semantic tool search failed: {e}")
            return []

    def get_semantic_suggestions(self, purpose: str) -> Dict[str, Any]:
        """
        Get tool suggestions with confidence scores for debugging/display.
        """
        matches = self.find_semantic_tools(purpose, top_k=8, threshold=0.3)
        return {
            "purpose": purpose,
            "matches": [
                {"tool": name, "confidence": round(score, 3)}
                for name, score in matches
            ],
            "semantic_enabled": self.semantic_model is not None
        }

    # ============== End Semantic Tool Matching ==============

    def _init_db(self):
        """Create factory tracking table (PostgreSQL handles this in schema)."""
        # Table already created in PostgreSQL schema
        pass
    
    def _init_db_old(self):
        """Legacy table creation - no longer needed."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dynamic_agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT,
                purpose TEXT,
                specialized_prompt TEXT,
                tools_json TEXT,
                ttl_hours REAL,
                created_at TEXT,
                expires_at TEXT,
                parent_agent TEXT,
                status TEXT,
                tasks_completed INTEGER DEFAULT 0,
                terminated_at TEXT
            )
        ''')
        conn.commit()
    
    def spawn_agent(self,
                purpose: str,
                context: Dict[str, Any],
                parent_agent: str,
                ttl_hours: float = 24.0) -> Union[DynamicAgent, Dict[str, Any]]:
        """
        Spawn a specialized agent for a specific purpose.

        LLM decides:
        - Agent name
        - Specialized prompt
        - Required tools (with self-validation)

        Returns:
            DynamicAgent if successful
            Dict with error/suggestions if purpose validation fails
        """

        # PHASE 1: Validate purpose before proceeding
        validation = self._validate_purpose(purpose)
        if not validation.get("valid"):
            logger.warning(f"Agent spawn rejected - {validation.get('error')}")
            return {
                "success": False,
                "error": validation.get("error"),
                "suggestions": validation.get("suggestions", []),
                "hint": "Try a more specific purpose with an action verb"
            }

        # Get tool specs with descriptions
        tool_specs = self.tool_registry.get_all_tool_specs()
        tools_info = "\n".join([
            f"- {spec['name']}: {spec['description']}" 
            for spec in tool_specs
        ])
        
        # Ask LLM to design the agent
        design_prompt = f"""
    You are CVA's Agent Factory. Design a specialized agent for this purpose:

    PURPOSE: {purpose}

    CONTEXT:
    {json.dumps(context, indent=2)}

    AVAILABLE TOOLS (choose only what's needed):
    {tools_info}

    CRITICAL RULES FOR TOOL SELECTION:
    1. Choose ONLY 2-4 tools that are DIRECTLY needed for the purpose
    2. Each tool must serve a clear function for this specific task
    3. NEVER mix unrelated domains:
    - Email/Gmail tasks → web_search, read_webpage, analyze_text_sentiment, send_desktop_notification
    - Calendar tasks → check_calendar, send_desktop_notification
    - Kubernetes tasks → kubernetes_pod_metrics, find_wasteful_deployments, k8s_scale, k8s_restart
    - Security tasks → analyze_threat_signature, isolate_network_segment, send_desktop_notification
    - System monitoring → disk_usage, get_system_cpu_load, top_processes, send_desktop_notification
    - Web research → web_search, read_webpage, analyze_text_sentiment

    4. FORBIDDEN COMBINATIONS (these make no sense):
    - Email monitoring + kubernetes tools
    - Web research + system monitoring tools
    - Calendar checks + security tools
    - Kubernetes + web_search/read_webpage

    Respond with JSON:
    {{
        "name": "Short descriptive name (e.g. 'Receipt_Auditor')",
        "specialized_prompt": "Detailed system prompt for this agent's specific role",
        "required_tools": ["tool1", "tool2"],
        "reasoning": "Explain why EACH tool is essential for this purpose"
    }}

    Think step-by-step: What is the core task? What data sources? What actions? Pick ONLY matching tools.
    """
        
        try:
            # STEP 1: Get initial design from LLM
            response = self.llm.generate_text(design_prompt, temperature=0.3)
            
            # Clean LLM response
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            design = json.loads(response)
            
            # Validate required fields
            if not all(k in design for k in ["name", "specialized_prompt", "required_tools"]):
                raise ValueError(f"Missing required fields in LLM response: {design.keys()}")
            
            # STEP 2: LLM self-validates its tool choices
            validation_prompt = f"""
    Review this agent design for the purpose: "{purpose}"

    Selected tools: {design["required_tools"]}

    Self-check:
    1. Is EACH tool directly needed for "{purpose}"?
    2. Are there any tools from unrelated domains?
    3. Can you accomplish the goal with fewer tools?
    4. Are you mixing incompatible domains (e.g., email + kubernetes)?

    Respond with JSON:
    {{
        "approved_tools": ["only essential tools"],
        "removed": ["tools that don't fit"],
        "reasoning": "brief explanation"
    }}

    Be strict - remove ANY tool that isn't absolutely necessary.
    """
            
            validation_response = self.llm.generate_text(validation_prompt, temperature=0.1)
            validation_response = validation_response.strip()
            if '```json' in validation_response:
                validation_response = validation_response.split('```json')[1].split('```')[0].strip()
            
            json_match = re.search(r'\{.*\}', validation_response, re.DOTALL)
            if json_match:
                validation_response = json_match.group(0)
                
            validation = json.loads(validation_response)
            tools = validation.get("approved_tools", design["required_tools"])
            
            logger.info(f"Agent Factory: LLM self-validation removed: {validation.get('removed', [])}")

        except Exception as e:
            logger.warning(f"LLM validation failed, using original tools: {e}")
            tools = design.get("required_tools", [])

        # STEP 2.5: PHASE 2 - Augment with semantic tool matching
        # This helps catch tools the LLM missed
        if self.semantic_model and len(tools) < 4:
            semantic_matches = self.find_semantic_tools(purpose, top_k=5, threshold=0.4)
            semantic_tools = [name for name, score in semantic_matches if name not in tools]

            if semantic_tools:
                # Add up to 2 semantically relevant tools the LLM missed
                for tool in semantic_tools[:2]:
                    if len(tools) < 5:
                        tools.append(tool)
                        logger.info(f"Agent Factory: Semantic matching added tool '{tool}'")

        # STEP 3: Safety rules - hard constraints
        purpose_lower = purpose.lower()
        
        # Rule 1: Email/Gmail - remove K8s and system monitoring
        if any(word in purpose_lower for word in ["gmail", "email", "receipt", "message"]):
            tools = [t for t in tools if not any(banned in t.lower() for banned in 
                    ["kubernetes", "k8s", "pod", "get_system_cpu", "get_system_resource", "top_processes"])]
            if "send_desktop_notification" not in tools:
                tools.append("send_desktop_notification")
        
        # Rule 2: Kubernetes - remove web/email tools
        elif any(word in purpose_lower for word in ["kubernetes", "k8s", "deployment", "pod", "container"]):
            tools = [t for t in tools if t not in ["read_webpage", "web_search", "analyze_text_sentiment"]]
        
        # Rule 3: Web research - remove K8s and system tools
        elif any(word in purpose_lower for word in ["web", "news", "research", "search"]):
            tools = [t for t in tools if not any(banned in t.lower() for banned in 
                    ["kubernetes", "k8s", "get_system_cpu", "top_processes", "disk_usage"])]
        
        # Rule 4: System monitoring - remove K8s and web tools
        elif any(word in purpose_lower for word in ["disk", "cpu", "memory", "system resource"]):
            tools = [t for t in tools if not any(banned in t.lower() for banned in 
                    ["kubernetes", "k8s", "web_search", "read_webpage"])]
            if "send_desktop_notification" not in tools:
                tools.append("send_desktop_notification")
        
        # Rule 5: Security - ensure notification tool
        elif any(word in purpose_lower for word in ["security", "threat", "alert", "intrusion"]):
            if "send_desktop_notification" not in tools:
                tools.append("send_desktop_notification")
        
        # Rule 6: Calendar - keep it simple
        elif "calendar" in purpose_lower:
            tools = [t for t in tools if t in ["check_calendar", "send_desktop_notification"]]
            if not tools:
                tools = ["check_calendar", "send_desktop_notification"]

        # Rule 7: Trackers/news/prices - ensure web tools + notifier
        if any(k in purpose_lower for k in ["track", "tracker", "price", "monitor", "news"]):
            if "web_search" not in tools:
                tools.append("web_search")
            if "read_webpage" not in tools:
                tools.append("read_webpage")
            if "send_desktop_notification" not in tools:
                tools.append("send_desktop_notification")
        
        # Limit to 5 tools maximum
        design["required_tools"] = tools[:5]
        
        logger.info(f"Agent Factory: Final tools for '{purpose}': {design['required_tools']}")
        
        # Create spec
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=ttl_hours)
        
        spec = AgentSpec(
            agent_id=agent_id,
            name=design["name"],
            purpose=purpose,
            specialized_prompt=design["specialized_prompt"],
            tools=design["required_tools"],
            ttl_hours=ttl_hours,
            created_at=now,
            expires_at=expires,
            parent_agent=parent_agent
        )
        
        # Create agent
        from shared_models import MessageBus, EventMonitor, SharedWorldModel
        
        agent = DynamicAgent(
            spec=spec,
            tool_registry=self.tool_registry,
            db=self.db,
            message_bus=MessageBus(),
            event_monitor=EventMonitor(),
            external_log_sink=logger,
            chroma_db_path="persistence_data/chroma_db",
            persistence_dir="persistence_data",
            paused_agents_file_path="persistence_data/paused_agents.json",
            world_model=SharedWorldModel(external_log_sink=logger)
        )
        
        # Register
        with self.lock:
            self.active_agents[agent_id] = agent
        
        # Persist to DB
        self._persist_spec(spec, design.get("reasoning", ""))
        
        logger.info(f"Spawned {spec.name} ({agent_id}): {purpose}")
        return agent
        
    def _persist_spec(self, spec: AgentSpec, reasoning: str):
        """Save agent spec to database."""
        import json
        from db_postgres import execute_query
        execute_query('''
            INSERT INTO dynamic_agents 
            (agent_id, name, purpose, specialized_prompt, tools_json, 
             ttl_hours, created_at, expires_at, parent_agent, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            spec.agent_id,
            spec.name,
            spec.purpose,
            spec.specialized_prompt,
            json.dumps(spec.tools),
            spec.ttl_hours,
            spec.created_at.isoformat(),
            spec.expires_at.isoformat(),
            spec.parent_agent,
            "active"
        ))
    
    def kill_agent(self, agent_id: str) -> bool:
        """Terminate an agent."""
        with self.lock:
            if agent_id in self.active_agents:
                agent = self.active_agents.pop(agent_id)
                agent.is_expired = True
                
                # Update DB
                from db_postgres import execute_query
                execute_query('''
                    UPDATE dynamic_agents 
                    SET status = 'terminated', terminated_at = %s
                    WHERE agent_id = %s
                ''', (datetime.now(timezone.utc).isoformat(), agent_id))
                
                logger.info(f"Killed agent {agent.spec.name}")
                return True
        return False
    
    def cleanup_expired(self) -> int:
        """Remove expired agents. Returns count removed."""
        count = 0
        with self.lock:
            expired = [
                aid for aid, agent in self.active_agents.items()
                if agent.check_expiration()
            ]
            for aid in expired:
                self.kill_agent(aid)
                count += 1
        return count
    
    def list_active(self) -> List[Dict[str, Any]]:
        """List all active agents."""
        with self.lock:
            return [
                {
                    "agent_id": spec.agent_id,
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "tools": spec.tools,
                    "created_at": spec.created_at.isoformat(),
                    "expires_at": spec.expires_at.isoformat(),
                    "parent": spec.parent_agent
                }
                for spec in [a.spec for a in self.active_agents.values()]
            ]
    
    def get_agent(self, agent_id: str) -> Optional[DynamicAgent]:
        """Get agent by ID."""
        with self.lock:
            return self.active_agents.get(agent_id)


# ==========================================
# ==========================================
# GEMINI™ PROTOCOL BRANDING LAYER
# ==========================================
GeminiAgentFactory = AgentFactory


# ==========================================
# MICROSOFT™ ENTERPRISE IMPLEMENTATION
# ==========================================
class MicrosoftAgentKernel(GeminiAgentFactory):
    """Microsoft™ edge-compatible agent deployment system"""
    pass