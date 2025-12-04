ARCHITECTURE.md — Catalyst Vector Alpha (CVA) System Architecture
Overview

Catalyst Vector Alpha (CVA) is a hybrid-intelligence agent operating system built around a continuous autonomous cognitive loop. It combines:

LLM reasoning (flexible decision-making)

Rule-based safety (hard constraints)

Semantic memory + vector search

Specialized agents and tool-based execution

CVA operates as a multi-agent ecosystem where the Planner acts as the executive cortex, coordinating a workforce of Worker, Observer, Security, Optimizer, Collector, Notifier, and DynamicAgent roles.

CVA integrates structured tools, memory, triggers, and dynamic agent creation into a cohesive, self-updating cognitive architecture.

System Flow: API → Cognitive Loop → Agents → Tools → Memory
1. API Entry (External Input)

External requests enter through endpoints like:

POST /api/command {"command": "..."}


The API constructs a directive:

type: AGENT_PERFORM_TASK

agent_name: ProtoAgent_Planner_instance_1

task_description: <user command>

This directive is injected into:

CatalystVectorAlpha.dynamic_directive_queue

2. Core Cognitive Loop (The Brain)

The loop in run_cognitive_loop() is the central brain:

Processes directives

Iterates through agents

Executes each agent’s .perform_task()

Handles stagnation detection, reflection, memory compression

Periodically applies health checks and persistence

Injects autonomous planning directives during idle cycles

This loop is where all intelligent behavior emerges.

3. Planner Agent (Executive Cortex)

ProtoAgent_Planner is the strategic decision-maker.

The Planner:

Interprets user intent

Queries memory

Detects alerts (email, calendar, K8s metrics)

Uses an LLM to reason about intent

Decides whether to act directly or spawn a specialist agent

The Planner is the OS scheduler + CEO of the agent workforce.

4. AgentFactory & DynamicAgents (Adaptive Workforce)

When a task requires specialization, Planner triggers:

AgentFactory.spawn_agent(purpose, context, parent_agent)


DynamicAgent creation uses a hybrid strategy:

A) LLM Designs the Agent

Name

System prompt

Tool selection (2–5 tools)

B) LLM Self-Validates Tools

Ensures tools match purpose:

remove unrelated domains

reduce irrelevant tools

C) Semantic Matching

Vector similarity adds missing relevant tools.

D) Hard-coded Safety Rules

Forbid dangerous or nonsensical mixes:

Gmail ❌ Kubernetes

CPU monitoring ❌ Web scraping

Result:
A minimal, safe, purpose-built specialist agent.

5. Tool Execution Layer (Hands of the System)

Tool calls route through ToolRegistry.safe_call():

Schema validation

Type coercion

Cooldown restrictions

Input normalization

Timeout protection (ThreadPoolExecutor)

Logging + memory recording

This ensures tools never:

hang

break the system

escape safety rules

mix incompatible behaviors

Tools are the action layer of CVA.

6. Memory Architecture — Short-Term, Mid-Term, Long-Term
Short-Term Memory (STM)

deque(maxlen=100)
Stores raw events:

TaskOutcome

ToolUse

IntentChange

MessageReceived

SystemAlert

Mid-Term (Compressed Memory)

LLM summarizes chunks → embedding → stored as:

type: CompressedMemory

Long-Term Memory (LTM)

ChromaDB vector store:

query_long_term_memory(query_text)


Planner and agents access LTM for:

Reflection

Planning

Pattern detection

Stagnation recovery

Backup

Each memory also persists to JSONL for transparency and replay.

7. Agents Overview (CVA’s Subsystems)
Agent	Function
Planner	Executive cortex; routes work, spawns agents, plans actions
Worker	Executes tools; performs actuations and tasks
Observer	Gathers metrics, especially K8s/system responsiveness
Collector	Pulls external data sources into memory
Security	Runs security checks; enforces system policies
Optimizer	Detects anomalies and performance issues
Notifier	Sends alerts and user-facing notifications
DynamicAgent	On-demand specialist agents with narrow prompts
8. MessageBus & Communication

All agents communicate through:

send_message()
receive_messages()


Messages become memories:

add_memory("MessageReceived", {...})


This keeps all thinking auditable and replayable.

9. Persistent State

CVA stores its state across cycles:

cva.db (SQLite): task history, agent state, snapshots

.chromadb/: vector embeddings

logs/*.jsonl: agent event streams

This makes CVA:

restartable

analyzable

trainable in the future

debuggable

10. Safety Backbone

CVA includes:

SovereignGradient (action compliance analysis)

IntentGuard (input sanitization)

InjectionLimiter (prompt/code injection detection)

Policy gates in tools

Hard-coded domain restrictions

CVA is built to avoid runaway behavior.

Conclusion

CVA is not a simple agent script.
It is a full hybrid-intelligence operating system built on top of:

autonomous cognitive cycles

LLM-driven reasoning

rule-based safety

semantic memory

dynamic specialization

safe tool execution

a structured multi-agent workforce

This document now serves as the canonical reference for all future development, AI assistance, and system extensions.

## Memory Inspection

CVA stores compressed memories in `persistence_data/chroma_db/` (NOT `.chromadb`).

### Quick Memory Check:
```bash
python3 check_cva_memory.py
```

### Query Agent Memory:
```bash
# Query specific patterns
python3 check_cva_memory.py observer kubernetes scaling
python3 check_cva_memory.py planner mission failures
python3 check_cva_memory.py worker tool execution patterns
python3 check_cva_memory.py security threat detection
```

### Key Insights from Current Session:
- **Core agents:** 895+ memories each (2+ weeks of learning)
- **Dynamic agents:** 47 specialized agents with domain knowledge
- **Embedding model:** mxbai-embed-large (1024-dim)
- **Total knowledge:** 3,600+ compressed memory entries

CVA learns through:
1. Memory compression every 5 cognitive cycles
2. Semantic vector storage for pattern retrieval
3. Cross-agent knowledge sharing via ChromaDB
