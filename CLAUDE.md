# CLAUDE.md — CVA Developer Instructions

## Project Overview
**Catalyst Vector Alpha (CVA)** is an autonomous Semantic Operating System prototype. It manages Kubernetes clusters using a swarm of specialized agents.

## Core Architecture
- **Agents:** (Planner, Observer, Security, Worker, Notifier, Evolution).
- **Brain:** Chroma DB/Postgres via `shared_memory.py`.
- **Evolutions:** Self-improving tools found in `evolved_tools/`.
- **Logic:** Event-driven; system goes idle (`_should_be_idle()`) when K8s is healthy.

## Coding Standards
- **Logging:** Use `self.log_sink` or `logging.getLogger("CatalystLogger")`. Avoid raw `print()`.
- **Returns:** Tools must return `{"success": bool, "message": str, "data": dict}`.
- **Async:** System is multi-threaded; ensure thread safety when modifying `SharedMemory`.
- **Imports:** Prefer internal singletons: `OllamaLLMIntegration`, `SharedMemory`, `AgentFactory`.

## Evolutionary Guardrails
When assisting with tool generation:
1. **Free APIs Only:** Never suggest tools that require paid API keys unless explicitly asked.
2. **Sandbox First:** Always suggest testing code in a separate process before integration.
3. **K8s Safe:** Ensure no heavy polling loops that might trigger cluster resource alerts.

## Common Tasks
- **Start System:** `python3 app.py`
- **Check Status:** `python3 semantic_dashboard.py`
- **Test Tools:** `python3 tools.py`
- **Deploy Hotfix:** Add to `evolved_tools/` and restart loop.
