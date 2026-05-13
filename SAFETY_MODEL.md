# CVA Safety Model: Defense in Depth

Safety is not a feature of CVA; it is the core constraint. CVA uses a multi-layered defense-in-depth approach to ensure that autonomous agents never cause an outage.

## Layer 1: Tool Capability Gating
Before a tool is even considered for execution, the `ToolExecutor` checks the agent's **Identity Profile**.
- **Observer** agents can only call `SAFE` tools (read-only).
- **Worker** agents can call `DESTRUCTIVE` tools, but only through the approval gate.
- No agent can call a tool that is not explicitly registered in the `ToolRegistry`.

## Layer 2: Risk-Based Interception
Tools are categorized by risk in `capabilities.py`:
- **SAFE**: No side effects (e.g., `get_pod_status`). Executed immediately.
- **CAUTION**: Minor side effects or external networking (e.g., `web_search`). Logged heavily.
- **DESTRUCTIVE**: Changes infrastructure state (e.g., `k8s_rollout_undo`). **Hard-blocked** until approved.

## Layer 3: Trace-Bound Approval Tokens
When a `DESTRUCTIVE` tool is called, CVA returns an `approval_required` response containing a `trace_id`.
- An approval MUST be granted specifically for that `trace_id`.
- The approval generates a one-time token.
- The tool will **only** execute if the token is passed back and matches the active trace.
- This prevents "token replay" or cross-incident execution.

## Layer 4: Input Sanitization & Validation
All structured remediation tools implement strict internal validation:
- **Image Names**: No spaces, no shell metacharacters (`;`, `|`, `&`, etc.).
- **Filesystem Paths**: Restricted to allow-listed directories; no `../` traversal allowed.
- **Numeric Bounds**: Ports (1-65535), Timing (0-3600), Revisions (1-100000).

## Layer 5: Circuit Breaking
The `ToolRegistry` monitors tool success rates. If a tool fails 3 times consecutively:
- The circuit breaker trips.
- The tool is **disabled cluster-wide** for 5 minutes.
- This prevents "remediation storms" where a failing tool is repeatedly retried.

## Layer 6: Immutable Audit Trail
Every tool call, result, and approval is recorded in `audit/actions.jsonl`. This file is the source of truth for the dashboard and cannot be modified by the agent swarm.
