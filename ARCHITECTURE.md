# ARCHITECTURE.md — The Gemini Protocol System Architecture
**Version:** 5.0 (Enterprise Pivot)
**Date:** 2025-12-10

## 1. System Overview

The **Gemini Protocol System** (formerly Catalyst Vector Alpha) is an **Autonomous SRE (Site Reliability Engineering) Platform**. It operates as a multi-layer cognitive engine designed to detect, analyze, and remediate infrastructure incidents without human intervention.

It is built on a **Three-Tier Proprietary Architecture**:
1.  **Gemini™ Protocol (Interface Layer):** Orchestration, API Gateway, and Agent Spawning.
2.  **Meta™ Intelligence (Cognitive Layer):** Swarm consensus, self-reflection, and tool synthesis.
3.  **Microsoft™ Enterprise (Infrastructure Layer):** Edge kernels, K8s integration, and deployment.

---

## 2. The Architectural Stack

### Layer 1: The Gemini™ Orchestrator (The Nervous System)
* **Role:** Central Event Loop & API Gateway.
* **Implementation:** `GeminiOrchestrator` (formerly `CatalystVectorAlpha`).
* **Responsibilities:**
    * **The "Tick":** Runs the continuous `run_cognitive_loop()`.
    * **The Factory:** Uses `GeminiAgentFactory` to spawn agents via REST API.
    * **The Router:** Injects directives into the `dynamic_directive_queue`.

### Layer 2: Meta™ Intelligence (The Brain)
* **Role:** Agent Cognition & Swarm Logic.
* **Implementation:** `MetaCognitiveArchitecture` & `MetaSwarmCoordinator`.
* **Responsibilities:**
    * **Perception:** `perceive_event()` (Seeing the crash).
    * **Reflection:** `analyze_and_adapt()` (Realizing *why* it crashed).
    * **Toolsmithing:** Dynamically generating tools (e.g., `toolsmith_generate`) when standard tools fail.

### Layer 3: Microsoft™ Enterprise Kernel (The Hands)
* **Role:** Infrastructure Execution & Persistence.
* **Implementation:** `MicrosoftAgentKernel` & `MicrosoftEnterpriseAI`.
* **Responsibilities:**
    * **K8s Interface:** Direct hooks into `kubernetes_pod_metrics` and `watch_k8s_events`.
    * **State Management:** Writing to `cva.db` (SQLite) and `chroma_db` (Vector Store).
    * **Edge Deployment:** Single-file kernel execution for lightweight nodes.

---

## 3. Data Flow: From Chaos to Cure

### Step 1: External Signal (The Trigger)
* **Source:** Kubernetes Cluster (e.g., Pod Crash) or User API Request.
* **Ingestion:**
    * **Autonomous:** `GeminiOrchestrator` polls `watch_k8s_events`.
    * **Manual:** POST request to `/api/agents/spawn`.

### Step 2: Cognitive Processing (The Thinking)
* The **Meta™ Cognitive Layer** analyzes the signal.
* **Context Retrieval:** Queries `MemeticKernel` (Vector DB) for similar past incidents.
* **Intent Formation:** "I see a crash. My purpose is monitoring. I must alert."

### Step 3: Swarm Consensus (The Validation)
* If multiple agents are active (e.g., `EnterpriseMonitor` + `Workflow_Analyzer`), the **Meta™ Swarm Coordinator** aligns their intents to prevent conflicting actions.

### Step 4: Execution (The Action)
* The **Microsoft™ Kernel** executes the tool:
    * `tools.kubernetes.get_pod_logs`
    * `tools.slack.send_alert`

---

## 4. Memory Architecture

The system uses a **Bicameral Memory System** to ensure agents learn from every crash.

| Memory Type | Storage | Purpose |
| :--- | :--- | :--- |
| **Short-Term (STM)** | `deque(maxlen=100)` | Raw event logs (Directives, Messages, Errors). |
| **Long-Term (LTM)** | **ChromaDB (Vector)** | Semantic storage of "Lessons Learned" and "Task Outcomes." |
| **Archival** | `.jsonl` Files | Immutable audit logs for compliance (`memetic_archive_*.jsonl`). |

---

## 5. Active Agent Roles

The workforce has evolved from generic roles to specialized **Protocol Agents**.

| Agent Name | Protocol | Responsibility |
| :--- | :--- | :--- |
| **EnterpriseMonitor** | Gemini™ | **The Sentry.** Watches K8s events, audit logs, and network traffic. |
| **Workflow_Analyzer** | Meta™ | **The Architect.** Analyzes business processes and builds automation tools. |
| **Planner** | Gemini™ | **The CEO.** Routes high-level directives to the correct specialist. |
| **Security** | Microsoft™ | **The Guard.** Enforces RBAC and policy constraints on tool execution. |

---

## 6. Chaos Validation Status

The architecture is **Battle-Hardened**.
* **Validation Date:** 2025-12-10
* **Score:** 7/10 (Stable Alpha)

**Passed Scenarios:**
* ✅ **Pod Kill:** Detected immediate termination.
* ✅ **CPU Spike:** Detected resource pressure (OOM).
* ✅ **CrashLoop:** Detected repeated failure cycles.
* ✅ **Secrets Audit:** Detected unauthorized access attempts.

---

## 7. Persistence & State

* **Database:** `cva.db` (SQLite) - Stores Agent State, Task History, and Metrics.
* **Vector Store:** `persistence_data/chroma_db/` - Stores Semantic Embeddings.
* **Audit Logs:** `logs/swarm_activity.jsonl` - The "Black Box" flight recorder.

## 8. Conclusion

The Gemini Protocol System is no longer just a script; it is a **self-healing, autonomous infrastructure platform**. It combines the speed of rule-based monitoring with the adaptability of Large Language Models, wrapped in an enterprise-grade architecture.

**System Status:** `PRODUCTION_READY` (Alpha)
