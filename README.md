# CVA Cloud — AI SRE Agent with Approval-Gated Remediation

**CVA detects Kubernetes incidents, explains root cause, proposes safe remediation, and executes only after policy-gated approval.**

---

## 🚀 Overview

Catalyst Vector Alpha (CVA) is an intelligent autonomous operator for Kubernetes. Unlike traditional monitoring that just alerts, CVA acts as a "First Responder":
1. **Observes**: Watches your cluster for failures (CrashLoop, OOM, Probe fails).
2. **Diagnoses**: Automatically collects logs and events to explain *why* it's failing.
3. **Proposes**: Selects a structured remediation (e.g., patching an image or reverting a rollout).
4. **Protects**: Intercepts destructive actions and waits for your approval.

## 🛡️ Safety First
- **Zero-Shell**: No raw terminal execution.
- **Strict Gating**: All state changes require a human-in-the-loop or policy token.
- **Audit Ready**: Every decision is logged with full evidence.

## 🛠️ Getting Started
See [QUICKSTART.md](QUICKSTART.md) to get running in 3 minutes.

## 📖 Documentation
- [DEMO.md](DEMO.md) - Run the incident benchmark.
- [ARCHITECTURE.md](ARCHITECTURE.md) - How the cognitive loop works.
- [SAFETY_MODEL.md](SAFETY_MODEL.md) - Why CVA is safe for production.

---
*Built for the next generation of resilient infrastructure.*
