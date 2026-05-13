# CVA Demo Guide

This guide walks you through the automated incident remediation demo.

## 1. Automated Incident Benchmark

The primary demo is the "5-Incident Benchmark". It simulates a cluster experiencing five distinct failure modes simultaneously:

1. **CrashLoopBackOff**: An application crashing on startup.
2. **ImagePullBackOff**: A deployment with a non-existent image.
3. **OOMKilled**: A workload exceeding its memory limits.
4. **Failed Probe**: A service failing its liveness/readiness health checks.
5. **Bad Rollout**: A deployment stuck due to configuration errors.

### How to Run:
```bash
./demo.sh
```

### What to Look For:
- **Detection**: CVA will identify the failure within 15-30 seconds.
- **Evidence**: CVA will automatically pull logs and `kubectl describe` events.
- **Classification**: Observe the intelligent classification (e.g., distinguishing between a transient crash and a persistent image pull error).
- **Security Gate**: Note that CVA **stops** before executing any remediation. It will return `approval_required`.

## 2. Local System Demo

CVA can also manage the host machine (safe system monitoring).

### Run:
```bash
python3 scratch/system_demo_flow.py
```

### Demonstration:
- CVA checks disk, memory, and CPU.
- CVA attempts to restart a systemd service (e.g., `cva-demo-service`).
- The security gate blocks the restart until a valid token is provided.

## 3. Security Validation

To prove the safety model is unbreakable even with valid tokens:
```bash
python3 scratch/remediation_tool_validation.py
```
This tests edge cases like path traversal in logs or command injection in image names.
