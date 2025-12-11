# Autonomous K8s Remediation (Observer)

## What’s detected
- Pod event issues via `watch_k8s_events` (Warnings/critical reasons)
- Pod health via `get_pod_status` fallback:
  - Phases: Failed, Error, CrashLoopBackOff
  - Issues: OOMKilled, ImagePullBackOff, ErrImagePull, ImageInspectError, CrashLoopBackOff
  - Skips pods with restarts > 10 (assumed stuck)

## How it remediates
1. Observer calls `watch_k8s_events` (namespace=all, minutes=10).
2. If `critical_count > 0`, directly calls `microsoft_autonomous_remediation` on up to 3 critical pods (deduped).
3. Always runs `get_pod_status` (rate-limited) to remediate failing pods not covered by events.
4. Sends a desktop notification on remediation (if tool available).
5. Deduplication: won’t remediate the same pod again within 10 minutes.

## Configuration
- Event lookback: 10 minutes (see Observer code in `agents.py`).
- Fallback rate limit: 60 seconds between `get_pod_status` runs.
- Dedup TTL: 10 minutes per pod.

## Disable
- Remove `get_pod_status` / `watch_k8s_events` from Observer tool access or comment out the K8s monitoring blocks in `agents.py`.
