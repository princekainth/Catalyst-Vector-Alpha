import time

from api_client import CVAApiClient
from config import CHECK_INTERVAL, CVA_CLUSTER_ID, LLM_MODEL, LLM_PROVIDER
from k8s_student import K8sStudent
from llm import OllamaLLMIntegration

REPORT_TTL_SECONDS = 600
_reported_incidents: dict[str, float] = {}


def calculate_severity(pod: dict) -> str:
    severity_map = {
        "OOMKilled": "critical",
        "CrashLoopBackOff": "critical",
        "CreateContainerConfigError": "high",
        "ImagePullBackOff": "high",
        "ErrImagePull": "high",
        "NotReady": "medium",
        "Pending": "low",
    }
    return severity_map.get(pod.get("reason"), "low")


def _check_llm_config() -> None:
    print(f"CVA Agent LLM provider: {LLM_PROVIDER}, model: {LLM_MODEL}")
    # Instantiation validates provider + required keys without network calls.
    OllamaLLMIntegration()


def main() -> None:
    api = CVAApiClient()
    k8s_student = K8sStudent(shared_memory=None, tool_registry=None)

    print(f"CVA Agent starting for cluster {CVA_CLUSTER_ID}")
    print("✅ CVA Agent booted (k8s mode). Beginning heartbeat + scan loop...", flush=True)
    _check_llm_config()
    api.send_heartbeat("1.0.0")
    print("💓 Heartbeat sent successfully", flush=True)

    while True:
        status = k8s_student.scan()
        problem_pods = status.get("problem_pods", []) if isinstance(status, dict) else []

        for pod in problem_pods:
            pod_name = pod.get("name", "unknown")
            namespace = pod.get("namespace", "default")
            issue_type = pod.get("reason", "Unknown")
            if namespace == "cva-system" and pod_name.startswith("cva-agent-"):
                print(f"[IGNORE] Skipping self pod {namespace}/{pod_name}")
                continue

            analysis = k8s_student.analyze(pod)
            if analysis.get("skipped"):
                continue
            report_key = f"{CVA_CLUSTER_ID}:{namespace}/{pod_name}:{issue_type}"
            now = time.time()
            last_report = _reported_incidents.get(report_key)
            if last_report and (now - last_report) < REPORT_TTL_SECONDS:
                print(f"[DEDUP] Skipping report for {report_key}")
                continue
            try:
                incident_id = api.report_incident(
                    pod_name=pod_name,
                    namespace=namespace,
                    issue_type=issue_type,
                    severity=calculate_severity(pod),
                    reasoning_trace=analysis.get("trace", {}),
                    action_type=analysis.get("action"),
                    action_config={
                        "recommended_actions": analysis.get("recommended_actions", []),
                        "action_plan": analysis.get("action_plan", {}),
                    },
                )
                if incident_id:
                    _reported_incidents[report_key] = now
                    print(f"Reported incident {incident_id} for {pod_name}")
            except Exception as e:
                print(f"Failed to report incident for {pod_name}: {e}")

        actions = api.get_pending_actions()
        for action in actions:
            result = k8s_student.execute_fix(
                incident_id=action.get("incident_id", ""),
                action_type=action.get("action_type", ""),
                config=action.get("action_config", {}),
            )
            status_value = "fixed" if result.get("success") else "failed"
            try:
                api.update_incident_status(
                    incident_id=action.get("incident_id", ""),
                    status=status_value,
                    outcome=result,
                )
            except Exception as e:
                print(f"Failed to update incident {action.get('incident_id')}: {e}")

        api.send_heartbeat("1.0.0")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
