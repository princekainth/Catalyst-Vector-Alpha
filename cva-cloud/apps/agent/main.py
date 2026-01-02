import time

from api_client import CVAApiClient
from config import CHECK_INTERVAL, CVA_CLUSTER_ID, LLM_MODEL, LLM_PROVIDER
from k8s_student import K8sStudent
from llm import OllamaLLMIntegration


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
    _check_llm_config()
    api.send_heartbeat("1.0.0")

    while True:
        status = k8s_student.work()
        problem_pods = status.get("details", {}).get("problem_pods", []) if isinstance(status, dict) else []

        for pod in problem_pods:
            analysis = k8s_student.analyze(pod)
            try:
                incident_id = api.report_incident(
                    pod_name=pod.get("name", "unknown"),
                    namespace=pod.get("namespace", "default"),
                    issue_type=pod.get("reason", "Unknown"),
                    severity=calculate_severity(pod),
                    reasoning_trace=analysis.get("trace", {}),
                    action_type=analysis.get("action"),
                    action_config={
                        "recommended_actions": analysis.get("recommended_actions", []),
                        "action_plan": analysis.get("action_plan", {}),
                    },
                )
                if incident_id:
                    print(f"Reported incident {incident_id} for {pod.get('name')}")
            except Exception as e:
                print(f"Failed to report incident for {pod.get('name')}: {e}")

        actions = api.get_pending_actions()
        for action in actions:
            result = k8s_student.execute_fix(
                incident_id=action.get("incident_id", ""),
                action_type=action.get("action_type", ""),
                config=action.get("action_config", {}),
            )
            status_value = "fixed" if result.get("success") else "failed"
            api.update_incident_status(
                incident_id=action.get("incident_id", ""),
                status=status_value,
                outcome=result,
            )

        api.send_heartbeat("1.0.0")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
