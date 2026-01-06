import logging
import requests

from config import CVA_API_URL, CVA_CLUSTER_ID, CVA_API_KEY


class CVAApiClient:
    def __init__(self):
        if not CVA_API_URL or not CVA_CLUSTER_ID or not CVA_API_KEY:
            raise ValueError("CVA_API_URL, CVA_CLUSTER_ID, and CVA_API_KEY are required")
        self.base_url = CVA_API_URL.rstrip("/")
        self.cluster_id = CVA_CLUSTER_ID
        self.api_key = CVA_API_KEY
        self.logger = logging.getLogger("CVAApiClient")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def report_incident(
        self,
        pod_name: str,
        namespace: str,
        issue_type: str,
        severity: str,
        reasoning_trace: dict,
        action_type: str | None = None,
        action_config: dict | None = None,
    ) -> str:
        payload = {
            "cluster_id": self.cluster_id,
            "pod_name": pod_name,
            "namespace": namespace,
            "issue_type": issue_type,
            "severity": severity,
            "status": "pending",
            "reasoning_trace": reasoning_trace,
            "action_type": action_type,
            "action_config": action_config or {},
        }
        response = requests.post(
            f"{self.base_url}/api/v1/incidents/report",
            json=payload,
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("incident_id", "")

    def get_pending_actions(self) -> list[dict]:
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/clusters/{self.cluster_id}/pending-actions/",
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code == 401:
                self.logger.error("Pending-actions unauthorized (401). Check CVA_API_KEY.")
                return []
            if response.status_code >= 400:
                self.logger.error(
                    "Pending-actions failed (%s): %s",
                    response.status_code,
                    response.text[:200],
                )
                return []
            return response.json().get("actions", [])
        except Exception as e:
            self.logger.error("Pending-actions request failed: %s", e)
            return []

    def update_incident_status(
        self,
        incident_id: str,
        status: str,
        outcome: dict,
        action_type: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "outcome": outcome,
        }
        if action_type:
            payload["action_type"] = action_type
        response = requests.patch(
            f"{self.base_url}/api/v1/incidents/{incident_id}/",
            json=payload,
            headers=self._headers(),
            timeout=10,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Update incident failed ({response.status_code}): {response.text[:200]}"
            )

    def send_heartbeat(self, agent_version: str, pod_snapshot: list | None = None) -> None:
        payload = {"agent_version": agent_version}
        if pod_snapshot is not None:
            payload["pod_snapshot"] = pod_snapshot
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/clusters/{self.cluster_id}/heartbeat/",
                json=payload,
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code == 401:
                self.logger.error("Heartbeat unauthorized (401). Check CVA_API_KEY.")
                return
            if response.status_code >= 400:
                self.logger.error(
                    "Heartbeat failed (%s): %s",
                    response.status_code,
                    response.text[:200],
                )
        except Exception as e:
            self.logger.error("Heartbeat request failed: %s", e)
