import os
import time
import unittest
from unittest.mock import patch

from cva_runtime.api.app import create_app
from cva_runtime.control_plane.approvals import ApprovalStore


class TestApprovals(unittest.TestCase):
    def test_issue_and_validate_success(self):
        store = ApprovalStore(default_ttl_seconds=60)
        token, _ = store.issue(
            trace_id="tr_1",
            tool="k8s_scale",
            args_hash="hash_1",
            agent_id="worker_1",
        )

        ok, reason = store.validate(
            token=token,
            trace_id="tr_1",
            tool="k8s_scale",
            args_hash="hash_1",
            agent_id="worker_1",
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "approval token valid")

        ok2, reason2 = store.validate(
            token=token,
            trace_id="tr_1",
            tool="k8s_scale",
            args_hash="hash_1",
            agent_id="worker_1",
        )
        self.assertFalse(ok2)
        self.assertIn(reason2, {"approval token already used", "approval token not found or expired"})

    def test_rejects_mismatch(self):
        store = ApprovalStore(default_ttl_seconds=60)
        token, _ = store.issue(
            trace_id="tr_2",
            tool="k8s_scale",
            args_hash="hash_2",
            agent_id="worker_1",
        )

        ok, reason = store.validate(
            token=token,
            trace_id="tr_other",
            tool="k8s_scale",
            args_hash="hash_2",
            agent_id="worker_1",
            consume=False,
        )
        self.assertFalse(ok)
        self.assertIn("trace mismatch", reason)

    def test_expired_token_fails(self):
        store = ApprovalStore(default_ttl_seconds=1)
        token, _ = store.issue(
            trace_id="tr_3",
            tool="k8s_scale",
            args_hash="hash_3",
            agent_id="worker_1",
            ttl_seconds=1,
        )
        time.sleep(1.2)

        ok, reason = store.validate(
            token=token,
            trace_id="tr_3",
            tool="k8s_scale",
            args_hash="hash_3",
            agent_id="worker_1",
        )
        self.assertFalse(ok)
        self.assertIn("expired", reason)

    def test_issue_endpoint_returns_token(self):
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        with patch.dict(os.environ, {"CVA_APPROVAL_TTL_S": "300"}, clear=False):
            resp = client.post(
                "/api/approvals/issue",
                json={
                    "trace_id": "tr_api",
                    "tool": "k8s_scale",
                    "args_hash": "hash_api",
                    "agent_id": "worker_1",
                },
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["data"]["approval_token"].startswith("appr_"))
        self.assertGreaterEqual(payload["data"]["expires_in_s"], 1)


if __name__ == "__main__":
    unittest.main()
