import json
import os
import tempfile
import unittest
from unittest.mock import patch

from cva_runtime.control_plane.approvals import issue_approval_token
from cva_runtime.control_plane.tool_executor import ToolExecutor


class FakeRegistry:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"status": "ok", "data": {"done": True}, "error": None, "summary": None}

    def _safe_call_direct(self, tool_name, timeout_seconds=None, **kwargs):
        self.calls.append(
            {
                "tool_name": tool_name,
                "timeout_seconds": timeout_seconds,
                "kwargs": dict(kwargs),
            }
        )
        return self.result


class TestToolExecutor(unittest.TestCase):
    def test_denied_unknown_tool_never_executes(self):
        fake = FakeRegistry()
        executor = ToolExecutor(registry=fake)

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "CVA_AUDIT_LOG_PATH": f"{tmpdir}/audit.jsonl",
                "CVA_APPROVAL_MODE": "manual",
            },
            clear=False,
        ):
            result = executor.execute(agent_id="worker", tool_name="unknown_tool", args={})

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["code"], "policy_denied")
            self.assertEqual(len(fake.calls), 0)

            with open(f"{tmpdir}/audit.jsonl", "r", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            self.assertTrue(any(r["decision"] == "POLICY_DECISION" for r in records))

    def test_approval_required_never_executes(self):
        fake = FakeRegistry()
        executor = ToolExecutor(registry=fake)

        args = {"namespace": "default", "deployment": "api", "replicas": 2, "approval": "auto"}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "CVA_AUDIT_LOG_PATH": f"{tmpdir}/audit.jsonl",
                "CVA_APPROVAL_MODE": "manual",
            },
            clear=False,
        ):
            result = executor.execute(
                agent_id="worker",
                tool_name="k8s_scale",
                args=args,
                trace_id="tr_need_approval",
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["code"], "approval_required")
            self.assertEqual(len(fake.calls), 0)

    def test_valid_token_executes_once_and_audits(self):
        fake = FakeRegistry()
        executor = ToolExecutor(registry=fake)

        args = {"namespace": "default", "deployment": "api", "replicas": 2, "approval": "auto"}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "CVA_AUDIT_LOG_PATH": f"{tmpdir}/audit.jsonl",
                "CVA_APPROVAL_MODE": "manual",
            },
            clear=False,
        ):
            first = executor.execute(
                agent_id="worker",
                tool_name="k8s_scale",
                args=args,
                trace_id="tr_exec",
            )
            self.assertEqual(first["code"], "approval_required")

            approval = first["approval"]
            token, _ = issue_approval_token(
                trace_id=approval["trace_id"],
                tool=approval["tool"],
                args_hash=approval["args_hash"],
                agent_id="worker",
            )

            second = executor.execute(
                agent_id="worker",
                tool_name="k8s_scale",
                args={**args, "approval_token": token},
                trace_id="tr_exec",
            )
            self.assertEqual(second["status"], "ok")
            self.assertEqual(second["code"], "ok")
            self.assertEqual(len(fake.calls), 1)

            with open(f"{tmpdir}/audit.jsonl", "r", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]

            self.assertTrue(any(r["decision"] == "TOOL_EXEC_START" for r in records))
            self.assertTrue(
                any(
                    r["decision"] == "POLICY_DECISION"
                    and isinstance(r.get("extra"), dict)
                    and isinstance(r["extra"].get("approval"), dict)
                    and r["extra"]["approval"].get("valid") is True
                    for r in records
                )
            )


if __name__ == "__main__":
    unittest.main()
