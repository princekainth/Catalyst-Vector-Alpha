import os
import unittest
from unittest.mock import patch

from cva_runtime.control_plane.capabilities import Capability, ToolProfile, ToolRisk
from cva_runtime.control_plane.policy import evaluate


class TestPolicy(unittest.TestCase):
    def test_denies_missing_capabilities(self):
        profile = ToolProfile(
            name="danger_tool",
            required_caps=frozenset({Capability.K8S_WRITE}),
            risk=ToolRisk.SAFE,
        )
        decision = evaluate(
            agent_id="observer_1",
            tool_name="danger_tool",
            args={},
            tool_profile=profile,
            agent_capabilities=frozenset(),
        )
        self.assertFalse(decision.allow)
        self.assertFalse(decision.requires_approval)
        self.assertIn("lacks capabilities", decision.reason)

    def test_destructive_requires_approval_in_manual_mode(self):
        profile = ToolProfile(
            name="k8s_scale",
            required_caps=frozenset({Capability.K8S_WRITE}),
            risk=ToolRisk.DESTRUCTIVE,
        )
        with patch.dict(os.environ, {"CVA_APPROVAL_MODE": "manual"}, clear=False):
            decision = evaluate(
                agent_id="worker_1",
                tool_name="k8s_scale",
                args={"namespace": "default", "deployment": "api", "replicas": 2},
                tool_profile=profile,
                agent_capabilities=frozenset({Capability.K8S_WRITE}),
            )
        self.assertFalse(decision.allow)
        self.assertTrue(decision.requires_approval)
        self.assertIsNotNone(decision.approval_token)

    def test_allows_safe_tool_with_required_caps(self):
        profile = ToolProfile(
            name="read_tool",
            required_caps=frozenset({Capability.K8S_READ}),
            risk=ToolRisk.SAFE,
        )
        decision = evaluate(
            agent_id="observer_1",
            tool_name="read_tool",
            args={},
            tool_profile=profile,
            agent_capabilities=frozenset({Capability.K8S_READ}),
        )
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_approval)

    def test_unknown_mode_denies_by_default(self):
        profile = ToolProfile(
            name="k8s_scale",
            required_caps=frozenset({Capability.K8S_WRITE}),
            risk=ToolRisk.DESTRUCTIVE,
        )
        with patch.dict(os.environ, {"CVA_APPROVAL_MODE": "mystery_mode"}, clear=False):
            decision = evaluate(
                agent_id="worker_1",
                tool_name="k8s_scale",
                args={"namespace": "default", "deployment": "api", "replicas": 2},
                tool_profile=profile,
                agent_capabilities=frozenset({Capability.K8S_WRITE}),
            )
        self.assertFalse(decision.allow)
        self.assertFalse(decision.requires_approval)
        self.assertIn("unknown approval mode", decision.reason)


if __name__ == "__main__":
    unittest.main()
