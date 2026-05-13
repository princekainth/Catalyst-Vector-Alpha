import pytest

# Adjust import if your project structure differs.
# If agents.py is at repo root, this works:
from agents import ProtoAgent_Planner


def _make_planner():
    # ProtoAgent_Planner should be constructible without extra deps.
    # If your __init__ requires args, adapt here.
    return ProtoAgent_Planner(name="ProtoAgent_Planner_test")


@pytest.mark.parametrize("mission_type", ["k8s_monitoring", "general_planning"])
def test_repair_normalize_keeps_microsoft_autonomous_remediation_for_k8s_and_general(mission_type):
    planner = _make_planner()

    raw_steps = [
        {"i": 1, "agent": "ProtoAgent_Worker_instance_1", "tool": "microsoft_autonomous_remediation", "title": "Fix pod"},
        {"i": 2, "agent": "ProtoAgent_Observer_instance_1", "tool": "get_pod_status", "title": "Check pod"},
    ]

    repaired = planner._repair_and_normalize_steps(raw_steps, mission_type=mission_type)

    tools = [s.get("tool") for s in repaired]
    assert "microsoft_autonomous_remediation" in tools, f"remediation step was dropped for {mission_type}"


def test_repair_normalize_does_not_rewrite_or_drop_remediation_even_if_mapped_from_verb_for_k8s_monitoring():
    planner = _make_planner()

    # If your verb_map maps "fix_pod" -> microsoft_autonomous_remediation, this ensures it survives.
    raw_steps = [
        {"i": 1, "agent": "ProtoAgent_Worker_instance_1", "tool": "fix_pod", "title": "Fix failing pod"},
    ]

    repaired = planner._repair_and_normalize_steps(raw_steps, mission_type="k8s_monitoring")

    tools = [s.get("tool") for s in repaired]
    assert "microsoft_autonomous_remediation" in tools or "fix_pod" in tools
    # Accept either outcome depending on whether mapping happens before/after your hard whitelist.
    # The key behavior is: do not drop the step.


def test_repair_normalize_allows_remediation_only_in_k8s_or_general_not_in_other_missions():
    planner = _make_planner()

    raw_steps = [
        {"i": 1, "agent": "ProtoAgent_Worker_instance_1", "tool": "microsoft_autonomous_remediation", "title": "Fix pod"},
    ]

    repaired_other = planner._repair_and_normalize_steps(raw_steps, mission_type="health_audit")

    # Depending on your existing policy rules, this may be dropped or rewritten/downgraded.
    # We assert it is NOT guaranteed-kept in non-k8s/non-general missions.
    tools_other = [s.get("tool") for s in repaired_other]
    assert "microsoft_autonomous_remediation" not in tools_other, (
        "remediation should not be auto-whitelisted for non-k8s/non-general missions"
    )
