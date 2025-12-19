"""
Worker guardrails tests.

NOTE:
The imported _validate() function returns an *iterable of required arg names that are present*
(e.g., (), ("deployment",), ("deployment","replicas")), not (ok, msg).
These tests assert required args presence accordingly.
"""

import pytest

# Import whatever validator you currently exposed in agents.py
try:
    from agents import validate_worker_step_args as _validate
except Exception:  # pragma: no cover
    # Fall back to any other exported name if you used a different one
    from agents import validate_required_args_present as _validate  # type: ignore


REQUIRED_FOR_K8S_SCALE = {"deployment", "replicas"}


def _present_required(step: dict) -> set[str]:
    """
    _validate(step) returns iterable of present required args.
    Normalize to a set[str] robustly.
    """
    out = _validate(step)

    # None -> treat as empty
    if out is None:
        return set()

    # If it's a single string, treat it as one field name
    if isinstance(out, str):
        return {out}

    # Otherwise assume iterable of strings
    try:
        return set(out)
    except TypeError:
        return set()


def test_missing_required_args():
    step = {"tool": "k8s_scale", "args": {}}
    present = _present_required(step)
    missing = REQUIRED_FOR_K8S_SCALE - present
    assert missing == {"deployment", "replicas"}


def test_partial_args():
    step = {"tool": "k8s_scale", "args": {"deployment": "default/foo"}}
    present = _present_required(step)
    missing = REQUIRED_FOR_K8S_SCALE - present
    assert missing == {"replicas"}


def test_valid_args_structure():
    step = {"tool": "k8s_scale", "args": {"deployment": "default/foo", "replicas": 1}}
    present = _present_required(step)
    missing = REQUIRED_FOR_K8S_SCALE - present
    assert missing == set()
