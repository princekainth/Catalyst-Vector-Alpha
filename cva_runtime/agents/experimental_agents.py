"""Experimental/non-v1 agents.

Phase 1 split keeps implementation source in agents_legacy.py.
"""

from agents_legacy import (
    MetaCognitiveArchitecture,
    ProtoAgent_Collector,
    ProtoAgent_Optimizer,
)

__all__ = [
    "MetaCognitiveArchitecture",
    "ProtoAgent_Collector",
    "ProtoAgent_Optimizer",
]
