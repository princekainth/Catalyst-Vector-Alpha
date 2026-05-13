"""
Evolution Recorder — Replayable Evaluation Harness

Records every evolution cycle as a structured JSON snapshot that can be
replayed later to detect regressions. Each snapshot captures:
  - The triggering directive / capability-gap description
  - The plan produced by the Planner
  - The tool code diff (before / after)
  - Sandbox test outputs
  - Final promotion status

Snapshots are stored under `.cva/evolution_runs/<timestamp>_<tool_name>/`.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional


_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".cva",
    "evolution_runs",
)


class EvolutionRecorder:
    """Immutable, append-only recorder for evolution cycle artefacts."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or _BASE
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, tool_name: str, directive: str) -> str:
        """Start recording a new evolution run.  Returns the run_id."""
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{ts}_{tool_name}"
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        self._write(run_dir, "directive.json", {
            "run_id": run_id,
            "tool_name": tool_name,
            "directive": directive,
            "started_at": datetime.utcnow().isoformat() + "Z",
        })
        return run_id

    def record_research(self, run_id: str, research: Dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        self._write(run_dir, "research.json", research)

    def record_generated_code(self, run_id: str, code: str, attempt: int) -> None:
        run_dir = self._run_dir(run_id)
        filename = f"generated_code_attempt_{attempt}.py"
        with open(os.path.join(run_dir, filename), "w") as f:
            f.write(code)

    def record_test_result(
        self, run_id: str, attempt: int, passed: bool, output: str,
    ) -> None:
        run_dir = self._run_dir(run_id)
        self._write(run_dir, f"test_result_attempt_{attempt}.json", {
            "attempt": attempt,
            "passed": passed,
            "output": output[:4000],
            "recorded_at": datetime.utcnow().isoformat() + "Z",
        })

    def record_manifest(self, run_id: str, manifest: Dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        self._write(run_dir, "manifest.json", manifest)

    def finalize_run(
        self,
        run_id: str,
        status: str,   # quarantined | promoted | dismissed
        summary: str = "",
    ) -> None:
        run_dir = self._run_dir(run_id)
        self._write(run_dir, "result.json", {
            "run_id": run_id,
            "final_status": status,
            "summary": summary,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        })

    # ------------------------------------------------------------------
    # Replay / Query
    # ------------------------------------------------------------------

    def list_runs(self) -> list[str]:
        """Return sorted list of run_ids."""
        if not os.path.isdir(self.base_dir):
            return []
        return sorted(
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        )

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """Load all artefacts for a given run into a dict."""
        run_dir = self._run_dir(run_id)
        data: Dict[str, Any] = {"run_id": run_id, "files": {}}
        for fname in sorted(os.listdir(run_dir)):
            fpath = os.path.join(run_dir, fname)
            if fname.endswith(".json"):
                with open(fpath, "r") as f:
                    data["files"][fname] = json.load(f)
            elif fname.endswith(".py"):
                with open(fpath, "r") as f:
                    data["files"][fname] = f.read()
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> str:
        return os.path.join(self.base_dir, run_id)

    @staticmethod
    def _write(directory: str, filename: str, data: Any) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), "w") as f:
            json.dump(data, f, indent=2, default=str)
