import json
import sys
from scripts import resource_report


def test_resource_report_does_not_require_local_legacy_run(tmp_path, monkeypatch):
    run = tmp_path / "experiments" / "new_run"
    controls = run / "ablations_seed42"
    controls.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({
        "state": "completed", "seconds": 12.0, "peak_cuda_bytes": 0,
    }), encoding="utf-8")
    (controls / "manifest.json").write_text(json.dumps({
        "state": "completed", "cells": 2,
    }), encoding="utf-8")
    monkeypatch.setattr(resource_report, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["resource_report.py", "--run", "experiments/new_run"])

    resource_report.main()

    result = json.loads((run / "resources.json").read_text(encoding="utf-8"))
    assert result["primary_matrix_wall_seconds"] == 12.0
    assert result["completed_supplementary_cells"] == 2
    assert result["baseline_parent"] is None
    assert result["superseded_before_test_elapsed_seconds"] is None
    assert result["elapsed_since_environment_script_created_hours"] is None
