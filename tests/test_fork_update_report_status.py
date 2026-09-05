"""ReportStatusFn message shaping.

The Lambda handlers under ops/fork_update_agent are deployed as flat bundles, so they
are imported by path here rather than as an installed package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HANDLER = (
    Path(__file__).resolve().parent.parent
    / "ops"
    / "fork_update_agent"
    / "source"
    / "lambdas"
    / "report_status"
    / "handler.py"
)


def _load_build_message():
    """Import the handler without importing boto3 clients at module scope."""
    boto3 = pytest.importorskip("boto3", reason="fork-update handlers need boto3")
    assert boto3 is not None

    spec = importlib.util.spec_from_file_location("fua_report_status", HANDLER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._build_message


def test_success_message_carries_the_version_and_evidence_links() -> None:
    build_message = _load_build_message()
    message = build_message(
        {
            "status": "SUCCESS",
            "detail": {
                "deploy": {"upstream_version": "v0.3.20", "execution_id": "deploy-1"},
                "smoke": {"smoke_execution_arn": "arn:aws:states:::execution/smoke-1"},
                "merge": {"build_id": "build-1"},
            },
        }
    )

    assert message["status"] == "SUCCESS"
    assert message["upstream_version"] == "v0.3.20"
    assert message["smoke_execution"].endswith("smoke-1")
    assert message["deploy_id"] == "deploy-1"


def test_failure_message_names_the_stage_that_broke() -> None:
    build_message = _load_build_message()
    message = build_message(
        {"status": "FAILED", "stage": "deploy", "detail": {"error": "stack rollback"}}
    )

    assert message["stage"] == "deploy"
    assert message["error"] == "stack rollback"


def test_skipped_message_records_why_nothing_happened() -> None:
    build_message = _load_build_message()
    message = build_message(
        {"status": "SKIPPED", "reason": "no new release", "current_version": "v0.3.19"}
    )

    assert message["reason"] == "no new release"
    assert message["current_version"] == "v0.3.19"
