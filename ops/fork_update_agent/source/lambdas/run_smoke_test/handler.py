import json
import logging
import os
import time
from typing import Any

import boto3


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

SFN = boto3.client("stepfunctions")


def _wait_for_execution(arn: str, poll_seconds: int = 15) -> dict[str, Any]:
    terminal = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}
    while True:
        execution = SFN.describe_execution(executionArn=arn)
        status = execution["status"]
        LOGGER.info("Smoke test execution %s status: %s", arn, status)
        if status in terminal:
            return execution
        time.sleep(poll_seconds)


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    LOGGER.info("Smoke test request: %s", event)
    state_machine_arn = os.environ["SMOKE_TEST_STEP_FUNCTION"]
    if not state_machine_arn:
        raise RuntimeError("SMOKE_TEST_STEP_FUNCTION environment variable is required.")

    payload = {
        "document_bucket": os.environ.get("SMOKE_TEST_BUCKET"),
        "document_key": os.environ.get("SMOKE_TEST_KEY"),
        "upstream_version": event.get("upstream_version"),
        "trigger": "fork-update-agent",
    }

    execution = SFN.start_execution(stateMachineArn=state_machine_arn, input=json.dumps(payload))
    exec_arn = execution["executionArn"]
    execution_result = _wait_for_execution(exec_arn)

    status = execution_result["status"]
    if status != "SUCCEEDED":
        error = RuntimeError(f"Smoke test failed with status {status}")
        error.execution = execution_result  # type: ignore[attr-defined]
        raise error

    output = execution_result.get("output")

    try:
        decoded_output = json.loads(output) if output else {}
    except json.JSONDecodeError:
        decoded_output = {"raw": output}

    return {
        "smoke_execution_arn": exec_arn,
        "status": status,
        "output": decoded_output,
        "upstream_version": event.get("upstream_version"),
    }
