import json
import logging
import os
from typing import Any

import boto3


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

SNS = boto3.client("sns")
SSM = boto3.client("ssm")


def _update_version_param(param_name: str, version: str) -> None:
    if not param_name or not version:
        return
    LOGGER.info("Updating %s to %s", param_name, version)
    SSM.put_parameter(Name=param_name, Value=version, Overwrite=True)


def _build_message(event: dict[str, Any]) -> dict[str, Any]:
    status = event.get("status", "UNKNOWN")
    summary: dict[str, Any] = {"status": status}

    if status == "SUCCESS":
        detail = event.get("detail", {})
        version = (
            detail.get("deploy", {}).get("upstream_version")
            or detail.get("merge", {}).get("upstream_version")
            or detail.get("detect", {}).get("upstream_version")
        )
        summary.update(
            {
                "upstream_version": version,
                "smoke_execution": detail.get("smoke", {}).get("smoke_execution_arn"),
                "build_id": detail.get("merge", {}).get("build_id"),
                "deploy_id": detail.get("deploy", {}).get("execution_id"),
            }
        )
    elif status == "FAILED":
        summary.update(
            {
                "stage": event.get("stage"),
                "error": event.get("detail", {}).get("error"),
            }
        )
    elif status == "SKIPPED":
        summary.update(
            {
                "reason": event.get("reason"),
                "current_version": event.get("current_version"),
            }
        )

    return summary


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    LOGGER.info("Reporting status: %s", event)
    topic_arn = os.environ["SNS_TOPIC_ARN"]
    current_version_param = os.environ["CURRENT_VERSION_PARAM"]

    message = _build_message(event)

    if event.get("status") == "SUCCESS":
        target_version = message.get("upstream_version")
        if target_version:
            _update_version_param(current_version_param, target_version)

    SNS.publish(
        TopicArn=topic_arn,
        Message=json.dumps(message, indent=2),
        Subject=f"Fork Update Agent - {event.get('status', 'UNKNOWN')}",
    )

    return message
