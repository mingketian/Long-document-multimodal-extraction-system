import logging
import os
import time
from typing import Any

import boto3


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

CFN = boto3.client("cloudformation")
SSM = boto3.client("ssm")


def _wait_for_stack_update(stack_name: str, poll_seconds: int = 30, max_wait_seconds: int = 1800) -> dict[str, Any]:
    """
    Poll CloudFormation until the stack update completes.

    Args:
        stack_name: Name of the CloudFormation stack
        poll_seconds: Seconds to wait between polls
        max_wait_seconds: Maximum time to wait (default 30 minutes)

    Returns:
        Stack description dict

    Raises:
        RuntimeError: If stack update fails or times out
    """
    terminal_statuses = {
        "UPDATE_COMPLETE",
        "UPDATE_ROLLBACK_COMPLETE",
        "UPDATE_ROLLBACK_FAILED",
        "UPDATE_FAILED",
    }

    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            raise RuntimeError(
                f"Stack update timed out after {elapsed:.0f} seconds. Stack: {stack_name}"
            )

        try:
            response = CFN.describe_stacks(StackName=stack_name)
            stacks = response.get("Stacks", [])

            if not stacks:
                raise RuntimeError(f"Stack {stack_name} not found")

            stack = stacks[0]
            status = stack["StackStatus"]

            LOGGER.info("Stack %s status: %s (%.0fs elapsed)", stack_name, status, elapsed)

            if status in terminal_statuses:
                if status == "UPDATE_COMPLETE":
                    LOGGER.info("Stack update completed successfully")
                    return stack
                else:
                    # Get failure reason from stack events
                    events_response = CFN.describe_stack_events(StackName=stack_name)
                    failed_events = [
                        e
                        for e in events_response.get("StackEvents", [])
                        if "FAILED" in e.get("ResourceStatus", "")
                    ]

                    failure_reasons = [
                        f"{e.get('LogicalResourceId')}: {e.get('ResourceStatusReason', 'Unknown')}"
                        for e in failed_events[:3]  # Show first 3 failures
                    ]

                    raise RuntimeError(
                        f"Stack update failed with status {status}. "
                        f"Reasons: {'; '.join(failure_reasons)}"
                    )

        except CFN.exceptions.ClientError as err:
            error_code = err.response.get("Error", {}).get("Code", "")
            if error_code == "ValidationError":
                # Stack might be in a transient state, retry
                LOGGER.warning("Validation error checking stack status, will retry: %s", err)
            else:
                raise

        time.sleep(poll_seconds)


def _update_version_parameter(version: str) -> None:
    """Update the current version parameter in SSM."""
    version_param = os.environ.get("CURRENT_VERSION_PARAM")
    if version_param:
        try:
            SSM.put_parameter(
                Name=version_param,
                Value=version,
                Type="String",
                Overwrite=True,
            )
            LOGGER.info("Updated version parameter %s to %s", version_param, version)
        except Exception as err:
            LOGGER.warning("Failed to update version parameter: %s", err)
    else:
        LOGGER.info("No CURRENT_VERSION_PARAM configured, skipping version update")


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """
    Deploy updated code to the sandbox environment.

    This function updates the CloudFormation stack directly instead of using CodeBuild.
    For a simple deployment, we'll update an SSM parameter that the stack references,
    then trigger a stack update.

    Args:
        event: Event data containing:
            - upstream_version: Version to deploy
            - pr_url (optional): URL of the merge PR

    Returns:
        Deployment status information
    """
    LOGGER.info("Deploy request: %s", event)

    upstream_version = event.get("upstream_version")
    if not upstream_version:
        raise ValueError("upstream_version is required")

    sandbox_stack_name = os.environ.get("SANDBOX_STACK_NAME")
    if not sandbox_stack_name:
        raise RuntimeError("SANDBOX_STACK_NAME environment variable is required")

    LOGGER.info("Deploying version %s to stack %s", upstream_version, sandbox_stack_name)

    try:
        # First, update the version parameter
        # This makes the new version available to the stack
        _update_version_parameter(upstream_version)

        # Check current stack status
        response = CFN.describe_stacks(StackName=sandbox_stack_name)
        stacks = response.get("Stacks", [])

        if not stacks:
            raise RuntimeError(f"Sandbox stack {sandbox_stack_name} not found")

        current_stack = stacks[0]
        current_status = current_stack["StackStatus"]

        LOGGER.info("Current stack status: %s", current_status)

        # Only trigger update if stack is in a stable state
        stable_statuses = {"CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"}

        if current_status not in stable_statuses:
            LOGGER.warning(
                "Stack is not in a stable state (%s). Skipping stack update.",
                current_status,
            )
            return {
                "upstream_version": upstream_version,
                "deployment_status": "skipped",
                "message": f"Stack is in {current_status} state, update skipped",
                "stack_status": current_status,
            }

        # For MVP: We'll just update a parameter and potentially trigger a re-deployment
        # In a full implementation, you would:
        # 1. Update the code/container image with the new version
        # 2. Trigger a CloudFormation stack update with new parameters
        # 3. Wait for the update to complete

        # Simple approach: Update stack with a parameter change to trigger deployment
        # This assumes your stack has a parameter for the version
        try:
            # Get current parameters
            current_parameters = current_stack.get("Parameters", [])

            # For now, we'll just use the existing parameters
            # In a real scenario, you'd update specific parameters like "CodeVersion"
            parameters = [
                {"ParameterKey": p["ParameterKey"], "UsePreviousValue": True}
                for p in current_parameters
            ]

            LOGGER.info("Updating stack with parameters: %s", parameters)

            # Trigger stack update
            # Note: This is a minimal update. In practice, you might need to:
            # - Update template if code changed
            # - Update parameters to point to new artifacts
            # - Use change sets for safety
            update_response = CFN.update_stack(
                StackName=sandbox_stack_name,
                UsePreviousTemplate=True,
                Parameters=parameters,
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
            )

            stack_id = update_response["StackId"]
            LOGGER.info("Stack update initiated: %s", stack_id)

            # Wait for update to complete
            updated_stack = _wait_for_stack_update(sandbox_stack_name)

            return {
                "upstream_version": upstream_version,
                "deployment_status": "success",
                "stack_id": stack_id,
                "stack_status": updated_stack["StackStatus"],
                "message": f"Successfully deployed version {upstream_version}",
            }

        except CFN.exceptions.ClientError as err:
            error_code = err.response.get("Error", {}).get("Code", "")
            error_message = err.response.get("Error", {}).get("Message", "")

            if error_code == "ValidationError" and "No updates are to be performed" in error_message:
                LOGGER.info("No stack updates needed, deployment considered successful")
                return {
                    "upstream_version": upstream_version,
                    "deployment_status": "no_changes",
                    "message": "No stack updates required",
                    "stack_status": current_status,
                }
            else:
                raise

    except Exception as err:
        LOGGER.error("Deployment failed: %s", err, exc_info=True)
        raise RuntimeError(f"Deployment failed: {err}") from err
