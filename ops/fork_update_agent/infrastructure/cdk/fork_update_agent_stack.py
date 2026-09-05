from __future__ import annotations

from pathlib import Path

from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_sns as sns,
    aws_ssm as ssm,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
LAMBDA_SRC = ROOT_DIR / "source" / "lambdas"


class ForkUpdateAgentStack(Stack):
    """Provision the Fork-update Agent workflow and supporting infrastructure."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        schedule_expression = self.node.try_get_context("default_schedule_expression") or "rate(6 hours)"

        github_token_param = ssm.StringParameter(
            self,
            "GitHubTokenParameter",
            parameter_name="/fork-update-agent/github/token",
            string_value="REPLACE_ME",
            description="GitHub token (PAT) used to query upstream releases.",
        )

        current_version_param = ssm.StringParameter(
            self,
            "CurrentVersionParameter",
            parameter_name="/fork-update-agent/state/latest-version",
            string_value="0.0.0",
            description="Tracks last successfully deployed upstream version.",
        )

        notification_topic = sns.Topic(
            self,
            "ForkUpdateNotifications",
            display_name="Fork Update Agent Notifications",
        )

        lambda_common_env = {
            "GITHUB_TOKEN_PARAM": github_token_param.parameter_name,
            "CURRENT_VERSION_PARAM": current_version_param.parameter_name,
            "SNS_TOPIC_ARN": notification_topic.topic_arn,
            "SMOKE_TEST_STEP_FUNCTION": self.node.try_get_context("smoke_test_step_function_arn") or "",
            "SANDBOX_STACK_NAME": self.node.try_get_context("sandbox_root_stack") or "IDP-ACCELERATOR-TEST-2",
            "SMOKE_TEST_BUCKET": self.node.try_get_context("smoke_test_bucket") or "",
            "SMOKE_TEST_KEY": self.node.try_get_context("smoke_test_key") or "fixtures/sample-invoice.pdf",
            "UPSTREAM_OWNER": self.node.try_get_context("upstream_owner") or "aws-solutions-library-samples",
            "UPSTREAM_REPO": self.node.try_get_context("upstream_repo") or "accelerated-intelligent-document-processing-on-aws",
            "FORK_REPO": self.node.try_get_context("fork_repo") or "ricoh/idp_common",
        }

        detect_release_fn = self._create_lambda(
            "DetectReleaseFn",
            code_dir=LAMBDA_SRC / "detect_release",
            environment=lambda_common_env,
        )

        prepare_merge_fn = self._create_lambda(
            "PrepareMergeFn",
            code_dir=LAMBDA_SRC / "prepare_merge",
            environment=lambda_common_env,
            timeout=Duration.minutes(5),
        )

        deploy_fn = self._create_lambda(
            "DeploySandboxFn",
            code_dir=LAMBDA_SRC / "deploy_sandbox",
            environment=lambda_common_env,
            timeout=Duration.minutes(5),
        )

        smoke_test_fn = self._create_lambda(
            "RunSmokeTestFn",
            code_dir=LAMBDA_SRC / "run_smoke_test",
            environment=lambda_common_env,
            timeout=Duration.minutes(5),
        )

        report_fn = self._create_lambda(
            "ReportStatusFn",
            code_dir=LAMBDA_SRC / "report_status",
            environment=lambda_common_env,
        )

        # Grant SSM parameter access
        github_token_param.grant_read(detect_release_fn)
        current_version_param.grant_read(detect_release_fn)
        current_version_param.grant_read(report_fn)
        current_version_param.grant_write(report_fn)
        current_version_param.grant_write(deploy_fn)  # Needs to update version after deployment

        # Grant SNS publish permission
        notification_topic.grant_publish(report_fn)
        notification_topic.grant_publish(prepare_merge_fn)  # Sends notifications about new releases

        smoke_test_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "stepfunctions:StartExecution",
                    "stepfunctions:DescribeExecution",
                    "stepfunctions:GetExecutionHistory",
                ],
                resources=["*"],
            )
        )

        # Grant CloudFormation permissions to deploy_fn for stack updates
        deploy_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudformation:DescribeStacks",
                    "cloudformation:DescribeStackEvents",
                    "cloudformation:UpdateStack",
                    "cloudformation:CreateChangeSet",
                    "cloudformation:ExecuteChangeSet",
                    "cloudformation:DescribeChangeSet",
                ],
                resources=["*"],
            )
        )

        # Step Functions workflow definition
        detect_release_task = tasks.LambdaInvoke(
            self,
            "DetectNewRelease",
            lambda_function=detect_release_fn,
            result_path="$.detect",
            payload_response_only=True,
        )

        merge_task = tasks.LambdaInvoke(
            self,
            "PrepareMergeAndBuild",
            lambda_function=prepare_merge_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "upstream_version.$": "$.detect.upstream_version",
                    "release_url.$": "$.detect.release_url",
                    "release_notes.$": "$.detect.release_notes",
                }
            ),
            result_path="$.merge",
            payload_response_only=True,
        )

        deploy_task = tasks.LambdaInvoke(
            self,
            "DeployToSandbox",
            lambda_function=deploy_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "upstream_version.$": "$.merge.upstream_version",
                    "pr_url.$": "$.merge.pr_url",
                }
            ),
            result_path="$.deploy",
            payload_response_only=True,
        )

        smoke_task = tasks.LambdaInvoke(
            self,
            "RunSmokeTest",
            lambda_function=smoke_test_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "upstream_version.$": "$.deploy.upstream_version",
                    "deployment_status.$": "$.deploy.deployment_status",
                }
            ),
            result_path="$.smoke",
            payload_response_only=True,
        )

        success_notify = tasks.LambdaInvoke(
            self,
            "NotifySuccess",
            lambda_function=report_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "status": "SUCCESS",
                    "detail.$": "$",
                }
            ),
            result_path="$.notification",
            payload_response_only=True,
        )

        skipped_notify = tasks.LambdaInvoke(
            self,
            "NotifyNoop",
            lambda_function=report_fn,
            payload=sfn.TaskInput.from_object(
                {
                    "status": "SKIPPED",
                    "reason": "No new upstream release detected",
                    "current_version.$": "$.detect.current_version",
                }
            ),
            result_path="$.notification",
            payload_response_only=True,
        )

        def failure_chain(stage: str, fail_id: str) -> sfn.Chain:
            notify = tasks.LambdaInvoke(
                self,
                f"NotifyFailure{stage}",
                lambda_function=report_fn,
                payload=sfn.TaskInput.from_object(
                    {
                        "status": "FAILED",
                        "stage": stage,
                        "detail.$": "$",
                    }
                ),
                result_path="$.notification",
                payload_response_only=True,
            )
            return notify.next(sfn.Fail(self, fail_id))

        merge_task.add_catch(failure_chain("MERGE", "MergeFailed"), result_path="$.error")
        deploy_task.add_catch(failure_chain("DEPLOY", "DeployFailed"), result_path="$.error")
        smoke_task.add_catch(failure_chain("SMOKE", "SmokeFailed"), result_path="$.error")

        success_chain = merge_task.next(deploy_task).next(smoke_task).next(success_notify)

        detect_release_task.add_catch(failure_chain("DETECTION", "DetectFailed"), result_path="$.error")

        definition = detect_release_task.next(
            sfn.Choice(self, "AnyUpdate?")
            .when(sfn.Condition.boolean_equals("$.detect.update_required", True), success_chain)
            .otherwise(skipped_notify)
        )

        state_machine = sfn.StateMachine(
            self,
            "ForkUpdateStateMachine",
            definition=definition,
            timeout=Duration.hours(2),
            logs=sfn.LogOptions(
                destination=logs.LogGroup(
                    self,
                    "ForkUpdateLogs",
                    retention=logs.RetentionDays.ONE_MONTH,
                    removal_policy=cdk.RemovalPolicy.DESTROY,
                ),
                level=sfn.LogLevel.ALL,
            ),
        )

        detection_schedule = events.Rule(
            self,
            "ForkUpdateAgentSchedule",
            schedule=events.Schedule.expression(schedule_expression),
            targets=[targets.SfnStateMachine(state_machine)],
        )

        cdk.CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        cdk.CfnOutput(self, "ScheduleName", value=detection_schedule.rule_name)
        cdk.CfnOutput(self, "NotificationTopicArn", value=notification_topic.topic_arn)

    def _create_lambda(
        self,
        logical_id: str,
        *,
        code_dir: Path,
        environment: dict[str, str],
        timeout: Duration | None = None,
    ) -> lambda_.Function:
        return lambda_.Function(
            self,
            logical_id,
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(code_dir)),
            timeout=timeout or Duration.seconds(30),
            environment=environment,
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
