> [!IMPORTANT]
> **This runbook contains environment-specific identifiers** — AWS account IDs, stack
> names, and state-machine ARNs from the sandbox this system ran in. They are retained
> because this repository is private and the runbook is only useful with them in place.
> **Redact them before making this repository public or sharing it outside the team.**
> Every occurrence is an account ID, a stack name, or an ARN; there are no credentials.

# Fork-Update Agent Operational Runbook

This runbook provides comprehensive operational procedures for deploying, configuring, monitoring, and troubleshooting the Fork-Update Agent.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Deployment](#initial-deployment)
- [Configuration](#configuration)
- [Day-2 Operations](#day-2-operations)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)
- [Extensibility](#extensibility)

---

## Prerequisites

### AWS Environment
- AWS account credentials with appropriate permissions (sandbox account ending in 899)
- AWS region: `us-east-1`
- AWS CLI installed and configured
- AWS CDK CLI installed: `npm install -g aws-cdk`

### Required AWS Permissions
The deployment user/role needs permissions for:
- **Lambda**: Create and manage Lambda functions
- **Step Functions**: Create and manage state machines
- **EventBridge**: Create and manage scheduled rules
- **IAM**: Create roles and policies with least-privilege permissions
- **SNS**: Create topics and manage subscriptions
- **SSM Parameter Store**: Create and manage parameters
- **CloudWatch Logs**: Create log groups and set retention
- **CloudFormation**: Deploy CDK stacks

### Software Requirements
- Python 3.12 or later
- pip (Python package manager)
- Git (for repository management)

### Optional Resources
- GitHub service account (for Phase 2 automation)
- S3 bucket with fixture documents for smoke testing
- Existing IDP Accelerator Step Functions workflow (for smoke tests)

---

## Initial Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/mingketian/Fork-Update-Agent-.git
cd Fork-Update-Agent-
```

### Step 2: Install CDK Dependencies

```bash
cd infrastructure/cdk
pip install -r requirements.txt
```

### Step 3: Configure CDK Context

Create or edit `cdk.context.json` in the `infrastructure/cdk/` directory:

```json
{
  "account": "922653976899",
  "region": "us-east-1",
  "default_schedule_expression": "rate(6 hours)",
  "upstream_owner": "aws-solutions-library-samples",
  "upstream_repo": "accelerated-intelligent-document-processing-on-aws",
  "fork_repo": "ricoh/idp_common",
  "sandbox_root_stack": "IDP-ACCELERATOR-TEST-2",
  "smoke_test_step_function_arn": "arn:aws:states:us-east-1:922653976899:stateMachine:YourIDPWorkflow",
  "smoke_test_bucket": "your-test-bucket",
  "smoke_test_key": "fixtures/sample-invoice.pdf"
}
```

**Note:** Replace placeholder values with your actual configuration.

### Step 4: Bootstrap CDK (First Time Only)

```bash
export CDK_DEFAULT_ACCOUNT=922653976899
export CDK_DEFAULT_REGION=us-east-1

cdk bootstrap aws://922653976899/us-east-1
```

### Step 5: Synthesize CloudFormation Template

```bash
cdk synth
```

Review the generated CloudFormation template in `cdk.out/ForkUpdateAgentStack.template.json`.

### Step 6: Deploy Stack

```bash
cdk deploy ForkUpdateAgentStack
```

Review the changes and approve when prompted. The deployment creates:
- 5 Lambda functions with IAM roles
- 1 Step Functions state machine
- 1 EventBridge scheduled rule (every 6 hours)
- 1 SNS topic for notifications
- 2 SSM parameters (GitHub token placeholder, version tracking)
- CloudWatch log groups with 30-day retention

### Step 7: Capture Stack Outputs

After deployment, save the stack outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name ForkUpdateAgentStack \
  --query 'Stacks[0].Outputs' \
  --output table
```

Key outputs:
- `StateMachineArn`: ARN for manual trigger commands
- `NotificationTopicArn`: ARN for SNS subscriptions
- `ScheduleName`: EventBridge rule name

---

## Configuration

### SNS Notification Setup

#### Subscribe Email Address

```bash
# Get the SNS Topic ARN from stack outputs
TOPIC_ARN=$(aws cloudformation describe-stacks \
  --stack-name ForkUpdateAgentStack \
  --query 'Stacks[0].Outputs[?OutputKey==`NotificationTopicArn`].OutputValue' \
  --output text)

# Subscribe your email
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint your-email@example.com
```

You'll receive a confirmation email. Click the confirmation link to activate the subscription.

#### Subscribe Multiple Team Members

```bash
# Subscribe additional emails
aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint teammate1@example.com
aws sns subscribe --topic-arn "$TOPIC_ARN" --protocol email --notification-endpoint teammate2@example.com
```

#### Optional: Slack Integration

For Slack notifications, create a Slack incoming webhook and subscribe it:

```bash
aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol https \
  --notification-endpoint https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### GitHub Token Configuration

#### Phase 1 (Current - No Token Required)
Phase 1 reads from public repositories and requires no authentication. The default SSM parameter value is `REPLACE_ME` and is ignored.

#### Phase 2 (Future - Service Account Required)

When ready for Phase 2 automation:

1. Create a GitHub service account (not tied to any individual)
2. Generate a Personal Access Token with `repo` scope
3. Update the SSM parameter:

```bash
aws ssm put-parameter \
  --name "/fork-update-agent/github/token" \
  --value "ghp_YourServiceAccountToken" \
  --type SecureString \
  --overwrite
```

**Security Best Practices:**
- Use a dedicated service account, not personal credentials
- Store token in SSM Parameter Store with encryption
- Rotate token periodically
- Restrict token scope to minimum required permissions

### Initial Version Setup

Set the initial version to track (if not using default `0.0.0`):

```bash
aws ssm put-parameter \
  --name "/fork-update-agent/state/latest-version" \
  --value "1.0.0" \
  --type String \
  --overwrite
```

---

## Day-2 Operations

### Manual Trigger Workflow

To manually trigger the workflow (useful for testing or on-demand validation):

```bash
# Get State Machine ARN
STATE_MACHINE_ARN=$(aws cloudformation describe-stacks \
  --stack-name ForkUpdateAgentStack \
  --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
  --output text)

# Start execution
aws stepfunctions start-execution \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --name "manual-test-$(date +%s)" \
  --input '{}'
```

### Check Execution Status

```bash
# List recent executions
aws stepfunctions list-executions \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --max-results 10

# Describe specific execution
aws stepfunctions describe-execution \
  --execution-arn "arn:aws:states:us-east-1:922653976899:execution:ForkUpdateStateMachine:execution-name"
```

### Team Workflow (Phase 1)

#### 1. Receive Notification
Team members receive email notification when new upstream release detected.

#### 2. Review Release Notes
- Click release URL in notification
- Review changes, breaking changes, and impact
- Assess compatibility with fork customizations

#### 3. Manual Fork Sync
- Navigate to fork repository on GitHub
- Click **"Sync fork"** button
- Review changes in diff view
- Click **"Update branch"** to merge

#### 4. Trigger Deployment (Optional)
Either wait for next scheduled run (within 6 hours) or manually trigger:

```bash
aws stepfunctions start-execution \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --name "post-merge-$(date +%s)" \
  --input '{}'
```

### Modify Schedule

To change the detection frequency:

```bash
# Update CDK context
# Edit infrastructure/cdk/cdk.context.json:
# "default_schedule_expression": "rate(12 hours)"

# Redeploy
cd infrastructure/cdk
cdk deploy ForkUpdateAgentStack
```

Common schedule expressions:
- `rate(6 hours)` - Every 6 hours
- `rate(12 hours)` - Every 12 hours
- `rate(1 day)` - Daily
- `cron(0 9 * * ? *)` - Daily at 9 AM UTC

### Disable Scheduled Execution

To temporarily stop automatic executions:

```bash
# Get rule name
RULE_NAME=$(aws events list-rules \
  --name-prefix ForkUpdateAgentSchedule \
  --query 'Rules[0].Name' \
  --output text)

# Disable the rule
aws events disable-rule --name "$RULE_NAME"
```

### Re-enable Scheduled Execution

```bash
aws events enable-rule --name "$RULE_NAME"
```

### Force Rollback

If a deployment causes issues, rollback to previous version:

```bash
# Update version parameter to previous version
aws ssm put-parameter \
  --name "/fork-update-agent/state/latest-version" \
  --value "1.0.0" \
  --type String \
  --overwrite

# Manually trigger to redeploy previous version
aws stepfunctions start-execution \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --name "rollback-$(date +%s)" \
  --input '{}'
```

---

## Monitoring

### CloudWatch Dashboards

Create a custom CloudWatch dashboard for monitoring:

```bash
aws cloudwatch put-dashboard \
  --dashboard-name ForkUpdateAgentMonitoring \
  --dashboard-body file://dashboard-config.json
```

**dashboard-config.json** example:
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/States", "ExecutionsSucceeded", {"stat": "Sum"}],
          [".", "ExecutionsFailed", {"stat": "Sum"}]
        ],
        "period": 3600,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "Step Functions Executions"
      }
    }
  ]
}
```

### CloudWatch Log Groups

Lambda function logs (30-day retention):
- `/aws/lambda/ForkUpdateAgentStack-DetectReleaseFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-PrepareMergeFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-DeploySandboxFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-RunSmokeTestFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-ReportStatusFn-XXXXX`

Step Functions execution logs:
- `/aws/vendedlogs/states/ForkUpdateLogs`

### View Recent Logs

```bash
# View DetectReleaseFn logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-DetectReleaseFn-XXXXX --follow

# Filter for errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/ForkUpdateAgentStack-DetectReleaseFn-XXXXX \
  --filter-pattern "ERROR"
```

### CloudWatch Insights Queries

```sql
-- Count executions by status
fields @timestamp, status
| stats count() by status

-- Find failed executions
fields @timestamp, @message
| filter @message like /FAILED/
| sort @timestamp desc

-- Average execution duration
fields @duration
| stats avg(@duration) as avg_duration, max(@duration) as max_duration
```

### Metrics to Monitor

- **Step Functions Execution Success Rate**: Should be >95%
- **Lambda Function Errors**: Should be minimal
- **Lambda Duration**: Should be well under timeout limits
- **SNS Notification Delivery**: Should be 100%

---

## Troubleshooting

### Detection Failure

**Symptoms:**
- Execution fails at DetectReleaseFn step
- Error: "Unable to connect to GitHub API"
- Error: "Unable to determine latest version"

**Diagnosis:**
```bash
# Check DetectReleaseFn logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-DetectReleaseFn-XXXXX --since 1h

# Check GitHub API accessibility
curl https://api.github.com/repos/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws/releases/latest
```

**Resolution:**
1. Verify repository names in environment variables
2. Check network connectivity to GitHub API
3. Verify SSM parameter exists (even with placeholder value)
4. Check IAM permissions for SSM parameter read

### Merge/Notification Failure

**Symptoms:**
- Execution fails at PrepareMergeFn step
- SNS notifications not received
- Error: "Failed to publish SNS notification"

**Diagnosis:**
```bash
# Check PrepareMergeFn logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-PrepareMergeFn-XXXXX --since 1h

# Verify SNS topic exists
aws sns get-topic-attributes --topic-arn "$TOPIC_ARN"

# Check subscriptions
aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN"
```

**Resolution:**
1. Verify SNS topic ARN in environment variables
2. Check IAM permissions for SNS publish
3. Confirm email subscriptions are confirmed (check spam folder)
4. Verify SNS topic exists and is in correct region

### Deployment Failure

**Symptoms:**
- Execution fails at DeploySandboxFn step
- Error: "Stack is not in a stable state"
- Error: "CloudFormation stack not found"

**Diagnosis:**
```bash
# Check DeploySandboxFn logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-DeploySandboxFn-XXXXX --since 1h

# Check CloudFormation stack status
aws cloudformation describe-stacks --stack-name IDP-ACCELERATOR-TEST-2

# Check recent stack events
aws cloudformation describe-stack-events \
  --stack-name IDP-ACCELERATOR-TEST-2 \
  --max-items 20
```

**Resolution:**
1. Verify sandbox stack name matches environment variable
2. Check stack is in stable state (CREATE_COMPLETE, UPDATE_COMPLETE)
3. Review CloudFormation stack events for error details
4. Check IAM permissions for CloudFormation operations
5. Wait for in-progress updates to complete

### Smoke Test Failure

**Symptoms:**
- Execution fails at RunSmokeTestFn step
- Error: "Smoke test execution failed"
- Error: "Step Functions workflow not found"

**Diagnosis:**
```bash
# Check RunSmokeTestFn logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-RunSmokeTestFn-XXXXX --since 1h

# Verify Step Functions workflow exists
aws stepfunctions describe-state-machine \
  --state-machine-arn "arn:aws:states:us-east-1:922653976899:stateMachine:YourIDPWorkflow"

# Check fixture document exists
aws s3 ls s3://your-test-bucket/fixtures/sample-invoice.pdf
```

**Resolution:**
1. Verify smoke test Step Functions ARN is correct
2. Check fixture document exists in S3
3. Verify IAM permissions for Step Functions execution
4. Review IDP workflow CloudWatch logs for actual failure cause
5. Test IDP workflow independently to isolate issues

### Notification Failure

**Symptoms:**
- Execution succeeds but no final notification received
- Error in ReportStatusFn logs

**Diagnosis:**
```bash
# Check ReportStatusFn logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-ReportStatusFn-XXXXX --since 1h

# Verify SNS deliveries
aws sns list-subscriptions-by-topic --topic-arn "$TOPIC_ARN"
```

**Resolution:**
1. Confirm email subscriptions are confirmed
2. Check spam/junk folders
3. Verify SNS topic permissions
4. Check IAM permissions for SNS publish and SSM parameter write

### Permission Errors

**Symptoms:**
- Error: "AccessDenied" in Lambda logs
- Error: "User is not authorized to perform"

**Diagnosis:**
```bash
# Check Lambda function role
aws lambda get-function --function-name ForkUpdateAgentStack-DetectReleaseFn-XXXXX \
  --query 'Configuration.Role'

# Check IAM role policies
aws iam list-attached-role-policies --role-name ForkUpdateAgentStack-DetectReleaseFnRole-XXXXX
```

**Resolution:**
1. Verify Lambda execution roles have required permissions
2. Check IAM policies are correctly attached
3. Review CDK stack for policy definitions
4. Redeploy stack if policies are incorrect: `cdk deploy ForkUpdateAgentStack`

---

## Maintenance

### Update Lambda Functions

To update Lambda function code:

```bash
cd infrastructure/cdk
cdk deploy ForkUpdateAgentStack
```

CDK will detect code changes and update Lambda functions automatically.

### Update State Machine Workflow

To modify the Step Functions workflow:

1. Edit `infrastructure/cdk/fork_update_agent_stack.py`
2. Update task definitions or add new steps
3. Deploy changes:

```bash
cd infrastructure/cdk
cdk deploy ForkUpdateAgentStack
```

### Rotate GitHub Token

```bash
# Generate new token in GitHub
# Update SSM parameter
aws ssm put-parameter \
  --name "/fork-update-agent/github/token" \
  --value "ghp_NewServiceAccountToken" \
  --type SecureString \
  --overwrite
```

### Update CloudWatch Log Retention

To change log retention period:

1. Edit `infrastructure/cdk/fork_update_agent_stack.py`
2. Modify `log_retention` parameter (e.g., `logs.RetentionDays.THREE_MONTHS`)
3. Redeploy stack

### Clean Up Old Executions

Step Functions execution history is retained indefinitely. To reduce costs:

```bash
# List old executions
aws stepfunctions list-executions \
  --state-machine-arn "$STATE_MACHINE_ARN" \
  --status-filter SUCCEEDED \
  --max-results 1000

# Note: AWS automatically manages execution history retention
# No manual cleanup required for executions older than 90 days
```

---

## Extensibility

### Add New Lambda Function

1. Create new Lambda handler in `source/lambdas/new_function/`
2. Add Lambda definition in CDK stack
3. Add task to Step Functions workflow
4. Deploy changes

### Multi-Branch Support

To monitor multiple branches:

1. Add branch parameter to execution input:
   ```json
   {"branch": "develop"}
   ```

2. Update Lambda functions to use branch parameter
3. Create separate SSM parameters per branch
4. Parameterize workflow based on branch

### Custom Notifications

To add Slack/Teams integration:

1. Create webhook in Slack/Teams
2. Update ReportStatusFn to format messages for platform
3. Add webhook URL to environment variables
4. Call webhook in addition to SNS

### Automated Rollback

To enable automatic rollback on smoke test failure:

1. Add rollback task to Step Functions workflow
2. Catch smoke test failure
3. Update version parameter to previous version
4. Trigger redeployment

Example workflow modification:
```python
smoke_task.add_catch(
    rollback_task,
    errors=["States.TaskFailed"],
    result_path="$.error"
)
```

### Webhook-Based Detection

To replace polling with instant webhook detection:

1. Create API Gateway endpoint
2. Configure GitHub webhook to call endpoint
3. Trigger Step Functions from API Gateway
4. Disable EventBridge schedule

---

## Support and Escalation

### Documentation References
- **Architecture**: [PROCRV Digest](PROCRV_document.pdf)
- **Quick Start**: [README.md](../README.md)
- **GitHub Repository**: https://github.com/mingketian/Fork-Update-Agent-

### Common Commands Reference

```bash
# View stack outputs
aws cloudformation describe-stacks --stack-name ForkUpdateAgentStack --query 'Stacks[0].Outputs'

# Manual trigger
aws stepfunctions start-execution --state-machine-arn "$STATE_MACHINE_ARN" --name "manual-$(date +%s)" --input '{}'

# Check execution status
aws stepfunctions list-executions --state-machine-arn "$STATE_MACHINE_ARN" --max-results 5

# View logs
aws logs tail /aws/lambda/ForkUpdateAgentStack-DetectReleaseFn-XXXXX --follow

# Disable schedule
aws events disable-rule --name ForkUpdateAgentStack-ForkUpdateAgentSchedule-XXXXX

# Update version
aws ssm put-parameter --name "/fork-update-agent/state/latest-version" --value "1.2.0" --overwrite
```

---

**Last Updated:** December 2024
**Version:** 1.0 (Phase 1 Deployment)
