> [!NOTE]
> This is the **original Fork-Update Agent project README, preserved as written**.
> When the project was vendored into this repository, two files moved to the repository
> root: the design document is now
> [`docs/PROCRV_document.pdf`](../../docs/PROCRV_document.pdf) and the runbook is now
> [`docs/FORK_UPDATE_RUNBOOK.md`](../../docs/FORK_UPDATE_RUNBOOK.md). The links below
> have been repointed accordingly; nothing else has been changed.
>
> For the write-up in the context of the whole project, see
> [`docs/FORK_UPDATE_AGENT.md`](../../docs/FORK_UPDATE_AGENT.md).

# Fork-Update Agent

An automated system for detecting upstream releases, notifying teams, deploying updates, and validating the GenAI IDP accelerator in the sandbox environment.

## Overview

The Fork-Update Agent automates the process of keeping a sandbox fork synchronized with upstream releases from the AWS Solutions Library's Intelligent Document Processing (IDP) accelerator. The system provides automated detection, team notifications, deployment orchestration, and smoke testing with comprehensive error handling and logging.

**Current Status:** Phase 1 is fully deployed and operational in AWS.

## Phase 1 vs. Full Automation

### Phase 1 (Currently Deployed)
- **Automated Detection:** Checks for new upstream releases every 6 hours via EventBridge schedule
- **Team Notifications:** Sends SNS alerts when new releases are detected
- **Manual Review Gate:** Team manually syncs the fork on GitHub after reviewing changes
- **Automated Deployment:** After manual sync, automatically deploys to sandbox CloudFormation stack
- **Smoke Testing:** Validates deployments by running IDP Step Functions workflow on fixture documents
- **Status Reporting:** Sends success/failure notifications with detailed execution information

### Phase 2 (Future Enhancement)
- **Automated PR Creation:** Will automatically create GitHub pull requests to sync the fork
- **Service Account Required:** Needs a GitHub service account token (not personal credentials)

**Note:** All code for Phase 2 is already implemented. Only configuration changes are needed to enable full automation.

## Architecture

The solution is built on AWS Step Functions orchestrating five Lambda functions:

```
┌─────────────────────────────────────────────────────────────────┐
│  EventBridge Schedule (every 6 hours)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step Functions State Machine                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  1. DetectReleaseFn - Check GitHub for new releases         │ │
│  │     ├─ If new release found ────┐                           │ │
│  │     └─ If no update ──┐         │                           │ │
│  │                        │        │                           │ │
│  │  2. PrepareMergeFn ◄───┘        │                           │ │
│  │     (Send SNS notification)     │                           │ │
│  │     ▼                           │                           │ │
│  │ 👤 Manual Fork Sync            │                           │ │
│  │     (Team reviews on GitHub)    │                           │ │
│  │     ▼                           │                           │ │
│  │  3. DeploySandboxFn             │                           │ │
│  │     (Update CloudFormation)     │                           │ │
│  │     ▼                           │                           │ │
│  │  4. RunSmokeTestFn              │                           │ │
│  │     (Validate with fixtures)    │                           │ │
│  │     ▼                           │                           │ │
│  │  5. ReportStatusFn ◄────────────┘                           │ │
│  │     (Send final notification)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
.
├── docs/                           # Design documentation and runbooks
│   ├── procrv_digest.md           # Architecture and design decisions
│   └── runbook.md                 # Operational procedures
├── infrastructure/
│   └── cdk/                        # AWS CDK infrastructure as code
│       ├── app.py                 # CDK app entry point
│       └── fork_update_agent_stack.py  # Main stack definition
├── source/
│   └── lambdas/                   # Lambda function handlers
│       ├── detect_release/        # Upstream release detection
│       ├── prepare_merge/         # Notification system
│       ├── deploy_sandbox/        # CloudFormation deployment
│       ├── run_smoke_test/        # Step Functions smoke test
│       └── report_status/         # Status reporting and notifications
├── state_machines/                # Step Functions ASL definitions
│   └── fork_update_agent.asl.json
├── tests/                         # Unit tests
└── README.md                      # This file
```

## Lambda Functions

### 1. DetectReleaseFn
**Purpose:** Detect new upstream releases from GitHub

- Queries GitHub API at `/repos/{owner}/{repo}/releases/latest`
- Falls back to `/tags` endpoint if releases are not available
- Compares upstream version with current deployed version from SSM Parameter Store
- Returns version metadata, release URL, and release notes
- **No authentication required** for public repositories

### 2. PrepareMergeFn
**Purpose:** Notify team about new releases

**Phase 1 (Current):**
- Sends SNS notification with release details
- Team manually syncs fork on GitHub after review
- No GitHub credentials required

**Phase 2 (Future):**
- Automatically creates GitHub pull requests
- Requires GitHub service account token
- Code structure already implemented, needs configuration

### 3. DeploySandboxFn
**Purpose:** Deploy validated updates to sandbox environment

- Updates SSM parameter with new version number
- Checks CloudFormation stack status for safety
- Triggers stack update with new parameters
- Polls for completion (30-minute timeout)
- Provides detailed failure reporting with stack events
- Gracefully handles "no changes" scenarios

### 4. RunSmokeTestFn
**Purpose:** Validate deployments through end-to-end testing

- Invokes existing IDP Accelerator Step Functions workflow
- Uses curated S3 fixture documents for consistent testing
- Polls execution every 15 seconds until completion
- Acts as quality gate to prevent broken deployments
- Returns execution ARN and detailed results

### 5. ReportStatusFn
**Purpose:** Report results and update state

- Publishes success/failure notifications to SNS
- Updates SSM parameter with latest successfully deployed version
- Formats human-readable status messages
- Provides detailed execution information for troubleshooting

## Deployment

### Prerequisites

- AWS CLI configured with credentials for the sandbox account
- Python 3.12 or later
- AWS CDK Toolkit installed: `npm install -g aws-cdk`

### Configuration Parameters

The CDK stack uses context values for configuration. Set these in `infrastructure/cdk/cdk.context.json` or pass via `--context`:

- `account`: AWS account ID (default: from environment)
- `region`: AWS region (default: `us-east-1`)
- `default_schedule_expression`: EventBridge schedule (default: `rate(6 hours)`)
- `upstream_owner`: GitHub owner of upstream repo (default: `aws-solutions-library-samples`)
- `upstream_repo`: Upstream repository name (default: `accelerated-intelligent-document-processing-on-aws`)
- `fork_repo`: Fork repository reference (default: `ricoh/idp_common`)
- `sandbox_root_stack`: CloudFormation stack name (default: `IDP-ACCELERATOR-TEST-2`)
- `smoke_test_step_function_arn`: ARN of IDP Step Functions workflow
- `smoke_test_bucket`: S3 bucket containing fixture documents
- `smoke_test_key`: S3 key for test document (default: `fixtures/sample-invoice.pdf`)

### Post-Deployment Setup

#### 1. Subscribe to SNS Notifications

Get the SNS Topic ARN from stack outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name ForkUpdateAgentStack \
  --query 'Stacks[0].Outputs[?OutputKey==`NotificationTopicArn`].OutputValue' \
  --output text
```

Subscribe your email:

```bash
aws sns subscribe \
  --topic-arn "arn:aws:sns:us-east-1:YOUR_AWS_ACCOUNT_ID:ForkUpdateAgentStack-ForkUpdateNotifications-XXXXX" \
  --protocol email \
  --notification-endpoint your-email@example.com
```

You'll receive a confirmation email. Click the link to confirm the subscription.

#### 2. Configure GitHub Token (Optional)

For Phase 1, no GitHub token is required since the system only reads from public repositories. For Phase 2 automation, update the SSM parameter:

```bash
aws ssm put-parameter \
  --name "/fork-update-agent/github/token" \
  --value "ghp_YourGitHubServiceAccountToken" \
  --type SecureString \
  --overwrite
```

**Security Note:** Use a GitHub **service account** token, not a personal access token.

## Usage

### How to Use (Team Workflow)

#### Step 1: Receive Notification
When a new upstream release is detected, you'll receive an email notification containing:
- Upstream repository and version information
- Direct link to release notes on GitHub
- Instructions for manual fork sync

#### Step 2: Review Changes
- Review the release notes and changes
- Assess impact on your fork
- Decide when to proceed with the merge

#### Step 3: Manually Sync Fork
1. Go to the fork repository on GitHub (link provided in notification)
2. Click the **"Sync fork"** button
3. Review the changes that will be merged
4. Click **"Update branch"** when ready

#### Step 4: Automatic Deployment (Optional)
After manual sync, the agent will automatically:
- Detect the fork has been updated (on next scheduled run, within 6 hours)
- Deploy to the sandbox CloudFormation stack
- Run smoke tests to validate the deployment
- Send success/failure notification

Or trigger immediately (see Manual Trigger below).

### Manual Trigger

To manually trigger the workflow (useful for testing or on-demand validation):

```bash
# Get the State Machine ARN
aws cloudformation describe-stacks \
  --stack-name ForkUpdateAgentStack \
  --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
  --output text

# Start execution
aws stepfunctions start-execution \
  --state-machine-arn "arn:aws:states:us-east-1:YOUR_AWS_ACCOUNT_ID:stateMachine:ForkUpdateStateMachine052A8A50-XXXXX" \
  --name "manual-test-$(date +%s)" \
  --input '{}'
```

### Monitor Executions

View execution history in the AWS Console:
1. Navigate to Step Functions → State machines
2. Select **ForkUpdateStateMachine**
3. View execution history and logs

Or use the CLI:

```bash
aws stepfunctions list-executions \
  --state-machine-arn "arn:aws:states:us-east-1:YOUR_AWS_ACCOUNT_ID:stateMachine:ForkUpdateStateMachine052A8A50-XXXXX" \
  --max-results 10
```

## Monitoring and Troubleshooting

### CloudWatch Logs

All Lambda functions log to CloudWatch with 30-day retention:
- `/aws/lambda/ForkUpdateAgentStack-DetectReleaseFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-PrepareMergeFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-DeploySandboxFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-RunSmokeTestFn-XXXXX`
- `/aws/lambda/ForkUpdateAgentStack-ReportStatusFn-XXXXX`

Step Functions execution logs:
- `/aws/vendedlogs/states/ForkUpdateLogs`

### Disable Automation

To temporarily disable scheduled executions:

```bash
# Get the schedule rule name
aws events list-rules --name-prefix ForkUpdateAgentSchedule

# Disable the rule
aws events disable-rule --name ForkUpdateAgentStack-ForkUpdateAgentSchedule-XXXXX
```

To re-enable:

```bash
aws events enable-rule --name ForkUpdateAgentStack-ForkUpdateAgentSchedule-XXXXX
```

### Technology Stack

- **Runtime:** Python 3.12
- **AWS Services:** Lambda, Step Functions, EventBridge, SNS, SSM Parameter Store, CloudFormation, CloudWatch
- **Infrastructure:** AWS CDK (Python)
- **Dependencies:** boto3 (AWS SDK), standard library only for Lambda functions

### Code Style

Lambda functions intentionally use only `boto3` and Python standard library to avoid dependency management complexity. All infrastructure is defined as code using AWS CDK for reproducible deployments.

## Security

- **Least Privilege IAM:** Each Lambda function has minimal required permissions
- **No Personal Credentials:** Phase 1 requires no authentication; Phase 2 uses service accounts
- **Secrets Management:** GitHub tokens stored in SSM Parameter Store with encryption
- **Audit Trail:** Complete CloudWatch logging with 30-day retention
- **Sandbox Only:** System only operates in non-production sandbox environment

## Roadmap

### Phase 2: Full Automation
- Configure GitHub service account token
- Enable automated PR creation in PrepareMergeFn
- Remove manual fork synchronization step
- **Estimated Effort:** Configuration only, no code changes required

### Future Enhancements
- Multi-branch support for different environments
- Enhanced smoke tests with quality validation
- Integration with Slack for real-time notifications
- Automated rollback on smoke test failures
- Webhook-based instant detection (replacing polling)

## Documentation

- **[PROCRV Document](../../docs/PROCRV_document.pdf):** Architecture, design decisions, and constraints
- **[Operational Runbook](../../docs/FORK_UPDATE_RUNBOOK.md):** Deployment procedures and troubleshooting
- **This README:** Quick start and usage guide

## License

This project was developed as part of an internship project for Ricoh USA, Inc.

---
