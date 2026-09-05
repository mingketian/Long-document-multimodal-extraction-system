#!/usr/bin/env python3
import aws_cdk as cdk

from fork_update_agent_stack import ForkUpdateAgentStack


app = cdk.App()

ForkUpdateAgentStack(
    app,
    "ForkUpdateAgentStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region"),
    ),
)

app.synth()
