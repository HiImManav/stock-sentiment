#!/usr/bin/env python3
"""CDK app entry point for the SEC Filings Agent infrastructure."""

import aws_cdk as cdk

from stacks.sec_agent_stack import SecAgentStack

app = cdk.App()
SecAgentStack(app, "SecAgentStack")
app.synth()
