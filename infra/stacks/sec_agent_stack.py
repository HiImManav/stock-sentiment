"""AWS CDK stack for the SEC Filings Agent."""

from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct


class SecAgentStack(Stack):
    """Provisions all AWS resources for the SEC Filings Agent."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------
        # S3 bucket for filing chunk cache
        # ---------------------------------------------------------------
        self.filings_bucket = s3.Bucket(
            self,
            "FilingsCacheBucket",
            bucket_name=f"sec-filings-cache-{cdk.Aws.ACCOUNT_ID}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # ---------------------------------------------------------------
        # Lambda function hosting the FastAPI app via Mangum
        # ---------------------------------------------------------------
        self.api_lambda = lambda_.Function(
            self,
            "ApiLambda",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="mangum_handler.handler",
            code=lambda_.Code.from_asset(
                "../",
                exclude=[
                    "infra/*",
                    ".git/*",
                    ".venv/*",
                    "__pycache__/*",
                    "*.pyc",
                    "tests/*",
                    "*.egg-info/*",
                ],
            ),
            memory_size=512,
            timeout=Duration.seconds(120),
            environment={
                "SEC_FILINGS_BUCKET": self.filings_bucket.bucket_name,
                "SEC_EDGAR_USER_AGENT": "SecAgent admin@example.com",
                "AWS_LWA_INVOKE_MODE": "RESPONSE_STREAM",
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant S3 read/write to Lambda
        self.filings_bucket.grant_read_write(self.api_lambda)

        # Grant Bedrock invoke access
        self.api_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],
            )
        )

        # ---------------------------------------------------------------
        # API Gateway REST API
        # ---------------------------------------------------------------
        self.api = apigw.LambdaRestApi(
            self,
            "SecAgentApi",
            handler=self.api_lambda,
            proxy=True,
            deploy_options=apigw.StageOptions(stage_name="prod"),
        )

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        cdk.CfnOutput(self, "ApiUrl", value=self.api.url)
        cdk.CfnOutput(self, "BucketName", value=self.filings_bucket.bucket_name)
