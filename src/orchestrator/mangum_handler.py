"""AWS Lambda handler using Mangum for FastAPI."""

from mangum import Mangum

from orchestrator.api.server import app

handler = Mangum(app, lifespan="off")
