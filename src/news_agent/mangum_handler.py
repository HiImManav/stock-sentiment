"""AWS Lambda handler using Mangum for FastAPI."""

from mangum import Mangum

from news_agent.api.server import app

handler = Mangum(app, lifespan="off")
