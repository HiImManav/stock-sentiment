"""AWS Lambda handler wrapping the FastAPI app via Mangum."""

from mangum import Mangum

from sec_agent.api.server import app

handler = Mangum(app, lifespan="off")
