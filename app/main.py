import logging
import subprocess
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi_limiter.depends import RateLimiter
from starlette.middleware.cors import CORSMiddleware

from app.config.Settings import get_settings
from app.config.logging_config import setup_logging
from app.database.data.batch_insert_resume import batch_insert_async
from app.database.vector_db import db
from app.routers import app_router
from app.schema.exceptions import ProviderUnavailableError
from pyrate_limiter import Duration, Limiter, Rate
# Initialize global logging before other imports
setup_logging()
logger = logging.getLogger(__name__)
app_name = get_settings().APP_NAME
port = get_settings().PORT


def _verify_tests_pass():
    subprocess.run(["pytest", "-q"], check=True)


@asynccontextmanager
async def lifespan(app_ins: FastAPI):
    logging.info(f'start: {app_ins.__str__()}')
    #_verify_tests_pass()
    await db.init()
    await batch_insert_async()
    try:
        yield
    finally:
        await db.close()
        logging.info('finish')


app = FastAPI(title=app_name, lifespan=lifespan)

# Define the security scheme for Swagger
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
app.include_router(app_router.router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Your Streamlit UI URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ProviderUnavailableError)
async def provider_exception_handler(request: Request, exc: ProviderUnavailableError):
    logger.warning(f"Upstream provider failure: {exc.provider} - {exc.message}")
    return JSONResponse(
        status_code=502,
        content={
            "error": "Upstream Provider Failure",
            "provider": exc.provider,
            "message": exc.message,
            "suggestion": "The service is temporarily degraded. Please retry in a few minutes."
        },
    )


@app.get("/", dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(2, Duration.SECOND * 1))))])
async def health():
    logger.info('{"health": "Server in fine health"}')
    return {"health": "Server in fine health"}


if __name__ == " __main__":
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
