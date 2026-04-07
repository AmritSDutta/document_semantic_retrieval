import logging
import subprocess
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config.Settings import get_settings
from app.config.logging_config import setup_logging
from app.database.data.batch_insert_resume import batch_insert_async
from app.database.vector_db import db
from app.routers import app_router
from app.schema.exceptions import ProviderUnavailableError

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
app.include_router(app_router.router, prefix="/api")


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


@app.get("/")
async def health():
    logger.info('{"health": "Server in fine health"}')
    return {"health": "Server in fine health"}


if __name__ == " __main__":
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
