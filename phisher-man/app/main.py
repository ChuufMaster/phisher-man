from fastapi import FastAPI, Request
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from typing import Callable
from fastapi.responses import JSONResponse
from app.api.v1 import endpoints
from app.routes.predict_raw import router as raw_router
from app.routes.predict_structured import router as structured_router
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="phisher-man.log",
)
logger = logging.getLogger(__name__)

apm = make_apm_client(
    {
        "SERVICE_NAME": "phisher-man",
        "SECRET_TOKEN": "changeme",
        "SERVER_URL": "http://localhost:8200",
        "LOG_LEVEL": "trace",
        "LOG_FILE": "./phisher-man.log",
    }
)

app = FastAPI(title="Phishing Detection API")

app.add_middleware(ElasticAPM, client=apm)


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    log_dict = {
        "url": str(request.url),
        "method": request.method,
        # "client_ip": request.client.host,
        "duration": f"{duration:.2f}s",
        "status_code": response.status_code,
    }

    logger.info(f"Request processed: {log_dict}")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(endpoints.router, prefix="/api/v1")
app.include_router(raw_router, prefix="/api")
app.include_router(structured_router, prefix="/api")
