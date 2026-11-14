
from contextlib import asynccontextmanager
from fastapi import FastAPI , Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from qdrant_rag_system.service import AeroRAGService
from backend.api import routes
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Initialize RAG service when application starts
    logger.info("Initializing AeroRAG service...")

    app.state.rag_service = AeroRAGService()

    yield

    # Cleanup when application shuts down
    logger.info("Shutting down AeroRAG API...")


app = FastAPI(
    title="AeroRAG Intelligence API",
    description="Backend API for the MiG-29 Flight Manual RAG application",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AeroRAG Intelligence API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/ready")
def readiness_check(request : Request):

    rag_service = getattr(
        request.app.state,
        "rag_service",
        None
    )

    if rag_service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"}
        )

    result = rag_service.check_readiness()

    if result["status"] != "ready":
        return JSONResponse(
            status_code=503,
            content=result
        )

    return result


# Register the API router
app.include_router(routes.router)
