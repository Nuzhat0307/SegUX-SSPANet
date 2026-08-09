"""
FastAPI application factory and main entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.api.routes import router
from app.services.inference import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""

    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION}"
    )
    logger.info(
        f"Model version: {settings.MODEL_VERSION}"
    )

    # Load the trained SegUX-SSPANet checkpoint
    logger.info(
        f"Loading trained model from: "
        f"{settings.MODEL_CHECKPOINT_PATH}"
    )

    inference_service.load_models()

    if inference_service.is_loaded():
        logger.info(
            "TRAINED MODEL LOADED SUCCESSFULLY"
        )
        logger.info(
            f"Checkpoint: {settings.MODEL_CHECKPOINT_PATH}"
        )
    else:
        logger.error(
            "TRAINED MODEL WAS NOT LOADED. "
            "Inference may fall back to mock inference."
        )

    logger.info("Application startup complete.")

    yield

    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-powered brain tumor diagnosis system with "
            "segmentation-guided attention learning, "
            "uncertainty estimation, and explainable AI."
        ),
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(
        router,
        prefix=settings.API_V1_PREFIX,
    )

    @app.get("/")
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "model_version": settings.MODEL_VERSION,
            "model_loaded": inference_service.is_loaded(),
            "checkpoint": settings.MODEL_CHECKPOINT_PATH,
            "docs": f"{settings.API_V1_PREFIX}/docs",
        }

    return app


app = create_app()