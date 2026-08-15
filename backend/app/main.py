"""
FastAPI application factory and main entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.core.config import settings
from app.api.routes import router
from app.services.trained_inference import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Model version: {settings.MODEL_VERSION}")
    logger.info(f"Loading trained model from: {settings.MODEL_CHECKPOINT_PATH}")

    # Never start a diagnostic API with synthetic/mock inference.
    inference_service.load_models()
    if not inference_service.is_loaded():
        raise RuntimeError("Trained SegUX-SSPANet model failed to load")

    logger.info("TRAINED MODEL LOADED SUCCESSFULLY")
    logger.info(f"Checkpoint: {settings.MODEL_CHECKPOINT_PATH}")
    logger.info("Inference source: trained checkpoint only")
    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=("AI-powered brain tumor diagnosis system with segmentation-guided attention learning, uncertainty estimation, and explainable AI."),
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
    app.include_router(router, prefix=settings.API_V1_PREFIX)

    @app.get("/")
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "model_version": settings.MODEL_VERSION,
            "model_loaded": inference_service.is_loaded(),
            "inference_source": "trained_checkpoint" if inference_service.is_loaded() else "unavailable",
            "checkpoint": settings.MODEL_CHECKPOINT_PATH,
            "docs": f"{settings.API_V1_PREFIX}/docs",
        }

    return app


app = create_app()
