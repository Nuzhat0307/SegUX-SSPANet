"""
SegUX-SSPANet Brain Tumor Diagnosis System
Configuration module — loads settings from environment variables.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = "SegUX-SSPANet Brain Tumor Diagnosis System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Security
    SECRET_KEY: str = "segux-sspanet-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Database (Supabase / PostgreSQL)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/segux_sspanet"
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Model
    MODEL_VERSION: str = "SegUX-SSPANet-v1.0.0"
    MODEL_CHECKPOINT_PATH: str = "ml/checkpoints/segux_sspanet_best.pth"
    NUM_CLASSES: int = 4
    IMAGE_SIZE: int = 224
    SEGMENTATION_SIZE: int = 256
    MC_DROPOUT_SAMPLES: int = 30
    UNCERTAINTY_THRESHOLD: float = 0.75

    # Tumor classes
    TUMOR_CLASSES: list[str] = ["glioma", "meningioma", "pituitary", "no_tumor"]

    # File storage
    MAX_UPLOAD_SIZE_MB: int = 10
    UPLOAD_DIR: str = "uploads"
    REPORT_DIR: str = "reports"

    # Dataset
    DATASET_DIR: str = "ml/data"
    FIGSHARE_URL: str = "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 50

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
