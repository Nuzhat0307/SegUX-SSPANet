"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID


# --- Auth ---
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
class TokenData(BaseModel):
    user_id: Optional[str] = None
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str
class UserLogin(BaseModel):
    email: EmailStr
    password: str
class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    full_name: Optional[str] = None
    created_at: datetime

# --- Patient ---
class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    mrn: str
    notes: Optional[str] = None
class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    mrn: str
    notes: Optional[str] = None
    created_at: datetime

# --- Prediction ---
class ClassPredictionSchema(BaseModel):
    label: str
    display_name: str
    probability: float
class UncertaintySchema(BaseModel):
    method: str
    num_samples: int
    predictive_entropy: float
    mutual_information: float
    confidence: float
    is_uncertain: bool
class SegmentationSchema(BaseModel):
    mask_base64: str
    overlay_base64: str
    dice_score: Optional[float] = None
    tumor_area_pixels: int
    tumor_area_percentage: float
    bounding_box: Optional[dict] = None
class GradCAMSchema(BaseModel):
    method: str
    heatmap_base64: str
    overlay_base64: str
class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    patient_id: str
    image_base64: str
    predicted_class: str
    predicted_class_display: str
    probabilities: List[ClassPredictionSchema]
    uncertainty: UncertaintySchema
    segmentation: SegmentationSchema
    gradcam_results: List[GradCAMSchema]
    feature_explanation: Optional[Any] = None
    model_version: str
    inference_time_ms: int
    created_at: datetime
    notes: Optional[str] = None

# --- Report ---
class ReportResponse(BaseModel):
    id: str
    prediction_id: str
    report_type: str
    created_at: datetime

# --- Health ---
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
