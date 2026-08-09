"""
API routes — authentication, patients, predictions, reports, health.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Optional
from loguru import logger
from pydantic import BaseModel
from datetime import datetime, timezone
from uuid import uuid4

from app.core.database import get_db
from app.core.security import create_access_token, verify_token
from app.core.config import settings
from app.models import User, Patient, Prediction, Report
from app.schemas import (
    Token, UserCreate, UserLogin, UserResponse,
    PatientCreate, PatientResponse,
    PredictionResponse, ReportResponse, HealthResponse,
)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")


# --- Dependencies ---
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Extract and verify the current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user_id = verify_token(token)
    if not user_id:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise credentials_exception
    return user


# --- Health ---
@router.get("/health", response_model=HealthResponse)
async def health_check():
    from app.services.inference import inference_service
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=inference_service.is_loaded(),
    )


# --- Auth ---
@router.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    from app.core.security import get_password_hash
    user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        created_at=user.created_at,
    )


@router.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate and return a JWT access token."""
    from app.core.security import verify_password
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = create_access_token(subject=str(user.id))
    return Token(access_token=token)


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
    )


# --- Patients ---
@router.post("/patients", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new patient record."""
    p = Patient(user_id=current_user.id, **patient.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return PatientResponse(
        id=str(p.id),
        name=p.name,
        age=p.age,
        gender=p.gender,
        mrn=p.mrn,
        notes=p.notes,
        created_at=p.created_at,
    )


@router.get("/patients", response_model=List[PatientResponse])
async def list_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all patients for the current user."""
    patients = db.query(Patient).filter(Patient.user_id == current_user.id).all()
    return [
        PatientResponse(
            id=str(p.id),
            name=p.name,
            age=p.age,
            gender=p.gender,
            mrn=p.mrn,
            notes=p.notes,
            created_at=p.created_at,
        )
        for p in patients
    ]


@router.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific patient by ID."""
    p = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.user_id == current_user.id,
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(
        id=str(p.id),
        name=p.name,
        age=p.age,
        gender=p.gender,
        mrn=p.mrn,
        notes=p.notes,
        created_at=p.created_at,
    )


# --- Predictions ---

from pydantic import BaseModel


class PredictRequest(BaseModel):
    patient_id: str
    image_base64: str


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictRequest,
):
    """
    Run AI inference on an MRI image.

    The frontend sends:
    - patient_id
    - image_base64

    FastAPI runs the trained model and returns the prediction.
    Database saving is handled by the frontend/Supabase.
    """

    from app.services.inference import inference_service

    try:
        logger.info(f"Prediction request received for patient: {request.patient_id}")

        result = await inference_service.predict(
            request.image_base64,
            request.patient_id,
        )

        logger.info(
            f"Prediction completed: "
            f"{result.get('predicted_class_display')} "
            f"in {result.get('inference_time_ms')} ms"
        )

        return PredictionResponse(
            id=str(uuid4()),
            patient_id=str(request.patient_id),
            image_base64=request.image_base64,
            predicted_class=result["predicted_class"],
            predicted_class_display=result["predicted_class_display"],
            probabilities=result["probabilities"],
            uncertainty=result["uncertainty"],
            segmentation=result["segmentation"],
            gradcam_results=result["gradcam_results"],
            model_version=result["model_version"],
            inference_time_ms=result["inference_time_ms"],
            created_at=result.get("created_at") or datetime.now(timezone.utc),
            notes=result.get("notes"),
        )

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )

@router.get("/predictions", response_model=List[PredictionResponse])
async def list_predictions(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List predictions for the current user."""
    preds = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_prediction_to_response(p) for p in preds]


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific prediction by ID."""
    pred = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id,
    ).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return _prediction_to_response(pred)


# --- Reports ---
@router.post("/reports/{prediction_id}", response_model=ReportResponse)
async def generate_report(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate and download a PDF report for a prediction."""
    pred = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id,
    ).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    patient = db.query(Patient).filter(Patient.id == pred.patient_id).first()

    from app.services.report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(pred, patient)

    report = Report(
        user_id=current_user.id,
        prediction_id=prediction_id,
        report_type="full",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return ReportResponse(
        id=str(report.id),
        prediction_id=str(report.prediction_id),
        report_type=report.report_type,
        created_at=report.created_at,
    )


@router.get("/reports", response_model=List[ReportResponse])
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all reports for the current user."""
    reports = (
        db.query(Report)
        .filter(Report.user_id == current_user.id)
        .order_by(Report.created_at.desc())
        .all()
    )
    return [
        ReportResponse(
            id=str(r.id),
            prediction_id=str(r.prediction_id),
            report_type=r.report_type,
            created_at=r.created_at,
        )
        for r in reports
    ]


# --- Helper ---
def _prediction_to_response(pred: Prediction) -> PredictionResponse:
    """Convert a Prediction ORM object to a response schema."""
    return PredictionResponse(
        id=str(pred.id),
        patient_id=str(pred.patient_id),
        image_base64=pred.image_base64,
        predicted_class=pred.predicted_class,
        predicted_class_display=pred.predicted_class_display,
        probabilities=pred.probabilities,
        uncertainty=pred.uncertainty,
        segmentation=pred.segmentation,
        gradcam_results=pred.gradcam_results,
        model_version=pred.model_version,
        inference_time_ms=pred.inference_time_ms,
        created_at=pred.created_at,
        notes=pred.notes,
    )
