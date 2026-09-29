# Lead Conversion Prediction API (optional REST interface)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import predict_lead, model_health


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Lead Conversion Prediction API",
    description="ML-powered API for predicting lead conversion probability.",
    version="1.0.0",
)


# ============================================================
# Request Schema
# ============================================================

class LeadRequest(BaseModel):
    lead_source: str
    industry: str
    location: str
    company_size: str

    lead_age_days: int = Field(ge=0)
    interactions: int = Field(ge=0)
    followups: int = Field(ge=0)
    response_time_hours: float = Field(ge=0)

    quotation_sent: int = Field(ge=0, le=1)
    quotation_value: float = Field(ge=0)

    website_visits: int = Field(ge=0)
    previous_customer: int = Field(ge=0, le=1)
    demo_attended: int = Field(ge=0, le=1)

    salesperson_experience: float = Field(ge=0)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Lead Conversion Prediction API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():
    try:
        health_status = model_health()

        return {
            "api_status": "healthy",
            "model": health_status,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model health check failed: {str(exc)}",
        )


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(request: LeadRequest):
    try:
        lead_data = request.model_dump()

        prediction = predict_lead(lead_data)

        return {
            "success": True,
            "prediction": prediction,
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )