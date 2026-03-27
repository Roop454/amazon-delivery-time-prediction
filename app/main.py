from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model_service import ModelService
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(
    title="Amazon Delivery Time Predictor",
    version="1.0.0",
    description="Predict delivery ETA using logistics context such as distance, traffic, weather, and warehouse load.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService(BASE_DIR / "models" / "eta_model.joblib")

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    metadata = service.metadata
    return HealthResponse(
        status="ok",
        model_name=str(metadata.get("model_name", "unknown")),
        trained_at=metadata["trained_at"],
        metrics=metadata.get("metrics", {}),
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_eta(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(**service.predict(request))
