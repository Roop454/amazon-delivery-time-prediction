from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import PredictionRequest
from ml.train import train_and_save_model



def haversine_distance_km(pickup_lat: float, pickup_lng: float, drop_lat: float, drop_lng: float) -> float:
    radius_km = 6371.0

    p_lat = np.radians(pickup_lat)
    p_lng = np.radians(pickup_lng)
    d_lat = np.radians(drop_lat)
    d_lng = np.radians(drop_lng)

    delta_lat = d_lat - p_lat
    delta_lng = d_lng - p_lng

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(p_lat) * np.cos(d_lat) * np.sin(delta_lng / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))

    return float(radius_km * c)


class ModelService:
    def __init__(self, model_path: str | Path = "models/eta_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self._load_or_train()

    def _load_or_train(self) -> None:
        if not self.model_path.exists():
            train_and_save_model(self.model_path)

        payload = joblib.load(self.model_path)
        self.pipeline = payload["model"]
        self.metadata = payload["metadata"]

    def _prepare_row(self, request: PredictionRequest) -> pd.DataFrame:
        distance_km = haversine_distance_km(
            request.pickup_lat,
            request.pickup_lng,
            request.drop_lat,
            request.drop_lng,
        )

        order_dt = request.order_timestamp
        if order_dt.tzinfo is None:
            order_dt = order_dt.replace(tzinfo=timezone.utc)

        row = {
            "distance_km": max(distance_km, 0.1),
            "traffic_bucket": request.traffic_bucket.value,
            "weather_bucket": request.weather_bucket.value,
            "vehicle_type": request.vehicle_type.value,
            "package_weight_kg": request.package_weight_kg,
            "warehouse_load_pct": request.warehouse_load_pct,
            "hour_of_day": order_dt.hour,
            "is_weekend": int(order_dt.weekday() >= 5),
        }

        return pd.DataFrame([row])

    def _build_explanation(self, row: pd.Series) -> list[str]:
        notes: list[str] = []

        if row["distance_km"] > 60:
            notes.append("Long distance route is adding major travel time.")
        elif row["distance_km"] < 12:
            notes.append("Short route keeps travel time relatively low.")

        if row["traffic_bucket"] in {"high", "extreme"}:
            notes.append("Current traffic conditions are expected to slow delivery speed.")

        if row["weather_bucket"] in {"rain", "storm"}:
            notes.append("Weather risk is increasing handling and transit delays.")

        if row["warehouse_load_pct"] > 80:
            notes.append("Warehouse is highly loaded, so dispatch latency is likely.")

        if row["package_weight_kg"] > 12:
            notes.append("Heavy package weight adds sorting and loading time.")

        if not notes:
            notes.append("Overall conditions are stable for normal fulfillment flow.")

        return notes

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        frame = self._prepare_row(request)
        raw_eta = float(self.pipeline.predict(frame)[0])
        eta_minutes = max(round(raw_eta, 2), 5.0)

        confidence_margin = max(6.0, round(eta_minutes * 0.12, 2))

        order_time = request.order_timestamp
        if order_time.tzinfo is None:
            order_time = order_time.replace(tzinfo=timezone.utc)

        return {
            "eta_minutes": eta_minutes,
            "eta_window_min": round(max(eta_minutes - confidence_margin, 5.0), 2),
            "eta_window_max": round(eta_minutes + confidence_margin, 2),
            "distance_km": round(float(frame.iloc[0]["distance_km"]), 2),
            "estimated_delivery_time_utc": order_time + timedelta(minutes=eta_minutes),
            "model_name": str(self.metadata.get("model_name", "unknown")),
            "trained_at": datetime.fromisoformat(str(self.metadata["trained_at"])),
            "explanation": self._build_explanation(frame.iloc[0]),
        }
