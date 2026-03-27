from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TrafficBucket(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


class WeatherBucket(str, Enum):
    clear = "clear"
    cloudy = "cloudy"
    rain = "rain"
    storm = "storm"


class VehicleType(str, Enum):
    bike = "bike"
    van = "van"
    truck = "truck"


class PredictionRequest(BaseModel):
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lng: float = Field(..., ge=-180, le=180)
    drop_lat: float = Field(..., ge=-90, le=90)
    drop_lng: float = Field(..., ge=-180, le=180)
    traffic_bucket: TrafficBucket
    weather_bucket: WeatherBucket
    vehicle_type: VehicleType
    package_weight_kg: float = Field(..., ge=0.1, le=50)
    warehouse_load_pct: float = Field(..., ge=0, le=100)
    order_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("drop_lat")
    @classmethod
    def validate_drop_lat(cls, value: float, info: object) -> float:
        return value


class PredictionResponse(BaseModel):
    eta_minutes: float
    eta_window_min: float
    eta_window_max: float
    distance_km: float
    estimated_delivery_time_utc: datetime
    model_name: str
    trained_at: datetime
    explanation: list[str]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    trained_at: datetime
    metrics: dict[str, float]
