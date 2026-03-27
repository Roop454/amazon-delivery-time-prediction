from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAFFIC_BUCKETS = ["low", "medium", "high", "extreme"]
WEATHER_BUCKETS = ["clear", "cloudy", "rain", "storm"]
VEHICLE_TYPES = ["bike", "van", "truck"]



def _haversine_distance_km(
    pickup_lat: np.ndarray,
    pickup_lng: np.ndarray,
    drop_lat: np.ndarray,
    drop_lng: np.ndarray,
) -> np.ndarray:
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
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_km * c



def generate_synthetic_data(n_samples: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pickup_lat = rng.uniform(8.0, 30.0, size=n_samples)
    pickup_lng = rng.uniform(68.0, 88.0, size=n_samples)
    drop_lat = pickup_lat + rng.normal(0.0, 0.45, size=n_samples)
    drop_lng = pickup_lng + rng.normal(0.0, 0.45, size=n_samples)

    distance_km = _haversine_distance_km(pickup_lat, pickup_lng, drop_lat, drop_lng)
    distance_km = np.clip(distance_km, 1.5, 220.0)

    traffic_bucket = rng.choice(TRAFFIC_BUCKETS, size=n_samples, p=[0.30, 0.42, 0.22, 0.06])
    weather_bucket = rng.choice(WEATHER_BUCKETS, size=n_samples, p=[0.48, 0.29, 0.18, 0.05])
    vehicle_type = rng.choice(VEHICLE_TYPES, size=n_samples, p=[0.26, 0.53, 0.21])

    package_weight_kg = np.clip(rng.lognormal(mean=0.7, sigma=0.6, size=n_samples), 0.2, 35.0)
    warehouse_load_pct = rng.uniform(18.0, 98.0, size=n_samples)
    hour_of_day = rng.integers(0, 24, size=n_samples)
    is_weekend = rng.integers(0, 2, size=n_samples)

    traffic_penalty_map = {"low": 0.95, "medium": 0.82, "high": 0.66, "extreme": 0.52}
    weather_penalty_map = {"clear": 1.00, "cloudy": 0.94, "rain": 0.84, "storm": 0.69}
    speed_by_vehicle = {"bike": 28.0, "van": 42.0, "truck": 34.0}

    traffic_penalty = np.vectorize(traffic_penalty_map.get)(traffic_bucket)
    weather_penalty = np.vectorize(weather_penalty_map.get)(weather_bucket)
    vehicle_speed = np.vectorize(speed_by_vehicle.get)(vehicle_type)

    rush_hour_penalty = np.where(((hour_of_day >= 8) & (hour_of_day <= 11)) | ((hour_of_day >= 17) & (hour_of_day <= 21)), 0.85, 1.0)
    load_penalty = np.clip(1.0 - (warehouse_load_pct / 350.0), 0.65, 1.0)

    effective_speed = vehicle_speed * traffic_penalty * weather_penalty * rush_hour_penalty * load_penalty
    effective_speed = np.clip(effective_speed, 7.0, 58.0)

    travel_minutes = (distance_km / effective_speed) * 60.0
    handling_minutes = 7.5 + (package_weight_kg * 0.95) + (warehouse_load_pct * 0.14) + (is_weekend * 3.2)
    random_noise = rng.normal(0.0, 5.2, size=n_samples)

    eta_minutes = np.clip(travel_minutes + handling_minutes + random_noise, 12.0, 360.0)

    return pd.DataFrame(
        {
            "pickup_lat": pickup_lat,
            "pickup_lng": pickup_lng,
            "drop_lat": drop_lat,
            "drop_lng": drop_lng,
            "distance_km": distance_km,
            "traffic_bucket": traffic_bucket,
            "weather_bucket": weather_bucket,
            "vehicle_type": vehicle_type,
            "package_weight_kg": package_weight_kg,
            "warehouse_load_pct": warehouse_load_pct,
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
            "eta_minutes": eta_minutes,
        }
    )



def build_pipeline() -> Pipeline:
    numeric_features = [
        "distance_km",
        "package_weight_kg",
        "warehouse_load_pct",
        "hour_of_day",
        "is_weekend",
    ]
    categorical_features = ["traffic_bucket", "weather_bucket", "vehicle_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    regressor = GradientBoostingRegressor(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        subsample=0.85,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])



def train_and_save_model(model_path: str | Path = "models/eta_model.joblib") -> dict[str, Any]:
    dataset = generate_synthetic_data()
    feature_columns = [
        "distance_km",
        "traffic_bucket",
        "weather_bucket",
        "vehicle_type",
        "package_weight_kg",
        "warehouse_load_pct",
        "hour_of_day",
        "is_weekend",
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[feature_columns],
        dataset["eta_minutes"],
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    payload = {
        "model": pipeline,
        "metadata": {
            "model_name": "gradient_boosting_eta_v1",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": feature_columns,
            "rows_trained": int(len(dataset)),
            "metrics": {
                "mae_minutes": round(float(mae), 3),
                "rmse_minutes": round(float(rmse), 3),
                "r2_score": round(float(r2), 4),
            },
        },
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    return payload


if __name__ == "__main__":
    artifact = train_and_save_model()
    metrics = artifact["metadata"]["metrics"]
    print("Model trained and saved to models/eta_model.joblib")
    print(
        f"MAE: {metrics['mae_minutes']} min | "
        f"RMSE: {metrics['rmse_minutes']} min | "
        f"R2: {metrics['r2_score']}"
    )
