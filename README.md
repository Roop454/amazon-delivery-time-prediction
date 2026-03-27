# Amazon Delivery Time Prediction

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live-Demo-c96a2f)](https://<your-live-demo-url>)

A production-style machine learning project that predicts parcel delivery ETA using logistics features such as route distance, traffic level, weather, package weight, warehouse load, and vehicle type.

This repository is designed to be portfolio-ready for resume and GitHub showcase use.

## Live Demo

- App URL: `https://<your-live-demo-url>`
- API docs: `https://<your-live-demo-url>/docs`

## Why this project is credible

- Uses a complete ML workflow: data generation, feature engineering, model training, evaluation, serialization, and API inference.
- Includes a real web interface connected to backend prediction APIs.
- Produces explainable output with ETA range and driver factors.
- Organized as a maintainable codebase with tests and clear architecture.

## Tech Stack

- Backend: FastAPI
- Model: scikit-learn Gradient Boosting Regressor
- Data tools: pandas, numpy
- Frontend: HTML, CSS, JavaScript
- Testing: pytest

## Project Structure

```text
amazon/
  app/
    main.py
    model_service.py
    schemas.py
  ml/
    train.py
  web/
    index.html
    styles.css
    app.js
  tests/
    test_api.py
  models/
    .gitkeep
  requirements.txt
  README.md
```

## Local Setup

1. Create and activate virtual environment.

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start the API server.

```bash
uvicorn app.main:app --reload
```

4. Open browser.

```text
http://127.0.0.1:8000
```

The first run auto-trains and saves `models/eta_model.joblib`.

## API Endpoints

- `GET /api/health` -> service status and model metrics
- `POST /api/predict` -> returns predicted ETA, confidence window, and explanations

Sample request:

```json
{
  "pickup_lat": 19.0760,
  "pickup_lng": 72.8777,
  "drop_lat": 18.5204,
  "drop_lng": 73.8567,
  "traffic_bucket": "medium",
  "weather_bucket": "clear",
  "vehicle_type": "van",
  "package_weight_kg": 2.5,
  "warehouse_load_pct": 62
}
```

## Run Tests

```bash
pytest -q
```

## Publish on GitHub

1. Create a new empty repository on GitHub.
2. Initialize and push from this folder:

```bash
git init
git add .
git commit -m "Initial commit: Amazon delivery ETA predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

3. Replace placeholders in this README:

- `<your-username>`
- `<your-repo>`
- `<your-live-demo-url>`

## Deploy Live Demo (Render)

This project includes `render.yaml` and is ready for one-click deployment.

1. Push the repository to GitHub.
2. Go to Render and click `New +` -> `Blueprint`.
3. Connect your GitHub account and select this repo.
4. Render reads `render.yaml` and deploys automatically.
5. Copy the public URL and update this README badge and demo link.

## Portfolio Showcase Checklist

- Add 2 screenshots in GitHub: input form and prediction result panel.
- Pin this repository on your GitHub profile.
- Keep the live demo URL active and tested.
- Add the same project title in your resume and LinkedIn projects section.

## Resume Bullet Points

- Built an end-to-end delivery ETA prediction system using logistics-aware ML features and a Gradient Boosting model.
- Integrated trained model with FastAPI for real-time inference and confidence-range output.
- Designed and implemented a responsive dashboard for interactive input handling and prediction visualization.
- Added automated API tests and project documentation to support production-grade development workflows.

## Future Enhancements

- Replace synthetic data with historical shipment records.
- Add model monitoring for concept drift and latency.
- Add request logging and basic analytics dashboard.
