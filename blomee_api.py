from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

REGIONS = {
    "usa_cherry_dc": {
        "name": "USA Cherry Blossoms (Washington DC)",
        "model_path": "models/usa_cherry_dc_simple_model.pkl",
        "preprocessor_path": "models/usa_cherry_dc_simple_preprocessor.pkl",
        "description": "Cherry blossoms in Washington DC",
    },
    "japan_cherry": {
        "name": "Japan Cherry Blossoms",
        "model_path": "models/japan_cherry_simple_model.pkl",
        "preprocessor_path": "models/japan_cherry_simple_preprocessor.pkl",
        "description": "Cherry blossoms in Japan",
    },
    "netherlands_tulips": {
        "name": "Netherlands Tulips",
        "model_path": "models/netherlands_tulips_simple_model.pkl",
        "preprocessor_path": "models/netherlands_tulips_simple_preprocessor.pkl",
        "description": "Tulip fields in the Netherlands",
    },
    "france_lavender": {
        "name": "France Lavender",
        "model_path": "models/france_lavender_simple_model.pkl",
        "preprocessor_path": "models/france_lavender_simple_preprocessor.pkl",
        "description": "Lavender fields in Provence, France",
    },
    "uk_bluebells": {
        "name": "UK Bluebells",
        "model_path": "models/uk_bluebells_simple_model.pkl",
        "preprocessor_path": "models/uk_bluebells_simple_preprocessor.pkl",
        "description": "Bluebell forests in the UK",
    },
    "california_poppies": {
        "name": "California Poppies",
        "model_path": "models/california_poppies_simple_model.pkl",
        "preprocessor_path": "models/california_poppies_simple_preprocessor.pkl",
        "description": "California poppy fields",
    },
    "texas_bluebonnets": {
        "name": "Texas Bluebonnets",
        "model_path": "models/texas_bluebonnets_simple_model.pkl",
        "preprocessor_path": "models/texas_bluebonnets_simple_preprocessor.pkl",
        "description": "Texas bluebonnet fields",
    },
}


# ? load models

models = {}
preprocessors = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and preprocessors on startup"""
    print("Loading models...")
    for region_id, config in REGIONS.items():
        try:
            model_path = Path(config["model_path"])
            preprocessor_path = Path(config["preprocessor_path"])

            if model_path.exists() and preprocessor_path.exists():
                models[region_id] = joblib.load(model_path)
                preprocessors[region_id] = joblib.load(preprocessor_path)
                print(f"Loaded model for {config['name']}")
            else:
                print(f"Model files not found for {config['name']}")
        except Exception as e:
            print(f"Error loading {config['name']}: {str(e)}")

    print(f"Loaded {len(models)} models successfully!")
    yield
    print("Shutting down ...")


app = FastAPI(
    title="Blomee NDVI Forecasting API",
    description="API for predicting NDVI (vegetation index) for flower bloom regions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ? pydnti models req/res schems


class NDVIPredictionResponse(BaseModel):
    """Response model for NDVI prediction"""

    region: str = Field(..., description="Region identifier")
    region_name: str = Field(..., description="Human-readable region name")
    date: str = Field(..., description="Prediction date (YYYY-MM-DD)")
    ndvi_score: float = Field(..., description="Predicted NDVI value (0-1 scale)")
    bloom_status: str = Field(..., description="Bloom status based on NDVI")
    confidence: str = Field(..., description="Model confidence level")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    detail: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    models_loaded: int
    available_regions: list


def create_time_features(date: datetime) -> dict:
    """Create time-based features from a date"""
    return {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.weekday(),
        "day_of_year": date.timetuple().tm_yday,
        "week_of_year": date.isocalendar()[1],
        "quarter": (date.month - 1) // 3 + 1,
        "month_sin": np.sin(2 * np.pi * date.month / 12),
        "month_cos": np.cos(2 * np.pi * date.month / 12),
        "day_sin": np.sin(2 * np.pi * date.weekday() / 7),
        "day_cos": np.cos(2 * np.pi * date.weekday() / 7),
        "is_weekend": int(date.weekday() >= 5),
        "is_month_start": int(date.day == 1),
        "is_month_end": int(
            date.day
            == (date.replace(day=28) + timedelta(days=4)).replace(day=1).day - 1
        ),
        "is_quarter_start": int(date.month in [1, 4, 7, 10] and date.day == 1),
        "is_quarter_end": int(date.month in [3, 6, 9, 12] and date.day >= 28),
    }


def interpret_ndvi(ndvi: float) -> str:
    """Interpret NDVI score into bloom status"""
    if ndvi >= 0.7:
        return "Peak Bloom"
    elif ndvi >= 0.5:
        return "Active Bloom"
    elif ndvi >= 0.3:
        return "Early Bloom"
    elif ndvi >= 0.1:
        return "Pre-Bloom"
    else:
        return "Dormant"


def get_confidence(region: str) -> str:
    """Estimate confidence based on region (placeholder - enhance with actual model metrics)"""
    # ! In production,  calculate  from models validation metrics
    return "High"


# API Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Blomee NDVI Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "available_endpoints": {
            "health": "/health",
            "regions": "/regions",
            "predict": "/predict/{region}?date=YYYY-MM-DD",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_regions": list(models.keys()),
    }


@app.get("/regions", tags=["Info"])
async def list_regions():
    """List all available regions"""
    available_regions = []
    for region_id, config in REGIONS.items():
        available_regions.append(
            {
                "id": region_id,
                "name": config["name"],
                "description": config["description"],
                "model_loaded": region_id in models,
            }
        )
    return {"total_regions": len(available_regions), "regions": available_regions}


@app.get(
    "/predict/{region}",
    response_model=NDVIPredictionResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
    tags=["Prediction"],
)
async def predict_ndvi(
    region: str,
    date: str = Query(
        ...,
        description="Date for prediction in YYYY-MM-DD format",
        example="2025-04-15",
    ),
):
    """
    Predict NDVI score for a specific region and date

    **Parameters:**
    - **region**: Region identifier (e.g., 'usa_cherry_dc', 'japan_cherry')
    - **date**: Prediction date in YYYY-MM-DD format (e.g., '2025-04-15')

    **Returns:**
    - NDVI score (0-1 scale, higher = more vegetation)
    - Bloom status interpretation
    - Model confidence level
    """

    # Validate region
    if region not in REGIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region}' not found. Available regions: {list(REGIONS.keys())}",
        )

    if region not in models:
        raise HTTPException(
            status_code=404, detail=f"Model for region '{region}' not loaded"
        )

    # Parse and validate date
    try:
        prediction_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD (e.g., '2025-04-15')",
        )

    # Check if date is too far in the past or future
    today = datetime.now()
    if prediction_date < today - timedelta(days=365 * 5):
        raise HTTPException(
            status_code=400, detail="Date is too far in the past (max 5 years ago)"
        )
    if prediction_date > today + timedelta(days=365):
        raise HTTPException(
            status_code=400, detail="Date is too far in the future (max 1 year ahead)"
        )

    try:
        # Create features
        features = create_time_features(prediction_date)

        # Convert to DataFrame (model expects this format)
        features_df = pd.DataFrame([features])

        # Get model and preprocessor
        model = models[region]
        preprocessor = preprocessors[region]

        # Preprocess features
        features_processed = preprocessor.transform(features_df)

        # Make prediction
        ndvi_prediction = float(model.predict(features_processed)[0])

        # Clip to valid NDVI range [0, 1]
        ndvi_prediction = np.clip(ndvi_prediction, 0, 1)

        # Interpret result
        bloom_status = interpret_ndvi(ndvi_prediction)
        confidence = get_confidence(region)

        return NDVIPredictionResponse(
            region=region,
            region_name=REGIONS[region]["name"],
            date=date,
            ndvi_score=round(ndvi_prediction, 4),
            bloom_status=bloom_status,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}"
        )


@app.get("/predict/{region}/forecast", tags=["Prediction"])
async def forecast_ndvi(
    region: str,
    start_date: str = Query(
        ..., description="Start date (YYYY-MM-DD)", example="2025-04-01"
    ),
    days: int = Query(7, description="Number of days to forecast", ge=1, le=90),
):
    """
    Get NDVI forecast for multiple days

    **Parameters:**
    - **region**: Region identifier
    - **start_date**: Start date for forecast (YYYY-MM-DD)
    - **days**: Number of days to forecast (1-90)

    **Returns:**
    - List of NDVI predictions for each day
    """

    if region not in models:
        raise HTTPException(
            status_code=404, detail=f"Model for region '{region}' not loaded"
        )

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )

    forecast = []
    for i in range(days):
        current_date = start + timedelta(days=i)

        # Create features
        features = create_time_features(current_date)
        features_df = pd.DataFrame([features])

        # Predict
        features_processed = preprocessors[region].transform(features_df)
        ndvi_pred = float(models[region].predict(features_processed)[0])
        ndvi_pred = np.clip(ndvi_pred, 0, 1)

        forecast.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "ndvi_score": round(ndvi_pred, 4),
                "bloom_status": interpret_ndvi(ndvi_pred),
            }
        )

    return {
        "region": region,
        "region_name": REGIONS[region]["name"],
        "forecast_start": start_date,
        "forecast_days": days,
        "predictions": forecast,
    }


# Run Server


if __name__ == "__main__":
    uvicorn.run(
        "blomee_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
    )
