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

from database import db
from feature_engineering import create_all_features


# ? region config

REGIONS = {
    "bandung_floriculture": {
        "name": "Bandung Floriculture",
        "simple_model_path": "models_simple/bandung_floriculture_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/bandung_floriculture_simple_preprocessor.pkl",
        "full_model_path": "models/bandung_floriculture_model.pkl",
        "full_preprocessor_path": "models/bandung_floriculture_preprocessor.pkl",
        "description": "Lembang / Bandung highland flower farms and gardens",
    },
    "usa_cherry_dc": {
        "name": "USA Cherry Blossoms (Washington DC)",
        "simple_model_path": "models_simple/usa_cherry_dc_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/usa_cherry_dc_simple_preprocessor.pkl",
        "full_model_path": "models/usa_cherry_dc_model.pkl",
        "full_preprocessor_path": "models/usa_cherry_dc_preprocessor.pkl",
        "description": "Cherry blossoms in Washington DC",
    },
    "japan_cherry": {
        "name": "Japan Cherry Blossoms",
        "simple_model_path": "models_simple/japan_cherry_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/japan_cherry_simple_preprocessor.pkl",
        "full_model_path": "models/japan_cherry_model.pkl",
        "full_preprocessor_path": "models/japan_cherry_preprocessor.pkl",
        "description": "Cherry blossoms in Japan",
    },
    "netherlands_tulips": {
        "name": "Netherlands Tulips",
        "simple_model_path": "models_simple/netherlands_tulips_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/netherlands_tulips_simple_preprocessor.pkl",
        "full_model_path": "models/netherlands_tulips_model.pkl",
        "full_preprocessor_path": "models/netherlands_tulips_preprocessor.pkl",
        "description": "Tulip fields in the Netherlands",
    },
    "france_lavender": {
        "name": "France Lavender",
        "simple_model_path": "models_simple/france_lavender_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/france_lavender_simple_preprocessor.pkl",
        "full_model_path": "models/france_lavender_model.pkl",
        "full_preprocessor_path": "models/france_lavender_preprocessor.pkl",
        "description": "Lavender fields in Provence, France",
    },
    "uk_bluebells": {
        "name": "UK Bluebells",
        "simple_model_path": "models_simple/uk_bluebells_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/uk_bluebells_simple_preprocessor.pkl",
        "full_model_path": "models/uk_bluebells_model.pkl",
        "full_preprocessor_path": "models/uk_bluebells_preprocessor.pkl",
        "description": "Bluebell forests in the UK",
    },
    "california_poppies": {
        "name": "California Poppies",
        "simple_model_path": "models_simple/california_poppies_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/california_poppies_simple_preprocessor.pkl",
        "full_model_path": "models/california_poppies_model.pkl",
        "full_preprocessor_path": "models/california_poppies_preprocessor.pkl",
        "description": "California poppy fields",
    },
    "texas_bluebonnets": {
        "name": "Texas Bluebonnets",
        "simple_model_path": "models_simple/texas_bluebonnets_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/texas_bluebonnets_simple_preprocessor.pkl",
        "full_model_path": "models/texas_bluebonnets_model.pkl",
        "full_preprocessor_path": "models/texas_bluebonnets_preprocessor.pkl",
        "description": "Texas bluebonnet fields",
    },
}


# ?  GLOBAL MODEL STORAGE

simple_models = {}
simple_preprocessors = {}
full_models = {}
full_preprocessors = {}


# ? lifespans


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and preprocessors on startup"""

    print("LOADING MODELS")

    for region_id, config in REGIONS.items():
        region_name = config["name"]
        loaded_count = 0

        try:
            simple_model_path = Path(config["simple_model_path"])
            simple_preprocessor_path = Path(config["simple_preprocessor_path"])

            if simple_model_path.exists() and simple_preprocessor_path.exists():
                simple_models[region_id] = joblib.load(simple_model_path)
                simple_preprocessors[region_id] = joblib.load(simple_preprocessor_path)
                print(f"Loaded SIMPLE model for {region_name}")
                loaded_count += 1
            else:
                print(f"Simple model files not found for {region_name}")
        except Exception as e:
            print(f"Error loading simple model for {region_name}: {str(e)}")

        try:
            full_model_path = Path(config["full_model_path"])
            full_preprocessor_path = Path(config["full_preprocessor_path"])

            if full_model_path.exists() and full_preprocessor_path.exists():
                full_models[region_id] = joblib.load(full_model_path)
                full_preprocessors[region_id] = joblib.load(full_preprocessor_path)
                print(f"Loaded FULL model for {region_name}")
                loaded_count += 1
            else:
                print(f"Full model files not found for {region_name}")
        except Exception as e:
            print(f"Error loading full model for {region_name}: {str(e)}")

        if loaded_count == 0:
            print(f"NO MODELS loaded for {region_name}")

        print()

    print(f"Loaded {len(simple_models)} SIMPLE models")
    print(f"Loaded {len(full_models)} FULL models")
    print(f"Database ready at {db.db_path}")

    yield

    print("Shutting down...")


# ?  FASTAPI APP init


app = FastAPI(
    title="Blomee NDVI Forecasting API",
    description="API for predicting NDVI (vegetation index) for flower bloom regions",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ? PYDANTIC MODELS (Request/Response Schemas)
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
    simple_models: int
    full_models: int
    available_regions: list


# ? HELPER FUNCTIONS


def create_time_features(date: datetime) -> dict:
    """Create time-based features from a date (16 features)"""
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
        "is_month_end": int(date.day >= 28),
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


# ? API ENDPOINTS


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Blomee NDVI Forecasting API",
        "version": "2.0.0",
        "docs": "/docs",
        "available_endpoints": {
            "health": "/health",
            "regions": "/regions",
            "predict": "/predict/{region}?date=YYYY-MM-DD&use_simple_model=false",
            "forecast": "/predict/{region}/forecast?start_date=YYYY-MM-DD&days=7",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with model status"""
    return {
        "status": "healthy",
        "models_loaded": len(simple_models) + len(full_models),
        "simple_models": len(simple_models),
        "full_models": len(full_models),
        "available_regions": list(set(simple_models.keys()) | set(full_models.keys())),
    }


@app.get("/regions", tags=["Info"])
async def list_regions():
    """List all available regions with model availability"""
    available_regions = []
    for region_id, config in REGIONS.items():
        available_regions.append(
            {
                "id": region_id,
                "name": config["name"],
                "description": config["description"],
                "simple_model_loaded": region_id in simple_models,
                "full_model_loaded": region_id in full_models,
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
    use_simple_model: bool = Query(
        False,
        description="Use simple 16-feature model (faster, less accurate)",
    ),
):
    """
    Predict NDVI Score with Flexible Model Selection

    Choose between:
    - Simple Model (16 features): Fast, time-features only
    - Full Model (37 features): Accurate, uses historical data

    Parameters:
    - region: Region identifier (e.g., 'usa_cherry_dc')
    - date: Prediction date (YYYY-MM-DD)
    - use_simple_model:
        - `false` (default) → Use full 37-feature model
        - `true` → Use simple 16-feature model

    Returns:
    - NDVI score (0-1 scale)
    - Bloom status
    - Model type used
    """

    # Validate region
    if region not in REGIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region}' not found. Available: {list(REGIONS.keys())}",
        )

    # Check if requested model is available
    if use_simple_model and region not in simple_models:
        raise HTTPException(
            status_code=404, detail=f"Simple model for region '{region}' not loaded"
        )

    if not use_simple_model and region not in full_models:
        # Fallback to simple model if full model not available
        if region in simple_models:
            print(
                f"Full model not available for {region}, falling back to simple model"
            )
            use_simple_model = True
        else:
            raise HTTPException(
                status_code=404, detail=f"No models available for region '{region}'"
            )

    # Parse and validate date
    try:
        prediction_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD (e.g., '2025-04-15')",
        )

    # Date range validation
    today = datetime.now()
    #! DONT CHANGE unless needed because of new data
    TRAINING_START_DATE = datetime(2022, 1, 1)

    if prediction_date < TRAINING_START_DATE:
        raise HTTPException(
            status_code=400,
            detail="Date is before training data range (earliest: 2022-01-01)",
        )

    if prediction_date > today + timedelta(days=365 * 2):
        raise HTTPException(
            status_code=400, detail="Date is too far in the future (max 2 year ahead)"
        )

    try:
        if use_simple_model:
            # ? SIMPLE MODEL: Use only time features (16 features)
            features = create_time_features(prediction_date)
            features_df = pd.DataFrame([features])

            model = simple_models[region]
            preprocessor = simple_preprocessors[region]
            feature_type = "simple (16 time features)"

        else:
            # ? FULL MODEL: Use all 37 features with historical data
            # ? Retrieve recent history from database
            recent_history = db.get_recent_predictions(
                region=region,
                before_date=date,
                days=30,  # Get last 30 days for rolling windows
            )

            # ? Create all features (time + lag + rolling)
            features_df = create_all_features(prediction_date, recent_history)

            model = full_models[region]
            preprocessor = full_preprocessors[region]
            feature_type = (
                f"full (37 features, {len(recent_history)} historical records)"
            )

        # ? MAKE PREDICTION
        # Preprocess features
        features_processed = preprocessor.transform(features_df)

        # Predict
        ndvi_prediction = float(model.predict(features_processed)[0])
        ndvi_prediction = np.clip(ndvi_prediction, 0, 1)

        # Store prediction in database (for future full model predictions)
        db.store_prediction(region, date, ndvi_prediction)

        # Interpret result
        bloom_status = interpret_ndvi(ndvi_prediction)

        return NDVIPredictionResponse(
            region=region,
            region_name=REGIONS[region]["name"],
            date=date,
            ndvi_score=round(ndvi_prediction, 4),
            bloom_status=bloom_status,
            confidence=f"Using {feature_type}",
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
    use_simple_model: bool = Query(
        False, description="Use simple model (faster but less accurate)"
    ),
):
    """
    Get multi-day NDVI forecast

    Simple Model: Uses only time features (fast)
    Full Model: Uses lag/rolling features from previous predictions (accurate)

    Parameters:
    - region: Region identifier
    - start_date: Start date for forecast (YYYY-MM-DD)
    - days: Number of days to forecast (1-90)
    - use_simple_model: Set to `true` for faster predictions

    Returns:
    - List of NDVI predictions for each day
    """

    # Validate region
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")

    # Check model availability
    if use_simple_model and region not in simple_models:
        raise HTTPException(
            status_code=404, detail=f"Simple model for region '{region}' not loaded"
        )

    if not use_simple_model and region not in full_models:
        if region in simple_models:
            print(f"Full model not available, using simple model")
            use_simple_model = True
        else:
            raise HTTPException(
                status_code=404, detail=f"No models available for region '{region}'"
            )

    # Parse date
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )

    forecast = []

    for i in range(days):
        current_date = start + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        try:
            if use_simple_model:
                # Simple model: only time features
                features = create_time_features(current_date)
                features_df = pd.DataFrame([features])

                model = simple_models[region]
                preprocessor = simple_preprocessors[region]

            else:
                # Full model: time + lag + rolling features
                recent_history = db.get_recent_predictions(
                    region=region, before_date=date_str, days=30
                )

                features_df = create_all_features(current_date, recent_history)

                model = full_models[region]
                preprocessor = full_preprocessors[region]

            # Predict
            features_processed = preprocessor.transform(features_df)
            ndvi_pred = float(model.predict(features_processed)[0])
            ndvi_pred = np.clip(ndvi_pred, 0, 1)

            # Store for next iteration (important for full model)
            db.store_prediction(region, date_str, ndvi_pred)

            forecast.append(
                {
                    "date": date_str,
                    "ndvi_score": round(ndvi_pred, 4),
                    "bloom_status": interpret_ndvi(ndvi_pred),
                }
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error forecasting for {date_str}: {str(e)}"
            )

    return {
        "region": region,
        "region_name": REGIONS[region]["name"],
        "forecast_start": start_date,
        "forecast_days": days,
        "model_type": (
            "simple (16 features)" if use_simple_model else "full (37 features)"
        ),
        "predictions": forecast,
    }


@app.post("/admin/seed/{region}", tags=["Admin"])
async def seed_historical_data(
    region: str,
    csv_path: str = Query(..., description="Path to CSV file with historical data"),
):
    """
    Admin only: Seed database with historical NDVI data

    CSV must have columns: date, ndvi_mean
    """
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")

    try:
        db.seed_historical_data(region, csv_path)
        return {
            "status": "success",
            "message": f"Historical data seeded for {region}",
            "csv_path": csv_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error seeding data: {str(e)}")


# RUN SERVER


if __name__ == "__main__":
    uvicorn.run(
        "blomee_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
    )
