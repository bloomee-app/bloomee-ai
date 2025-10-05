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

from fastapi.responses import StreamingResponse
from PIL import Image
import rasterio
import io
import numpy as np
from typing import Any
import httpx


# ? region config

REGIONS = {
    "bandung_floriculture": {
        "name": "Bandung Floriculture",
        "simple_model_path": "models_simple/bandung_floriculture_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/bandung_floriculture_simple_preprocessor.pkl",
        "full_model_path": "models/bandung_floriculture_model.pkl",
        "full_preprocessor_path": "models/bandung_floriculture_preprocessor.pkl",
        "description": "Lembang / Bandung highland flower farms and gardens",
        "lat": -6.8402,
        "lon": 107.5043,
        "image_dir": "bloomwatch_data/forecast_training/bandung_floriculture",
    },
    "usa_cherry_dc": {
        "name": "USA Cherry Blossoms (Washington DC)",
        "simple_model_path": "models_simple/usa_cherry_dc_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/usa_cherry_dc_simple_preprocessor.pkl",
        "full_model_path": "models/usa_cherry_dc_model.pkl",
        "full_preprocessor_path": "models/usa_cherry_dc_preprocessor.pkl",
        "description": "Cherry blossoms in Washington DC",
        "lat": 38.8853,
        "lon": -77.0386,
        "image_dir": "bloomwatch_data/forecast_training/usa_cherry_dc",
    },
    "japan_cherry": {
        "name": "Japan Cherry Blossoms",
        "simple_model_path": "models_simple/japan_cherry_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/japan_cherry_simple_preprocessor.pkl",
        "full_model_path": "models/japan_cherry_model.pkl",
        "full_preprocessor_path": "models/japan_cherry_preprocessor.pkl",
        "description": "Cherry blossoms in Japan",
        "lat": 34.8500,
        "lon": 135.7000,
        "image_dir": "bloomwatch_data/forecast_training/japan_cherry",
    },
    "netherlands_tulips": {
        "name": "Netherlands Tulips",
        "simple_model_path": "models_simple/netherlands_tulips_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/netherlands_tulips_simple_preprocessor.pkl",
        "full_model_path": "models/netherlands_tulips_model.pkl",
        "full_preprocessor_path": "models/netherlands_tulips_preprocessor.pkl",
        "description": "Tulip fields in the Netherlands",
        "lat": 52.3000,
        "lon": 4.5000,
        "image_dir": "bloomwatch_data/forecast_training/netherlands_tulips",
    },
    "france_lavender": {
        "name": "France Lavender",
        "simple_model_path": "models_simple/france_lavender_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/france_lavender_simple_preprocessor.pkl",
        "full_model_path": "models/france_lavender_model.pkl",
        "full_preprocessor_path": "models/france_lavender_preprocessor.pkl",
        "description": "Lavender fields in Provence, France",
        "lat": 43.8500,
        "lon": 5.4500,
        "image_dir": "bloomwatch_data/forecast_training/france_lavender",
    },
    "uk_bluebells": {
        "name": "UK Bluebells",
        "simple_model_path": "models_simple/uk_bluebells_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/uk_bluebells_simple_preprocessor.pkl",
        "full_model_path": "models/uk_bluebells_model.pkl",
        "full_preprocessor_path": "models/uk_bluebells_preprocessor.pkl",
        "description": "Bluebell forests in the UK",
        "lat": 51.5000,
        "lon": -0.6000,
        "image_dir": "bloomwatch_data/forecast_training/uk_bluebells",
    },
    "california_poppies": {
        "name": "California Poppies",
        "simple_model_path": "models_simple/california_poppies_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/california_poppies_simple_preprocessor.pkl",
        "full_model_path": "models/california_poppies_model.pkl",
        "full_preprocessor_path": "models/california_poppies_preprocessor.pkl",
        "description": "California poppy fields",
        "lat": 34.7000,
        "lon": -118.7500,
        "image_dir": "bloomwatch_data/forecast_training/california_poppies",
    },
    "texas_bluebonnets": {
        "name": "Texas Bluebonnets",
        "simple_model_path": "models_simple/texas_bluebonnets_simple_model.pkl",
        "simple_preprocessor_path": "models_simple/texas_bluebonnets_simple_preprocessor.pkl",
        "full_model_path": "models/texas_bluebonnets_model.pkl",
        "full_preprocessor_path": "models/texas_bluebonnets_preprocessor.pkl",
        "description": "Texas bluebonnet fields",
        "lat": 30.3000,
        "lon": -98.4500,
        "image_dir": "bloomwatch_data/forecast_training/texas_bluebonnets",
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


class WeatherData(BaseModel):
    """Weather information for a date"""

    temperature_max_c: Optional[float] = Field(None, description="Max temperature (°C)")
    temperature_min_c: Optional[float] = Field(None, description="Min temperature (°C)")
    temperature_mean_c: Optional[float] = Field(
        None, description="Mean temperature (°C)"
    )
    humidity_max_percent: Optional[float] = Field(
        None, description="Max relative humidity (%)"
    )
    humidity_min_percent: Optional[float] = Field(
        None, description="Min relative humidity (%)"
    )
    humidity_mean_percent: Optional[float] = Field(
        None, description="Mean relative humidity (%)"
    )
    precipitation_mm: Optional[float] = Field(
        None, description="Total precipitation (mm)"
    )
    rain_mm: Optional[float] = Field(None, description="Total rainfall (mm)")
    snowfall_cm: Optional[float] = Field(None, description="Total snowfall (cm)")
    wind_speed_max_kmh: Optional[float] = Field(
        None, description="Max wind speed (km/h)"
    )
    sunshine_duration_s: Optional[float] = Field(
        None, description="Sunshine duration (seconds)"
    )


class EnhancedNDVIPredictionResponse(BaseModel):
    """Enhanced response with weather data"""

    region: str
    region_name: str
    date: str
    ndvi_score: float
    bloom_status: str
    confidence: str
    weather: Optional[WeatherData] = Field(
        None, description="Weather data for the date"
    )
    satellite_image_available: bool = Field(
        False, description="Whether satellite image exists"
    )
    satellite_image_url: Optional[str] = Field(
        None, description="URL to get satellite image"
    )
    satellite_image_date: Optional[str] = Field(
        None, description="Actual date of nearest image"
    )


class EnhancedForecastDay(BaseModel):
    """Single day in forecast with weather and image data"""

    date: str
    ndvi_score: float
    bloom_status: str
    weather: Optional[WeatherData] = Field(
        None, description="Weather data for the date"
    )
    satellite_image_available: bool = Field(
        False, description="Whether satellite image exists"
    )
    satellite_image_url: Optional[str] = Field(
        None, description="URL to get satellite image"
    )
    satellite_image_date: Optional[str] = Field(
        None, description="Actual date of nearest image"
    )


class EnhancedForecastResponse(BaseModel):
    """Enhanced forecast response with weather and images"""

    region: str
    region_name: str
    forecast_start: str
    forecast_days: int
    model_type: str
    predictions: list[EnhancedForecastDay]


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


def find_nearest_image(
    region: str, target_date: datetime, tolerance_days: int = 7
) -> Optional[dict[str, Any]]:
    """
    Find nearest available satellite image for a date

    Returns:
    - image_path: Path to .tif file
    - actual_date: Date of the image
    - days_difference: Days between requested and actual
    """
    image_dir = Path(REGIONS[region]["image_dir"])

    if not image_dir.exists():
        return None

    # Get all .tif files in directory
    image_files = list(image_dir.glob("*.tif"))

    if not image_files:
        return None

    # Parse dates from filenames (format: bandung_floriculture_2024-06-15.tif)
    nearest_image = None
    min_difference = float("inf")

    for img_path in image_files:
        try:
            # Extract date from filename
            date_str = img_path.stem.split("_")[-1]  # e.g., "2024-06-15"
            img_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Calculate difference in days
            difference = abs((img_date - target_date).days)

            # Check if within tolerance and closer than previous
            if difference <= tolerance_days and difference < min_difference:
                min_difference = difference
                nearest_image = {
                    "image_path": img_path,
                    "actual_date": img_date.strftime("%Y-%m-%d"),
                    "days_difference": difference,
                    "requested_date": target_date.strftime("%Y-%m-%d"),
                }
        except (ValueError, IndexError):
            continue

    return nearest_image


def convert_tif_to_png(tif_path: Path) -> io.BytesIO:
    """
    Convert GeoTIFF to PNG for web display
    Applies NDVI colormap (red=low vegetation, green=high)
    """
    with rasterio.open(tif_path) as src:
        # Read NDVI band
        ndvi = src.read(1)

        # Normalize to 0-255 range
        ndvi_normalized = np.clip((ndvi + 1) * 127.5, 0, 255).astype(np.uint8)

        # Apply colormap (you can customize this)
        # Green channel = NDVI (high vegetation = bright green)
        img_array = np.zeros((ndvi.shape[0], ndvi.shape[1], 3), dtype=np.uint8)
        img_array[:, :, 1] = ndvi_normalized  # Green channel
        img_array[:, :, 0] = 255 - ndvi_normalized  # Red channel (inverse)

        # Convert to PIL Image
        img = Image.fromarray(img_array, mode="RGB")

        # Save to BytesIO buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer


async def fetch_weather_data(
    lat: float, lon: float, date: str
) -> Optional[dict[str, Any]]:
    """
    Fetch weather data from Open-Meteo API
    Free API: https://open-meteo.com/
    Returns temperature, humidity, precipitation, etc.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "precipitation_hours",
            "wind_speed_10m_max",
            "sunshine_duration",
        ],
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if "daily" not in data:
                return None

            daily = data["daily"]

            # Extract single day data (index 0)
            weather = {
                "temperature_max_c": daily["temperature_2m_max"][0],
                "temperature_min_c": daily["temperature_2m_min"][0],
                "temperature_mean_c": daily["temperature_2m_mean"][0],
                "humidity_max_percent": daily["relative_humidity_2m_max"][0],
                "humidity_min_percent": daily["relative_humidity_2m_min"][0],
                "humidity_mean_percent": daily["relative_humidity_2m_mean"][0],
                "precipitation_mm": daily["precipitation_sum"][0],
                "rain_mm": daily["rain_sum"][0],
                "snowfall_cm": daily["snowfall_sum"][0],
                "precipitation_hours": daily["precipitation_hours"][0],
                "wind_speed_max_kmh": daily["wind_speed_10m_max"][0],
                "sunshine_duration_s": daily["sunshine_duration"][0],
            }

            return weather

    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None


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
    response_model=EnhancedNDVIPredictionResponse,  # ✅ Changed from NDVIPredictionResponse
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
    include_weather: bool = Query(  # ✅ NEW parameter
        True, description="Include weather data from Open-Meteo API"
    ),
    include_images: bool = Query(
        True, description="Check satellite image availability"
    ),
):
    """
    Predict NDVI Score with Weather Data & Satellite Images

    - Weather data (temperature, humidity, precipitation)
    - Satellite image availability check
    - Enhanced bloom predictions

    Parameters:
    - region: Region identifier (e.g., 'usa_cherry_dc')
    - date: Prediction date (YYYY-MM-DD)
    - use_simple_model: Use fast model (16 features) vs full model (37 features)
    - include_weather: Fetch weather data from Open-Meteo (default: true)

    Returns:
    - NDVI prediction
    - Bloom status
    - Weather conditions (if available)
    - Satellite image info
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

        weather_data = None
        if include_weather:
            lat = REGIONS[region]["lat"]
            lon = REGIONS[region]["lon"]
            weather_raw = await fetch_weather_data(lat, lon, date)

            if weather_raw:
                weather_data = WeatherData(**weather_raw)

        # Check satellite image availability

        satellite_available = False
        satellite_url = None
        satellite_date = None

        if include_images:
            image_info = find_nearest_image(region, prediction_date, tolerance_days=7)
            satellite_available = image_info is not None
            satellite_url = (
                f"/predict/{region}/image?date={date}" if satellite_available else None
            )
            satellite_date = image_info["actual_date"] if satellite_available else None

        return EnhancedNDVIPredictionResponse(
            region=region,
            region_name=REGIONS[region]["name"],
            date=date,
            ndvi_score=round(ndvi_prediction, 4),
            bloom_status=bloom_status,
            confidence=f"Using {feature_type}",
            weather=weather_data,
            satellite_image_available=satellite_available,
            satellite_image_url=satellite_url,
            satellite_image_date=satellite_date,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}"
        )


@app.get(
    "/predict/{region}/forecast",
    response_model=EnhancedForecastResponse,
    tags=["Prediction"],
)
async def forecast_ndvi(
    region: str,
    start_date: str = Query(
        ..., description="Start date (YYYY-MM-DD)", example="2025-04-01"
    ),
    days: int = Query(7, description="Number of days to forecast", ge=1, le=90),
    use_simple_model: bool = Query(
        False, description="Use simple model (faster but less accurate)"
    ),
    include_weather: bool = Query(
        True, description="Include weather data for each day"
    ),
    include_images: bool = Query(
        True, description="Check satellite image availability for each day"
    ),
):
    """
    Get multi-day NDVI forecast with weather & satellite images

    - Weather data for each forecast day
    - Satellite image availability check for each day
    - Complete forecast data in single request

    Simple Model: Uses only time features (fast)
    Full Model: Uses lag/rolling features from previous predictions (accurate)

    Parameters:
    - region: Region identifier
    - start_date: Start date for forecast (YYYY-MM-DD)
    - days: Number of days to forecast (1-90)
    - use_simple_model: Set to `true` for faster predictions
    - include_weather: Fetch weather data for each day (default: true)
    - include_images: Check image availability for each day (default: true)

    Returns:
    - List of NDVI predictions with weather and image info for each day

    Example Use Case:
    User selects: "7-day forecast from 2025-04-01"
    → Frontend calls: /predict/usa_cherry_dc/forecast?start_date=2025-04-01&days=7
    → Gets all data at once (NDVI + weather + images)
    → User can slide through days without additional API call
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

    # Validate date range
    today = datetime.now()
    TRAINING_START_DATE = datetime(2022, 1, 1)

    if start < TRAINING_START_DATE:
        raise HTTPException(
            status_code=400,
            detail=f"Start date is before training data range (earliest: 2022-01-01)",
        )

    if start > today + timedelta(days=365 * 2):
        raise HTTPException(
            status_code=400,
            detail="Start date is too far in the future (max 2 years ahead)",
        )

    forecast = []
    lat = REGIONS[region]["lat"]
    lon = REGIONS[region]["lon"]

    for i in range(days):
        current_date = start + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        try:
            # ? STEP 1: NDVI PREDICTION
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

            bloom_status = interpret_ndvi(ndvi_pred)

            # ?  STEP 2: WEATHER DATA (if requested)

            weather_data = None
            if include_weather:
                weather_raw = await fetch_weather_data(lat, lon, date_str)
                if weather_raw:
                    weather_data = WeatherData(**weather_raw)

            # ?  STEP 3: SATELLITE IMAGE INFO (if requested)

            satellite_available = False
            satellite_url = None
            satellite_date = None

            if include_images:
                image_info = find_nearest_image(region, current_date, tolerance_days=7)
                satellite_available = image_info is not None
                satellite_url = (
                    f"/predict/{region}/image?date={date_str}"
                    if satellite_available
                    else None
                )
                satellite_date = (
                    image_info["actual_date"] if satellite_available else None
                )

            # ? STEP 4: BUILD FORECAST DAY

            forecast.append(
                EnhancedForecastDay(
                    date=date_str,
                    ndvi_score=round(ndvi_pred, 4),
                    bloom_status=bloom_status,
                    weather=weather_data,
                    satellite_image_available=satellite_available,
                    satellite_image_url=satellite_url,
                    satellite_image_date=satellite_date,
                )
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error forecasting for {date_str}: {str(e)}"
            )

    return EnhancedForecastResponse(
        region=region,
        region_name=REGIONS[region]["name"],
        forecast_start=start_date,
        forecast_days=days,
        model_type=(
            "simple (16 features)" if use_simple_model else "full (37 features)"
        ),
        predictions=forecast,
    )


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


@app.get("/predict/{region}/image", tags=["Satellite Images"])
async def get_satellite_image(
    region: str,
    date: str = Query(
        ...,
        description="Requested date (YYYY-MM-DD). Returns nearest available image within ±7 days",
        example="2024-06-15",
    ),
    tolerance_days: int = Query(
        7, description="Maximum days to search for nearest image", ge=1, le=30
    ),
):
    """
    Get satellite NDVI image for a region and date

    **Note:** MODIS doesn't capture images daily due to:
    - Cloud cover
    - Satellite orbit (8-day revisit)

    This endpoint returns the **nearest available image** within the tolerance window.

    Returns:
    - PNG image with NDVI visualization (green=vegetation, red=bare soil)
    - Metadata about actual image date
    """

    # Validate region
    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")

    # Parse date
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
        )

    # Find nearest image
    image_info = find_nearest_image(region, target_date, tolerance_days)

    if not image_info:
        raise HTTPException(
            status_code=404,
            detail=f"No satellite images found for {region} within {tolerance_days} days of {date}",
        )

    try:
        # Convert to PNG
        png_buffer = convert_tif_to_png(image_info["image_path"])

        # Return image with metadata in headers
        return StreamingResponse(
            png_buffer,
            media_type="image/png",
            headers={
                "X-Requested-Date": image_info["requested_date"],
                "X-Actual-Date": image_info["actual_date"],
                "X-Days-Difference": str(image_info["days_difference"]),
                "X-Region": region,
                "Content-Disposition": f"inline; filename={region}_{image_info['actual_date']}.png",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/predict/{region}/image/metadata", tags=["Satellite Images"])
async def get_image_metadata(
    region: str,
    date: str = Query(..., description="Date (YYYY-MM-DD)", example="2024-06-15"),
):
    """
    Get metadata about nearest available satellite image without downloading
    """

    if region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found")

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    image_info = find_nearest_image(region, target_date, tolerance_days=7)

    if not image_info:
        return {
            "available": False,
            "message": f"No images found for {region} near {date}",
        }

    return {
        "available": True,
        "region": region,
        "requested_date": image_info["requested_date"],
        "actual_date": image_info["actual_date"],
        "days_difference": image_info["days_difference"],
        "image_url": f"/predict/{region}/image?date={date}",
    }


if __name__ == "__main__":
    uvicorn.run(
        "blomee_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
