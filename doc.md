** API Documentation**

### **🌸 Blomee API - Quick Start Guide**

#### **Base URL**

```
http://localhost:8000
```

---

### **📍 Available Endpoints**

| Endpoint                           | Method | Description                      |
| ---------------------------------- | ------ | -------------------------------- |
| `/`                                | GET    | API info & documentation links   |
| `/health`                          | GET    | Check API status & loaded models |
| `/regions`                         | GET    | List all available regions       |
| `/predict/{region}`                | GET    | Single-day NDVI prediction       |
| `/predict/{region}/forecast`       | GET    | Multi-day NDVI forecast          |
| `/predict/{region}/image`          | GET    | Get satellite image (PNG)        |
| `/predict/{region}/image/metadata` | GET    | Check image availability         |

---

### **🎯 Common Use Cases**

#### **1. Get Single Day Prediction (Full Data)**

```bash
# Complete data: NDVI + Weather + Image
curl "http://localhost:8000/predict/japan_cherry?date=2025-01-01&include_weather=true&include_images=true"
```

**Response:**

```json
{
  "region": "japan_cherry",
  "date": "2025-01-01",
  "ndvi_score": 0.4174,
  "bloom_status": "Early Bloom",
  "weather": {
    "temperature_mean_c": 4.4,
    "precipitation_mm": 0.0
  },
  "satellite_image_available": true,
  "satellite_image_url": "/predict/japan_cherry/image?date=2025-01-01"
}
```

---

#### **2. Get 7-Day Forecast (Timeline/Slider UI)**

```bash
# Perfect for frontend slider - all data in one request
curl "http://localhost:8000/predict/bandung_floriculture/forecast?start_date=2024-06-10&days=7&include_weather=true&include_images=true"
```

**Response:**

```json
{
  "region": "bandung_floriculture",
  "forecast_start": "2024-06-10",
  "forecast_days": 7,
  "predictions": [
    {
      "date": "2024-06-10",
      "ndvi_score": 0.5123,
      "bloom_status": "Active Bloom",
      "weather": {...},
      "satellite_image_url": "/predict/bandung_floriculture/image?date=2024-06-10"
    },
    // ... 6 more days
  ]
}
```

---

#### **3. View Satellite Image**

```bash
# In browser:
http://localhost:8000/predict/japan_cherry/image?date=2025-01-01

# Download:
curl "http://localhost:8000/predict/japan_cherry/image?date=2025-01-01" -o satellite.png
```

---

#### **4. Fast Prediction (NDVI Only)**

```bash
# Skip weather & images for speed
curl "http://localhost:8000/predict/usa_cherry_dc?date=2025-03-15&use_simple_model=true&include_weather=false&include_images=false"
```

---

### **⚙️ Query Parameters**

#### **Single Prediction (`/predict/{region}`)**

| Parameter          | Type    | Default      | Description               |
| ------------------ | ------- | ------------ | ------------------------- |
| `date`             | string  | **required** | Date (YYYY-MM-DD)         |
| `use_simple_model` | boolean | `false`      | Use fast 16-feature model |
| `include_weather`  | boolean | `true`       | Fetch weather data        |
| `include_images`   | boolean | `true`       | Check image availability  |

#### **Forecast (`/predict/{region}/forecast`)**

| Parameter          | Type    | Default      | Description                 |
| ------------------ | ------- | ------------ | --------------------------- |
| `start_date`       | string  | **required** | Start date (YYYY-MM-DD)     |
| `days`             | integer | `7`          | Forecast length (1-90 days) |
| `use_simple_model` | boolean | `false`      | Use fast model              |
| `include_weather`  | boolean | `true`       | Fetch weather for each day  |
| `include_images`   | boolean | `true`       | Check images for each day   |

---

### **🌍 Available Regions**

```bash
# List all regions:
curl "http://localhost:8000/regions"
```

| Region ID              | Name                  | Location      |
| ---------------------- | --------------------- | ------------- |
| `bandung_floriculture` | Bandung Floriculture  | Indonesia     |
| `usa_cherry_dc`        | USA Cherry Blossoms   | Washington DC |
| `japan_cherry`         | Japan Cherry Blossoms | Kyoto         |
| `netherlands_tulips`   | Netherlands Tulips    | Keukenhof     |
| `france_lavender`      | France Lavender       | Provence      |
| `uk_bluebells`         | UK Bluebells          | England       |
| `california_poppies`   | California Poppies    | USA           |
| `texas_bluebonnets`    | Texas Bluebonnets     | USA           |

---

### **🎨 NDVI Bloom Status**

| NDVI Range | Status       | Color (in images) |
| ---------- | ------------ | ----------------- |
| 0.7 - 1.0  | Peak Bloom   | 🟢 Bright Green   |
| 0.5 - 0.7  | Active Bloom | 🟡 Yellow-Green   |
| 0.3 - 0.5  | Early Bloom  | 🟠 Orange         |
| 0.1 - 0.3  | Pre-Bloom    | 🔴 Red-Orange     |
| 0.0 - 0.1  | Dormant      | 🔴 Red            |

---

### **⚡ Performance Tips**

| Scenario           | Best Parameters                                                    | Speed        |
| ------------------ | ------------------------------------------------------------------ | ------------ |
| **Quick check**    | `use_simple_model=true&include_weather=false&include_images=false` | ⚡ Very Fast |
| **Production app** | `use_simple_model=false&include_weather=true&include_images=true`  | 🔄 Medium    |
| **Timeline UI**    | Use `/forecast` endpoint (fetches all days at once)                | 🚀 Optimal   |

---

### **🔍 Interactive Documentation**

Visit **Swagger UI** for full interactive docs:

```
http://localhost:8000/docs
```

**Features:**

- Try API calls directly in browser
- See request/response examples
- View all parameters & models

---

### **🚨 Error Handling**

All errors return consistent format:

```json
{
  "detail": "Error message here"
}
```

**Common Errors:**

| Error                    | Cause               | Solution                             |
| ------------------------ | ------------------- | ------------------------------------ |
| `Region not found`       | Invalid region ID   | Check `/regions`                     |
| `Date before 2022-01-01` | Date too old        | Use dates 2022+                      |
| `No images found`        | No satellite data   | Normal - MODIS doesn't capture daily |
| `Model not loaded`       | Model files missing | Run training notebooks               |

---

### **📦 Setup**

```bash
# 1. Install dependencies
pip install fastapi uvicorn httpx rasterio pillow

# 2. Start API
python blomee_api.py

# 3. Access API
http://localhost:8000
```

---
