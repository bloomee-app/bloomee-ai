Possible Region:
california_poppies
france_lavender
japan_cherry
netherlands_tulips
texas_bluebonnets
uk_bluebells
usa_cherry_dc

EXAMPLE CURL:

CHECK REGIONS:
^_^ ☛ curl "http://localhost:8000/regions"
{"total_regions":7,"regions":[{"id":"usa_cherry_dc","name":"USA Cherry Blossoms (Washington DC)","description":"Cherry blossoms in Washington DC","model_loaded":true},{"id":"japan_cherry","name":"Japan Cherry Blossoms","description":"Cherry blossoms in Japan","model_loaded":true},{"id":"netherlands_tulips","name":"Netherlands Tulips","description":"Tulip fields in the Netherlands","model_loaded":true},{"id":"france_lavender","name":"France Lavender","description":"Lavender fields in Provence, France","model_loaded":true},{"id":"uk_bluebells","name":"UK Bluebells","description":"Bluebell forests in the UK","model_loaded":true},{"id":"california_poppies","name":"California Poppies","description":"California poppy fields","model_loaded":true},{"id":"texas_bluebonnets","name":"Texas Bluebonnets","description":"Texas bluebonnet fields","model_loaded":true}]}%


PREDICT NDVI FROM A day:
^_^ ☛ curl "http://localhost:8000/predict/japan_cherry?date=2025-10-1"
{"region":"japan_cherry","region_name":"Japan Cherry Blossoms","date":"2025-10-1","ndvi_score":0.1127,"bloom_status":"Pre-Bloom","confidence":"High"}%
F

ORECAST:
^_^ ☛ curl "http://localhost:8000/predict/usa_cherry_dc/forecast?start_date=2025-04-01&days=14"
{"region":"usa_cherry_dc","region_name":"USA Cherry Blossoms (Washington DC)","forecast_start":"2025-04-01","forecast_days":14,"predictions":[{"date":"2025-04-01","ndvi_score":0.2639,"bloom_status":"Pre-Bloom"},{"date":"2025-04-02","ndvi_score":0.2639,"bloom_status":"Pre-Bloom"},{"date":"2025-04-03","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-04","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-05","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-06","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-07","ndvi_score":0.2639,"bloom_status":"Pre-Bloom"},{"date":"2025-04-08","ndvi_score":0.2639,"bloom_status":"Pre-Bloom"},{"date":"2025-04-09","ndvi_score":0.2639,"bloom_status":"Pre-Bloom"},{"date":"2025-04-10","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-11","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-12","ndvi_score":0.2654,"bloom_status":"Pre-Bloom"},{"date":"2025-04-13","ndvi_score":0.265,"bloom_status":"Pre-Bloom"},{"date":"2025-04-14","ndvi_score":0.2635,"bloom_status":"Pre-Bloom"}]}%
