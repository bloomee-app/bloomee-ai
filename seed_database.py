from database import db

HISTORICAL_DATA = {
    "usa_cherry_dc": "bloomwatch_data/forecast_ready/usa_cherry_dc_timeseries.csv",
    "japan_cherry": "bloomwatch_data/forecast_ready/japan_cherry_timeseries.csv",
    "netherlands_tulips": "bloomwatch_data/forecast_ready/netherlands_tulips_timeseries.csv",
    "france_lavender": "bloomwatch_data/forecast_ready/france_lavender_timeseries.csv",
    "uk_bluebells": "bloomwatch_data/forecast_ready/uk_bluebells_timeseries.csv",
    "california_poppies": "bloomwatch_data/forecast_ready/california_poppies_timeseries.csv",
    "texas_bluebonnets": "bloomwatch_data/forecast_ready/texas_bluebonnets_timeseries.csv",
    "bandung_floriculture": "bloomwatch_data/forecast_ready/bandung_floriculture_timeseries.csv",
}

for region, csv_path in HISTORICAL_DATA.items():
    print(f"\nSeeding {region}...")
    db.seed_historical_data(region, csv_path)

print("\nAll regions seeded!")
