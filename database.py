import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


class PredictionDatabase:
    """SQLite database for storing NDVI predictions"""

    def __init__(self, db_path: str = "data/predictions.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # satellite feature columns
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                date DATE NOT NULL,
                ndvi_score REAL NOT NULL,
                ndvi_median REAL,
                ndvi_std REAL,
                ndvi_min REAL,
                ndvi_max REAL,
                valid_pixels INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(region, date)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_region_date
            ON predictions(region, date DESC)
        """
        )

        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")

    def store_prediction(
        self,
        region: str,
        date: str,
        ndvi_score: float,
        ndvi_median: float = None,
        ndvi_std: float = None,
        ndvi_min: float = None,
        ndvi_max: float = None,
        valid_pixels: int = None,
    ):
        """Store a prediction with optional satellite features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO predictions
            (region, date, ndvi_score, ndvi_median, ndvi_std, ndvi_min, ndvi_max, valid_pixels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                region,
                date,
                ndvi_score,
                ndvi_median,
                ndvi_std,
                ndvi_min,
                ndvi_max,
                valid_pixels,
            ),
        )

        conn.commit()
        conn.close()

    def get_recent_predictions(
        self, region: str, before_date: str, days: int = 30
    ) -> pd.DataFrame:
        """Get predictions for the last N days before a given date"""
        conn = sqlite3.connect(self.db_path)

        end_date = datetime.strptime(before_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=days)

        query = """
            SELECT date, ndvi_score, ndvi_median, ndvi_std, ndvi_min, ndvi_max, valid_pixels
            FROM predictions
            WHERE region = ?
              AND date >= ?
              AND date < ?
            ORDER BY date ASC
        """

        df = pd.read_sql_query(
            query, conn, params=(region, start_date.strftime("%Y-%m-%d"), before_date)
        )

        conn.close()

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

        return df

    def seed_historical_data(self, region: str, csv_path: str):
        """
        Seed database with historical data from CSV
        Now includes satellite features if available
        """
        df = pd.read_csv(csv_path)

        conn = sqlite3.connect(self.db_path)

        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions
                (region, date, ndvi_score, ndvi_median, ndvi_std, ndvi_min, ndvi_max, valid_pixels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    region,
                    row["date"],
                    row["ndvi_mean"],
                    row.get("ndvi_median", None),
                    row.get("ndvi_std", None),
                    row.get("ndvi_min", None),
                    row.get("ndvi_max", None),
                    row.get("valid_pixels", None),
                ),
            )

        conn.commit()
        conn.close()

        print(f"Seeded {len(df)} historical records for {region}")


db = PredictionDatabase()
