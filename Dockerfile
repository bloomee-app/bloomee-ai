FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY blomee_api.py .
COPY database.py .
COPY feature_engineering.py .
COPY seed_database.py .
RUN mkdir -p models
COPY models/ ./models/
RUN mkdir -p models_simple
COPY models_simple/ ./models_simple/
RUN mkdir -p data
COPY data/ ./data/
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "blomee_api:app", "--host", "0.0.0.0", "--port", "8000"]
