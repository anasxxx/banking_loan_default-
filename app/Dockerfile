FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    lightgbm==4.1.0 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    joblib==1.3.2 \
    scikit-learn==1.3.2 \
    python-multipart==0.0.6 \
    requests==2.31.0 \
    prometheus-client==0.19.0 \
    prometheus-fastapi-instrumentator==6.1.0

# Copy application code
COPY app/main.py .
COPY app/config.py .
COPY app/schemas.py .
COPY app/__init__.py .
COPY app/feature_preparation.py .

# DON'T copy model - it will be mounted from EC2

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]