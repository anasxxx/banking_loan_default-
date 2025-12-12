FROM quay.io/astronomer/astro-runtime:12.1.1

# Install system dependencies for LightGBM
USER root
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
USER astro

# Copy notebooks and models
COPY notebooks /usr/local/airflow/notebooks
COPY models/production /usr/local/airflow/models/production

# Create directories
RUN mkdir -p /usr/local/airflow/data/processed \
             /usr/local/airflow/data/raw \
             /usr/local/airflow/outputs