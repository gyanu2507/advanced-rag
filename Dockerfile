# Multi-stage Dockerfile for AI Document Q&A App
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python3", "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]

