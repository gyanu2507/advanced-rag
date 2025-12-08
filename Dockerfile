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

# Expose port (Render uses 10000 by default)
EXPOSE 10000

# Set default PORT for Render
ENV PORT=10000

# Command using shell to expand $PORT
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port $PORT"]
