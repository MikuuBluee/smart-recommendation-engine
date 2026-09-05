FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies (TAMBAHAN: python3-dev untuk compile LightFM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# FIX: Downgrade setuptools agar lightfm bisa di-compile (lightfm tidak support setuptools >= 60)
RUN pip install --no-cache-dir "setuptools<60" wheel cython && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src ./src
COPY ./data ./data
# COPY ./models ./models

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]