# Inference Docker image for Vertex AI Endpoints - ONNX variant
# Lightweight Python image for optimized inference
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (lock file for reproducible builds, fallback to unpinned)
COPY requirements-serve-onnx.lock* requirements-serve-onnx.txt ./

# Install Python dependencies (prefer lock file if available)
RUN if [ -f requirements-serve-onnx.lock ]; then \
        pip install --no-cache-dir -r requirements-serve-onnx.lock; \
    else \
        pip install --no-cache-dir -r requirements-serve-onnx.txt; \
    fi

# Copy minimal package for inference
COPY etpgt/__init__.py etpgt/
COPY etpgt/utils/ etpgt/utils/
COPY scripts/__init__.py scripts/
COPY scripts/serve/ scripts/serve/

# Create model directory (model downloaded from GCS at runtime)
RUN mkdir -p /app/model

# Environment variables for Vertex AI
ENV PORT=8080
ENV MODEL_PATH=/app/model/session_recommender.onnx
ENV EMBEDDINGS_PATH=/app/model/item_embeddings.npy
ENV INFERENCE_MODE=onnx
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

# Expose port
EXPOSE 8080

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run as non-root user
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Run server
CMD ["python", "-m", "uvicorn", "scripts.serve.vertex_app:app", "--host", "0.0.0.0", "--port", "8080"]
