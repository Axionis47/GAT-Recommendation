# Inference Docker image for Vertex AI Endpoints - PyTorch variant
# Using PyTorch 2.1.0 with CUDA 11.8 (Python 3.10)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (lock file for reproducible builds, fallback to unpinned)
COPY requirements-serve.lock* requirements-serve.txt ./

# Install Python dependencies (prefer lock file if available)
RUN if [ -f requirements-serve.lock ]; then \
        pip install --no-cache-dir -r requirements-serve.lock; \
    else \
        pip install --no-cache-dir -r requirements-serve.txt; \
    fi

# Install PyTorch Geometric (CPU-only for inference to reduce dependencies)
# Using CPU wheels since inference doesn't need CUDA for this model
RUN pip install --no-cache-dir \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Copy package
COPY etpgt/ etpgt/
COPY scripts/serve/ scripts/serve/
COPY pyproject.toml .

# Install package
RUN pip install --no-cache-dir -e .

# Create model directory (model downloaded from GCS at runtime)
RUN mkdir -p /app/model

# Environment variables for Vertex AI
ENV PORT=8080
ENV MODEL_PATH=/app/model/model.pt
ENV INFERENCE_MODE=pytorch
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

# Expose port
EXPOSE 8080

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run as non-root user
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Run server
CMD ["python", "-m", "uvicorn", "scripts.serve.vertex_app:app", "--host", "0.0.0.0", "--port", "8080"]
