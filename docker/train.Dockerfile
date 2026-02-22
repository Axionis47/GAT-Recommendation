# Training Docker image for Vertex AI
# Using PyTorch 2.1.0 with CUDA 11.8 (Python 3.10)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (lock file for reproducible builds, fallback to unpinned)
COPY requirements.lock* requirements.txt ./

# Install Python dependencies (prefer lock file if available)
RUN if [ -f requirements.lock ]; then \
        pip install --no-cache-dir -r requirements.lock; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Install PyTorch Geometric dependencies
RUN pip install --no-cache-dir \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install Google Cloud Storage
RUN pip install --no-cache-dir google-cloud-storage

# Copy package
COPY etpgt/ etpgt/
COPY scripts/ scripts/
COPY pyproject.toml .
COPY README.md .

# Install package
RUN pip install --no-cache-dir -e .

# Run as non-root user for security
RUN useradd -m -r trainuser && \
    chown -R trainuser:trainuser /app
USER trainuser

# Health check (no HTTP server, so validate Python + torch import)
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; print('healthy')" || exit 1

# Default entrypoint (can be overridden)
# For baselines: python scripts/train/train_baseline.py
ENTRYPOINT ["python"]

