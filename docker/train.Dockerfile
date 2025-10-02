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

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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

# Default entrypoint (can be overridden)
# For baselines: python scripts/train/train_baseline.py
# For ETP-GT: python scripts/train/train_etpgt.py
ENTRYPOINT ["python"]

