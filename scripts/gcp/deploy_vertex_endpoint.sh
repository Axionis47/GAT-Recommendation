#!/usr/bin/env bash
# Master script to deploy GAT-Recommendation to Vertex AI Endpoints
#
# This script orchestrates the full deployment:
# 1. Export production ONNX model
# 2. Upload artifacts to GCS
# 3. Build and push Docker images
# 4. Register models with Vertex AI
# 5. Deploy endpoint with A/B traffic split
# 6. Run benchmark
#
# Usage:
#   bash scripts/gcp/deploy_vertex_endpoint.sh
#
# Prerequisites:
#   - GCP project with Vertex AI API enabled
#   - Artifact Registry repository
#   - GCS bucket for model artifacts
#   - Docker installed and authenticated to Artifact Registry
#   - .env file with GCP configuration

set -euo pipefail

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Required environment variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID not set. Set in .env or export GCP_PROJECT_ID=your-project}"
: "${GCP_REGION:?GCP_REGION not set. Set in .env or export GCP_REGION=us-central1}"
: "${GCS_BUCKET:?GCS_BUCKET not set. Set in .env or export GCS_BUCKET=your-bucket}"
: "${AR_REPO:?AR_REPO not set. Set in .env or export AR_REPO=etpgt-repo}"

# Configuration
MODEL_VERSION="${MODEL_VERSION:-$(date +%Y%m%d-%H%M%S)}"
PYTORCH_IMAGE="etpgt-serve"
ONNX_IMAGE="etpgt-serve-onnx"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENDPOINT_NAME="${ENDPOINT_NAME:-gat-recommendation-endpoint}"

# Image URIs
PYTORCH_IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${PYTORCH_IMAGE}:${IMAGE_TAG}"
ONNX_IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${ONNX_IMAGE}:${IMAGE_TAG}"

# GCS paths
GCS_PYTORCH_URI="gs://${GCS_BUCKET}/models/serving/${MODEL_VERSION}/pytorch"
GCS_ONNX_URI="gs://${GCS_BUCKET}/models/serving/${MODEL_VERSION}/onnx"

echo "=================================================================="
echo "VERTEX AI ENDPOINT DEPLOYMENT"
echo "=================================================================="
echo "Project:       ${GCP_PROJECT_ID}"
echo "Region:        ${GCP_REGION}"
echo "Bucket:        ${GCS_BUCKET}"
echo "AR Repo:       ${AR_REPO}"
echo "Model Version: ${MODEL_VERSION}"
echo "Endpoint:      ${ENDPOINT_NAME}"
echo ""
echo "Images:"
echo "  PyTorch: ${PYTORCH_IMAGE_URI}"
echo "  ONNX:    ${ONNX_IMAGE_URI}"
echo "=================================================================="
echo ""

# Confirm
read -p "Continue with deployment? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

cd "${PROJECT_ROOT}"

# Step 1: Export production ONNX model
echo ""
echo "=================================================================="
echo "STEP 1: Export production ONNX model"
echo "=================================================================="

if [ ! -f "exports/onnx/production/session_recommender.onnx" ]; then
    python scripts/pipeline/export_onnx.py --production
else
    echo "ONNX model already exists, skipping export."
fi

# Step 2: Upload model artifacts to GCS
echo ""
echo "=================================================================="
echo "STEP 2: Upload model artifacts to GCS"
echo "=================================================================="

export MODEL_VERSION="${MODEL_VERSION}"
bash scripts/gcp/04_upload_model_artifacts.sh

# Step 3: Build and push Docker images
echo ""
echo "=================================================================="
echo "STEP 3: Build and push Docker images"
echo "=================================================================="

# Configure Docker for Artifact Registry
echo "Configuring Docker for Artifact Registry..."
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# Build PyTorch image
echo ""
echo "Building PyTorch inference image..."
docker build \
    -f docker/serve.Dockerfile \
    -t "${PYTORCH_IMAGE_URI}" \
    --platform linux/amd64 \
    .

echo "Pushing PyTorch image..."
docker push "${PYTORCH_IMAGE_URI}"

# Build ONNX image
echo ""
echo "Building ONNX inference image..."
docker build \
    -f docker/serve-onnx.Dockerfile \
    -t "${ONNX_IMAGE_URI}" \
    --platform linux/amd64 \
    .

echo "Pushing ONNX image..."
docker push "${ONNX_IMAGE_URI}"

# Step 4: Register models with Vertex AI
echo ""
echo "=================================================================="
echo "STEP 4: Register models with Vertex AI"
echo "=================================================================="

# Register PyTorch model
echo ""
echo "Registering PyTorch model..."
PYTORCH_MODEL_NAME="gat-rec-pytorch-${MODEL_VERSION}"

python scripts/gcp/05_register_vertex_model.py \
    --project-id "${GCP_PROJECT_ID}" \
    --region "${GCP_REGION}" \
    --model-name "${PYTORCH_MODEL_NAME}" \
    --container-image "${PYTORCH_IMAGE_URI}" \
    --artifact-uri "${GCS_PYTORCH_URI}" \
    --variant pytorch \
    --description "GAT-Recommendation PyTorch model v${MODEL_VERSION}"

# Get model ID (extract from list)
PYTORCH_MODEL_ID=$(gcloud ai models list \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_REGION}" \
    --filter="displayName=${PYTORCH_MODEL_NAME}" \
    --format="value(name)" \
    | head -1)

echo "PyTorch Model ID: ${PYTORCH_MODEL_ID}"

# Register ONNX model
echo ""
echo "Registering ONNX model..."
ONNX_MODEL_NAME="gat-rec-onnx-${MODEL_VERSION}"

python scripts/gcp/05_register_vertex_model.py \
    --project-id "${GCP_PROJECT_ID}" \
    --region "${GCP_REGION}" \
    --model-name "${ONNX_MODEL_NAME}" \
    --container-image "${ONNX_IMAGE_URI}" \
    --artifact-uri "${GCS_ONNX_URI}" \
    --variant onnx \
    --description "GAT-Recommendation ONNX model v${MODEL_VERSION}"

# Get model ID
ONNX_MODEL_ID=$(gcloud ai models list \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_REGION}" \
    --filter="displayName=${ONNX_MODEL_NAME}" \
    --format="value(name)" \
    | head -1)

echo "ONNX Model ID: ${ONNX_MODEL_ID}"

# Step 5: Deploy to endpoint with A/B traffic split
echo ""
echo "=================================================================="
echo "STEP 5: Deploy to Vertex AI Endpoint"
echo "=================================================================="

python scripts/gcp/06_deploy_endpoint.py \
    --project-id "${GCP_PROJECT_ID}" \
    --region "${GCP_REGION}" \
    --endpoint-name "${ENDPOINT_NAME}" \
    --pytorch-model-id "${PYTORCH_MODEL_ID}" \
    --onnx-model-id "${ONNX_MODEL_ID}" \
    --pytorch-traffic 50 \
    --machine-type n1-standard-4 \
    --min-replicas 1 \
    --max-replicas 3 \
    --ab-test

# Step 6: Run benchmark
echo ""
echo "=================================================================="
echo "STEP 6: Run benchmark"
echo "=================================================================="

# Wait for deployment to stabilize
echo "Waiting 60 seconds for deployment to stabilize..."
sleep 60

python scripts/gcp/07_benchmark_endpoint.py \
    --project-id "${GCP_PROJECT_ID}" \
    --region "${GCP_REGION}" \
    --endpoint-name "${ENDPOINT_NAME}" \
    --num-requests 50 \
    --concurrent 5 \
    --output "benchmark_results_${MODEL_VERSION}.json"

# Summary
echo ""
echo "=================================================================="
echo "DEPLOYMENT COMPLETE"
echo "=================================================================="
echo ""
echo "Model Version: ${MODEL_VERSION}"
echo "Endpoint:      ${ENDPOINT_NAME}"
echo "Traffic Split: PyTorch=50%, ONNX=50%"
echo ""
echo "Test the endpoint:"
echo ""
echo "  curl -X POST 'https://${GCP_REGION}-aiplatform.googleapis.com/v1/projects/${GCP_PROJECT_ID}/locations/${GCP_REGION}/endpoints/${ENDPOINT_NAME}:predict' \\"
echo "    -H 'Authorization: Bearer \$(gcloud auth print-access-token)' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"instances\": [{\"session_items\": [1, 2, 3], \"k\": 10}]}'"
echo ""
echo "Benchmark results: benchmark_results_${MODEL_VERSION}.json"
echo "=================================================================="
