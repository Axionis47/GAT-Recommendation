#!/usr/bin/env bash
# Build and push training Docker image to Artifact Registry

set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "❌ .env file not found"
    exit 1
fi

# Required variables
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID not set}"
: "${GCP_REGION:?GCP_REGION not set}"
: "${AR_REPO:?AR_REPO not set}"

# Configuration
IMAGE_NAME="etpgt-train"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="docker/train.Dockerfile"

# Full image path
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=================================================="
echo "Building Training Docker Image"
echo "=================================================="
echo "Project: ${GCP_PROJECT_ID}"
echo "Region: ${GCP_REGION}"
echo "Repository: ${AR_REPO}"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "URI: ${IMAGE_URI}"
echo "=================================================="

# Configure Docker for Artifact Registry
echo ""
echo "📦 Configuring Docker authentication..."
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# Build image
echo ""
echo "🔨 Building Docker image..."
docker build \
    -f "${DOCKERFILE}" \
    -t "${IMAGE_URI}" \
    --platform linux/amd64 \
    .

echo "✅ Image built successfully"

# Push image
echo ""
echo "📤 Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}"

echo "✅ Image pushed successfully"

# Verify image
echo ""
echo "🔍 Verifying image in Artifact Registry..."
gcloud artifacts docker images list \
    "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}" \
    --filter="package=${IMAGE_NAME}" \
    --format="table(package, version, create_time, update_time)"

echo ""
echo "=================================================="
echo "✅ Training image ready!"
echo "=================================================="
echo "Image URI: ${IMAGE_URI}"
echo ""
echo "To use this image in Vertex AI:"
echo "  export TRAIN_IMAGE_URI=\"${IMAGE_URI}\""
echo "=================================================="

