#!/usr/bin/env bash
# Submit ETP-GT training job to Vertex AI

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
: "${GCS_BUCKET:?GCS_BUCKET not set}"
: "${AR_REPO:?AR_REPO not set}"
: "${VAI_MACHINE:?VAI_MACHINE not set}"
: "${VAI_ACCELERATOR:?VAI_ACCELERATOR not set}"
: "${VAI_ACCELERATOR_COUNT:?VAI_ACCELERATOR_COUNT not set}"

# Parse arguments
JOB_NAME="${1:-etpgt-$(date +%Y%m%d-%H%M%S)}"

# Configuration
IMAGE_NAME="etpgt-train"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# ETP-GT specific arguments
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_TEMPORAL_BUCKETS="${NUM_TEMPORAL_BUCKETS:-7}"
NUM_PATH_BUCKETS="${NUM_PATH_BUCKETS:-3}"
DROPOUT="${DROPOUT:-0.1}"
READOUT_TYPE="${READOUT_TYPE:-mean}"
USE_LAPLACIAN_PE="${USE_LAPLACIAN_PE:-true}"
LAPLACIAN_K="${LAPLACIAN_K:-16}"
USE_CLS_TOKEN="${USE_CLS_TOKEN:-false}"

# Loss arguments
LOSS_TYPE="${LOSS_TYPE:-dual}"
LOSS_ALPHA="${LOSS_ALPHA:-0.7}"
LOSS_TEMPERATURE="${LOSS_TEMPERATURE:-1.0}"

# Training arguments
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
LR="${LR:-0.001}"
PATIENCE="${PATIENCE:-10}"

echo "=================================================="
echo "Submitting ETP-GT Training Job"
echo "=================================================="
echo "Project: ${GCP_PROJECT_ID}"
echo "Region: ${GCP_REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Image: ${IMAGE_URI}"
echo "Machine: ${VAI_MACHINE}"
echo "Accelerator: ${VAI_ACCELERATOR} x ${VAI_ACCELERATOR_COUNT}"
echo ""
echo "Model Configuration:"
echo "  Embedding Dim: ${EMBEDDING_DIM}"
echo "  Hidden Dim: ${HIDDEN_DIM}"
echo "  Layers: ${NUM_LAYERS}"
echo "  Heads: ${NUM_HEADS}"
echo "  Temporal Buckets: ${NUM_TEMPORAL_BUCKETS}"
echo "  Path Buckets: ${NUM_PATH_BUCKETS}"
echo "  Laplacian PE: ${USE_LAPLACIAN_PE} (k=${LAPLACIAN_K})"
echo "  CLS Token: ${USE_CLS_TOKEN}"
echo ""
echo "Loss Configuration:"
echo "  Type: ${LOSS_TYPE}"
echo "  Alpha: ${LOSS_ALPHA}"
echo "  Temperature: ${LOSS_TEMPERATURE}"
echo "=================================================="

# Build training arguments
# Docker ENTRYPOINT is "python", so we only need script path + args
TRAINING_ARGS=(
    "scripts/train/train_etpgt.py"
    "--embedding-dim=${EMBEDDING_DIM}"
    "--hidden-dim=${HIDDEN_DIM}"
    "--num-layers=${NUM_LAYERS}"
    "--num-heads=${NUM_HEADS}"
    "--num-temporal-buckets=${NUM_TEMPORAL_BUCKETS}"
    "--num-path-buckets=${NUM_PATH_BUCKETS}"
    "--dropout=${DROPOUT}"
    "--readout-type=${READOUT_TYPE}"
    "--laplacian-k=${LAPLACIAN_K}"
    "--loss-type=${LOSS_TYPE}"
    "--loss-alpha=${LOSS_ALPHA}"
    "--loss-temperature=${LOSS_TEMPERATURE}"
    "--batch-size=${BATCH_SIZE}"
    "--max-epochs=${MAX_EPOCHS}"
    "--lr=${LR}"
    "--patience=${PATIENCE}"
    "--gcs-bucket=${GCS_BUCKET}"
    "--device=cuda"
)

# Add boolean flags
if [ "${USE_LAPLACIAN_PE}" = "true" ]; then
    TRAINING_ARGS+=("--use-laplacian-pe")
fi

if [ "${USE_CLS_TOKEN}" = "true" ]; then
    TRAINING_ARGS+=("--use-cls-token")
fi

# Create job config YAML
JOB_CONFIG=$(cat <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: ${VAI_MACHINE}
      acceleratorType: ${VAI_ACCELERATOR}
      acceleratorCount: ${VAI_ACCELERATOR_COUNT}
    replicaCount: 1
    containerSpec:
      imageUri: ${IMAGE_URI}
      args:
$(printf '        - "%s"\n' "${TRAINING_ARGS[@]}")
scheduling:
  timeout: 86400s
  restartJobOnWorkerRestart: false
EOF
)

# Write config to temp file
TEMP_CONFIG=$(mktemp)
echo "${JOB_CONFIG}" > "${TEMP_CONFIG}"

# Submit job
echo ""
echo "🚀 Submitting ETP-GT training job..."
gcloud ai custom-jobs create \
    --region="${GCP_REGION}" \
    --display-name="${JOB_NAME}" \
    --config="${TEMP_CONFIG}"

# Clean up
rm -f "${TEMP_CONFIG}"

echo "✅ Job submitted successfully"

# Monitor job
echo ""
echo "📊 To monitor the job:"
echo "  gcloud ai custom-jobs list --region=${GCP_REGION} --filter='displayName:${JOB_NAME}'"
echo ""
echo "📋 To view job logs:"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=${GCP_REGION}"
echo ""
echo "🌐 View in console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${GCP_PROJECT_ID}"
echo "=================================================="

