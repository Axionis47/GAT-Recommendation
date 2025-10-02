#!/usr/bin/env bash
# Submit Vertex AI training job

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
MODEL_TYPE="${1:-graphsage}"
JOB_NAME="${2:-etpgt-${MODEL_TYPE}-$(date +%Y%m%d-%H%M%S)}"

# Configuration
IMAGE_NAME="etpgt-train"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# Training arguments
EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
READOUT_TYPE="${READOUT_TYPE:-mean}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
LR="${LR:-0.001}"
PATIENCE="${PATIENCE:-10}"

echo "=================================================="
echo "Submitting Vertex AI Training Job"
echo "=================================================="
echo "Project: ${GCP_PROJECT_ID}"
echo "Region: ${GCP_REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Model: ${MODEL_TYPE}"
echo "Image: ${IMAGE_URI}"
echo "Machine: ${VAI_MACHINE}"
echo "Accelerator: ${VAI_ACCELERATOR} x ${VAI_ACCELERATOR_COUNT}"
echo "=================================================="

# Create job config YAML
# Docker ENTRYPOINT is "python", so we need to pass the script path + args
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
        - scripts/train/train_baseline.py
        - --model=${MODEL_TYPE}
        - --embedding-dim=${EMBEDDING_DIM}
        - --hidden-dim=${HIDDEN_DIM}
        - --num-layers=${NUM_LAYERS}
        - --num-heads=${NUM_HEADS}
        - --dropout=${DROPOUT}
        - --readout-type=${READOUT_TYPE}
        - --batch-size=${BATCH_SIZE}
        - --max-epochs=${MAX_EPOCHS}
        - --lr=${LR}
        - --patience=${PATIENCE}
        - --gcs-bucket=${GCS_BUCKET}
        - --device=cuda
scheduling:
  timeout: 604800s
  restartJobOnWorkerRestart: false
EOF
)

# Write config to temp file
TEMP_CONFIG=$(mktemp)
echo "${JOB_CONFIG}" > "${TEMP_CONFIG}"

# Submit job
echo ""
echo "🚀 Submitting training job..."
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

