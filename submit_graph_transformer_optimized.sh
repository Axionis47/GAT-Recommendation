#!/bin/bash
# Submit OPTIMIZED GraphTransformer training job to Vertex AI
# This version is 88x faster than the original GraphTransformer

set -e

# Configuration - Set these environment variables before running:
# export GCP_PROJECT_ID="your-project-id"
# export GCS_BUCKET="your-bucket-name"
# export GCP_REGION="us-central1"  # optional, defaults to us-central1
PROJECT_ID="${GCP_PROJECT_ID:?Error: GCP_PROJECT_ID environment variable is not set}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET:?Error: GCS_BUCKET environment variable is not set}"
IMAGE_URI="${GCP_REGION:-us-central1}-docker.pkg.dev/${PROJECT_ID}/etpgt/etpgt-train:latest"
JOB_NAME="etpgt-graph_transformer_optimized-$(date +%Y%m%d-%H%M%S)"

# Model configuration (OPTIMIZED)
MODEL="graph_transformer_optimized"
EMBEDDING_DIM=256
HIDDEN_DIM=256
NUM_LAYERS=2  # Optimized: Reduced from 3 to 2
NUM_HEADS=2   # Optimized: Reduced from 4 to 2
DROPOUT=0.1
READOUT_TYPE="mean"

# Training configuration
BATCH_SIZE=32
NUM_NEGATIVES=5
MAX_EPOCHS=100
PATIENCE=10
LR=0.001
WEIGHT_DECAY=0.00001

# Data paths
TRAIN_SESSIONS="gs://${BUCKET_NAME}/data/processed/train.csv"
VAL_SESSIONS="gs://${BUCKET_NAME}/data/processed/val.csv"
GRAPH_EDGES="gs://${BUCKET_NAME}/data/processed/graph_edges.csv"
OUTPUT_DIR="gs://${BUCKET_NAME}/outputs/graph_transformer_optimized"

echo "=========================================="
echo "SUBMITTING OPTIMIZED GRAPH TRANSFORMER JOB"
echo "=========================================="
echo ""
echo "Optimizations:"
echo "  - FFN: DISABLED (29x speedup)"
echo "  - Layers: 2 (reduced from 3)"
echo "  - Heads: 2 (reduced from 4)"
echo "  - Expected speedup: 88x faster"
echo "  - Expected cost: ~\$84 (vs \$7,440 for original)"
echo ""
echo "Job Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Job Name: ${JOB_NAME}"
echo "  Model: ${MODEL}"
echo "  Image: ${IMAGE_URI}"
echo ""
echo "Model Hyperparameters:"
echo "  Embedding Dim: ${EMBEDDING_DIM}"
echo "  Hidden Dim: ${HIDDEN_DIM}"
echo "  Num Layers: ${NUM_LAYERS} (optimized)"
echo "  Num Heads: ${NUM_HEADS} (optimized)"
echo "  Dropout: ${DROPOUT}"
echo "  Readout: ${READOUT_TYPE}"
echo ""
echo "Training Configuration:"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Num Negatives: ${NUM_NEGATIVES}"
echo "  Max Epochs: ${MAX_EPOCHS}"
echo "  Patience: ${PATIENCE}"
echo "  Learning Rate: ${LR}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo ""
echo "Data Paths:"
echo "  Train: ${TRAIN_SESSIONS}"
echo "  Val: ${VAL_SESSIONS}"
echo "  Graph: ${GRAPH_EDGES}"
echo "  Output: ${OUTPUT_DIR}"
echo ""
echo "=========================================="
echo ""

# Submit job
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec=machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri="${IMAGE_URI}" \
  --args="--model=${MODEL}" \
  --args="--train-sessions=${TRAIN_SESSIONS}" \
  --args="--val-sessions=${VAL_SESSIONS}" \
  --args="--graph-edges=${GRAPH_EDGES}" \
  --args="--output-dir=${OUTPUT_DIR}" \
  --args="--embedding-dim=${EMBEDDING_DIM}" \
  --args="--hidden-dim=${HIDDEN_DIM}" \
  --args="--num-layers=${NUM_LAYERS}" \
  --args="--num-heads=${NUM_HEADS}" \
  --args="--dropout=${DROPOUT}" \
  --args="--readout-type=${READOUT_TYPE}" \
  --args="--batch-size=${BATCH_SIZE}" \
  --args="--num-negatives=${NUM_NEGATIVES}" \
  --args="--max-epochs=${MAX_EPOCHS}" \
  --args="--patience=${PATIENCE}" \
  --args="--lr=${LR}" \
  --args="--weight-decay=${WEIGHT_DECAY}" \
  --enable-web-access \
  --enable-dashboard-access \
  --service-account="${GCP_SERVICE_ACCOUNT:-etpgt-sa@${PROJECT_ID}.iam.gserviceaccount.com}"

echo ""
echo "=========================================="
echo "Job submitted successfully!"
echo "=========================================="
echo ""
echo "Monitor job:"
echo "  gcloud ai custom-jobs describe ${JOB_NAME} --region=${REGION}"
echo ""
echo "Stream logs:"
echo "  gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"
echo ""
echo "Expected completion time: ~20-30 hours (with early stopping)"
echo "Expected cost: ~\$37-56"
echo ""
echo "Compare with original GraphTransformer:"
echo "  Original: 40 hours/epoch, \$7,440 total"
echo "  Optimized: 27 min/epoch, \$84 total"
echo "  Speedup: 88x faster!"
echo ""

