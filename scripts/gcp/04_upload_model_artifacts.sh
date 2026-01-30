#!/usr/bin/env bash
# Upload model artifacts to GCS for Vertex AI serving
#
# Usage:
#   bash scripts/gcp/04_upload_model_artifacts.sh [CHECKPOINT] [ONNX_MODEL] [EMBEDDINGS]
#
# Example:
#   bash scripts/gcp/04_upload_model_artifacts.sh \
#       checkpoints/graph_transformer_optimized_best.pt \
#       exports/onnx/production/session_recommender.onnx \
#       exports/onnx/production/item_embeddings.npy

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
: "${GCS_BUCKET:?GCS_BUCKET not set. Set in .env or export GCS_BUCKET=your-bucket}"
: "${GCP_PROJECT_ID:?GCP_PROJECT_ID not set. Set in .env or export GCP_PROJECT_ID=your-project}"

# Model version (use timestamp or git commit)
MODEL_VERSION="${MODEL_VERSION:-$(date +%Y%m%d-%H%M%S)}"

# Default paths
CHECKPOINT_PATH="${1:-${PROJECT_ROOT}/checkpoints/graph_transformer_optimized_best.pt}"
ONNX_PATH="${2:-${PROJECT_ROOT}/exports/onnx/production/session_recommender.onnx}"
EMBEDDINGS_PATH="${3:-${PROJECT_ROOT}/exports/onnx/production/item_embeddings.npy}"

# GCS destination
GCS_MODEL_DIR="gs://${GCS_BUCKET}/models/serving/${MODEL_VERSION}"

echo "=================================================="
echo "Uploading Model Artifacts to GCS"
echo "=================================================="
echo "Project:     ${GCP_PROJECT_ID}"
echo "Bucket:      ${GCS_BUCKET}"
echo "Version:     ${MODEL_VERSION}"
echo "Checkpoint:  ${CHECKPOINT_PATH}"
echo "ONNX Model:  ${ONNX_PATH}"
echo "Embeddings:  ${EMBEDDINGS_PATH}"
echo "Destination: ${GCS_MODEL_DIR}"
echo "=================================================="
echo ""

# Verify files exist
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: PyTorch checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# Upload PyTorch checkpoint
echo "Uploading PyTorch model..."
gsutil -m cp "${CHECKPOINT_PATH}" "${GCS_MODEL_DIR}/pytorch/model.pt"
echo "  -> ${GCS_MODEL_DIR}/pytorch/model.pt"

# Upload ONNX model and embeddings if they exist
if [ -f "${ONNX_PATH}" ]; then
    echo ""
    echo "Uploading ONNX model..."
    gsutil -m cp "${ONNX_PATH}" "${GCS_MODEL_DIR}/onnx/session_recommender.onnx"
    echo "  -> ${GCS_MODEL_DIR}/onnx/session_recommender.onnx"
else
    echo ""
    echo "Warning: ONNX model not found: ${ONNX_PATH}"
    echo "Run: python scripts/pipeline/export_onnx.py --production"
fi

if [ -f "${EMBEDDINGS_PATH}" ]; then
    echo ""
    echo "Uploading item embeddings..."
    gsutil -m cp "${EMBEDDINGS_PATH}" "${GCS_MODEL_DIR}/onnx/item_embeddings.npy"
    echo "  -> ${GCS_MODEL_DIR}/onnx/item_embeddings.npy"
else
    echo ""
    echo "Warning: Embeddings not found: ${EMBEDDINGS_PATH}"
fi

# Upload metadata
echo ""
echo "Creating and uploading metadata..."
METADATA_FILE=$(mktemp)
cat > "${METADATA_FILE}" <<EOF
{
    "version": "${MODEL_VERSION}",
    "model_type": "graph_transformer_optimized",
    "metrics": {
        "recall@10": 0.3828,
        "recall@20": 0.4129,
        "ndcg@10": 0.3065
    },
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "source_checkpoint": "$(basename ${CHECKPOINT_PATH})",
    "artifacts": {
        "pytorch": "${GCS_MODEL_DIR}/pytorch/model.pt",
        "onnx": "${GCS_MODEL_DIR}/onnx/session_recommender.onnx",
        "embeddings": "${GCS_MODEL_DIR}/onnx/item_embeddings.npy"
    }
}
EOF
gsutil cp "${METADATA_FILE}" "${GCS_MODEL_DIR}/metadata.json"
rm "${METADATA_FILE}"
echo "  -> ${GCS_MODEL_DIR}/metadata.json"

echo ""
echo "=================================================="
echo "Model artifacts uploaded successfully!"
echo "=================================================="
echo ""
echo "Use these URIs for deployment:"
echo ""
echo "  PyTorch: ${GCS_MODEL_DIR}/pytorch"
echo "  ONNX:    ${GCS_MODEL_DIR}/onnx"
echo ""
echo "Environment variables for deployment:"
echo ""
echo "  export MODEL_VERSION=${MODEL_VERSION}"
echo "  export GCS_PYTORCH_URI=${GCS_MODEL_DIR}/pytorch"
echo "  export GCS_ONNX_URI=${GCS_MODEL_DIR}/onnx"
echo ""
echo "=================================================="
