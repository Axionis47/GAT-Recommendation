#!/usr/bin/env bash
# Submit all training jobs to Vertex AI

set -euo pipefail

echo "=== Submitting all training jobs to Vertex AI ==="
echo ""

# Submit GraphSAGE
echo "1/2: Submitting GraphSAGE..."
bash scripts/gcp/03_submit_training_job.sh graphsage
echo ""
sleep 5

# Submit GraphTransformer
echo "2/2: Submitting GraphTransformer..."
bash scripts/gcp/03_submit_training_job.sh graph_transformer
echo ""

echo "=== All jobs submitted! ==="
