#!/usr/bin/env bash
# Submit all training jobs to Vertex AI

set -euo pipefail

echo "=== Submitting all training jobs to Vertex AI ==="
echo ""

# Submit GraphSAGE
echo "1/3: Submitting GraphSAGE..."
bash scripts/gcp/03_submit_training_job.sh graphsage
echo ""
sleep 5

# Submit GraphTransformer
echo "2/3: Submitting GraphTransformer..."
bash scripts/gcp/03_submit_training_job.sh graph_transformer
echo ""
sleep 5

# Submit ETP-GT
echo "3/3: Submitting ETP-GT..."
bash scripts/gcp/04_submit_etpgt_job.sh
echo ""

echo "=== All jobs submitted! ==="

