.PHONY: help setup fmt lint typecheck test data baseline train ablate export serve-local \
        docker-train-build docker-train-push gcp-validate gcp-bootstrap gcp-train \
        docker-infer-build docker-infer-push gcp-deploy gcp-smoke clean

# Load environment variables (optional, ignore if not present)
-include .env
export

help:
	@echo "ETP-GT Makefile"
	@echo ""
	@echo "Setup & Development:"
	@echo "  make setup              - Install dependencies and setup environment"
	@echo "  make fmt                - Format code with black and isort"
	@echo "  make lint               - Lint code with ruff"
	@echo "  make typecheck          - Type check with mypy"
	@echo "  make test               - Run tests with pytest"
	@echo ""
	@echo "Data & Training (Local):"
	@echo "  make data               - Prepare RetailRocket dataset"
	@echo "  make baseline           - Train baseline models locally"
	@echo "  make train              - Train ETP-GT locally"
	@echo "  make ablate             - Run ablation studies"
	@echo "  make export             - Export embeddings"
	@echo "  make serve-local        - Run inference API locally"
	@echo ""
	@echo "Docker & GCP:"
	@echo "  make docker-train-build - Build training Docker image"
	@echo "  make docker-train-push  - Push training image to Artifact Registry"
	@echo "  make gcp-validate       - Validate GCP environment and prerequisites"
	@echo "  make gcp-bootstrap      - Bootstrap GCP infrastructure"
	@echo "  make gcp-train          - Submit training job to Vertex AI"
	@echo "  make docker-infer-build - Build inference Docker image"
	@echo "  make docker-infer-push  - Push inference image to Artifact Registry"
	@echo "  make gcp-deploy         - Deploy API to Cloud Run"
	@echo "  make gcp-smoke          - Run smoke tests against Cloud Run"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean              - Clean temporary files and caches"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel
	.venv/bin/pip install -r requirements.txt
	@echo "✓ Setup complete. Activate with: source .venv/bin/activate"

fmt:
	.venv/bin/black etpgt/ tests/ scripts/
	.venv/bin/isort etpgt/ tests/ scripts/

lint:
	.venv/bin/ruff check etpgt/ tests/ scripts/

typecheck:
	.venv/bin/mypy etpgt/

test:
	.venv/bin/pytest tests/ -v

data:
	.venv/bin/python scripts/prep_retailrocket.py

baseline:
	.venv/bin/python -m etpgt.cli.train --config configs/yoochoose_baselines.yaml

train:
	.venv/bin/python -m etpgt.cli.train --config configs/yoochoose_etpgt_small.yaml

ablate:
	.venv/bin/python -m etpgt.cli.ablate --config configs/ablations.yaml

export:
	.venv/bin/python scripts/export_embeddings.py

serve-local:
	.venv/bin/uvicorn etpgt.serve.rerank_api:app --host 0.0.0.0 --port 8000 --reload

# Docker - Training
docker-train-build:
	bash scripts/gcp/02_build_training_image.sh

docker-train-push: docker-train-build
	@echo "✓ Image already pushed by build script"

# GCP Infrastructure
gcp-validate:
	bash scripts/gcp/00_validate_env.sh

gcp-bootstrap:
	bash scripts/gcp/01_bootstrap.sh

gcp-train:
	bash scripts/gcp/03_submit_training_job.sh $(MODEL) $(JOB_NAME)

# Docker - Inference
docker-infer-build:
	docker build -f docker/Dockerfile.infer -t ${AR_REPO}/infer:latest .

docker-infer-push: docker-infer-build
	docker push ${AR_REPO}/infer:latest

# GCP Deployment
gcp-deploy:
	bash scripts/gcp/05_deploy_cloud_run.sh

gcp-smoke:
	bash scripts/gcp/06_smoketest_cloud_run.sh

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "✓ Cleaned temporary files"

