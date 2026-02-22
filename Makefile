.PHONY: help setup fmt lint typecheck test test-cov smoke-test ablate ci-local \
        data baseline train export pipeline mlflow-run onnx-export quality-gate \
        docker-train-build docker-train-push gcp-validate gcp-bootstrap gcp-train \
        docker-infer-build docker-infer-push gcp-deploy gcp-smoke dvc-init lock clean

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
	@echo "  make test-cov           - Run tests with coverage (60% threshold)"
	@echo "  make smoke-test         - Run smoke tests for all model variants"
	@echo "  make ablate             - Run ablation study tests"
	@echo "  make ci-local           - Run full CI locally (lint + typecheck + test)"
	@echo ""
	@echo "Data & Training (Local):"
	@echo "  make data               - Run data pipeline (sessionize, split, build graph)"
	@echo "  make baseline           - Train baseline models locally"
	@echo "  make train              - Train Graph Transformer locally"
	@echo "  make export             - Export model to ONNX format"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make pipeline           - Run full pipeline (validate all models with real data)"
	@echo "  make mlflow-run         - Run training with MLflow tracking"
	@echo "  make onnx-export        - Export model to ONNX format"
	@echo "  make quality-gate       - Run model quality gate (validate before deploy)"
	@echo "  make dvc-init           - Initialize DVC with GCS remote"
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

test-cov:
	.venv/bin/pytest tests/ -v --cov=etpgt --cov-report=term-missing --cov-fail-under=60

smoke-test:
	.venv/bin/python scripts/smoke_test_all_models.py --device cpu

ablate:
	.venv/bin/pytest tests/test_ablations.py -v

ci-local:
	$(MAKE) lint && $(MAKE) typecheck && $(MAKE) test-cov

data:
	.venv/bin/python scripts/data/02_sessionize.py
	.venv/bin/python scripts/data/03_temporal_split.py
	.venv/bin/python scripts/data/04_build_graph.py

baseline:
	.venv/bin/python scripts/train/train_baseline.py --model graphsage

train:
	.venv/bin/python scripts/train/train_baseline.py --model graph_transformer_optimized

export:
	.venv/bin/python scripts/pipeline/export_onnx.py --demo

# ML Pipeline
pipeline:
	.venv/bin/python scripts/pipeline/run_full_pipeline.py --num-sessions 100 --num-epochs 3

mlflow-run:
	.venv/bin/python scripts/pipeline/mlflow_experiment.py --model $(MODEL) --epochs 5

onnx-export:
	.venv/bin/python scripts/pipeline/export_onnx.py --demo

quality-gate:
	.venv/bin/python scripts/pipeline/model_quality_gate.py \
		--checkpoint checkpoints/graph_transformer_optimized_best.pt \
		--config configs/quality_thresholds.yaml

dvc-init:
	.venv/bin/dvc init
	.venv/bin/dvc remote add -d gcs gs://${GCS_BUCKET}/dvc
	@echo "✓ DVC initialized. Run 'dvc repro' to execute the pipeline."

# Dependency lock files
lock:  ## Regenerate dependency lock files
	.venv/bin/pip-compile requirements.txt -o requirements.lock --strip-extras
	.venv/bin/pip-compile requirements-serve.txt -o requirements-serve.lock --strip-extras
	.venv/bin/pip-compile requirements-serve-onnx.txt -o requirements-serve-onnx.lock --strip-extras
	@echo "Lock files regenerated"

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

