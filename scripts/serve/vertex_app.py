#!/usr/bin/env python3
"""Vertex AI Endpoint compatible FastAPI server.

This server adapts the recommendation API for Vertex AI custom containers:
- Health check at /health (AIP_HEALTH_ROUTE)
- Predictions at /predict (AIP_PREDICT_ROUTE)
- Loads model from GCS or local path
- Supports both PyTorch and ONNX inference modes
- Includes OpenTelemetry distributed tracing
- Includes Evidently-based drift detection

Usage:
    # Local testing
    uvicorn scripts.serve.vertex_app:app --host 0.0.0.0 --port 8080

    # Environment variables
    MODEL_PATH=/path/to/model.pt          # Local model path
    GCS_MODEL_URI=gs://bucket/models/v1/  # GCS path (downloaded on startup)
    INFERENCE_MODE=pytorch                # "pytorch" or "onnx"
"""

import json
import logging
import os
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter as PromCounter
from prometheus_client import Gauge, Histogram, generate_latest
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenTelemetry tracing setup
# ---------------------------------------------------------------------------


def setup_tracing():
    """Initialize OpenTelemetry with Google Cloud Trace exporter."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

            exporter = CloudTraceSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OpenTelemetry: Google Cloud Trace exporter configured")
        except Exception as e:
            logger.warning("OpenTelemetry: Cloud Trace unavailable (%s), using no-op", e)

        trace.set_tracer_provider(provider)
        return trace.get_tracer("gat-recommendation")
    except ImportError:
        logger.warning("OpenTelemetry not installed, tracing disabled")
        return None


tracer = setup_tracing()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

REQUEST_COUNT = PromCounter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
MODEL_LOADED = Gauge("model_loaded", "Whether the model is loaded (1=yes, 0=no)")
MODEL_ITEMS = Gauge("model_num_items", "Number of items in the loaded model")

# Drift detection metrics
PREDICTION_SCORE_MEAN = Gauge("prediction_score_mean", "Rolling mean of top prediction scores")
PREDICTION_SCORE_STD = Gauge("prediction_score_std", "Rolling std of top prediction scores")
SESSION_LENGTH_MEAN = Gauge("session_length_mean", "Rolling mean session length")
TOP_ITEM_ENTROPY = Gauge("top_item_entropy", "Entropy of top-1 recommended items (diversity)")
DRIFT_DETECTED = Gauge("drift_detected", "Whether distribution drift is detected (1=yes)")

# Configuration from environment
PORT = int(os.getenv("PORT", "8080"))
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "pytorch")  # "pytorch" or "onnx"
GCS_MODEL_URI = os.getenv("GCS_MODEL_URI", "")  # e.g., gs://bucket/models/v1/


# Vertex AI sets AIP_STORAGE_URI to the GCS path of artifact_uri contents
AIP_STORAGE_URI = os.getenv("AIP_STORAGE_URI", "")

# Local model paths (download destination)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pt")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "/app/model/item_embeddings.npy")

# Determine the GCS source for model artifacts
# Priority: AIP_STORAGE_URI (Vertex AI managed) > GCS_MODEL_URI (explicit)
GCS_ARTIFACT_SOURCE = AIP_STORAGE_URI.rstrip("/") if AIP_STORAGE_URI else GCS_MODEL_URI.rstrip("/") if GCS_MODEL_URI else ""


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class VertexInstance(BaseModel):
    """Single prediction instance."""

    session_items: list[int]
    k: int = 10


class VertexPredictRequest(BaseModel):
    """Vertex AI prediction request format."""

    instances: list[dict]  # Each instance: {"session_items": [1,2,3], "k": 10}
    parameters: dict | None = None


class VertexPredictResponse(BaseModel):
    """Vertex AI prediction response format."""

    predictions: list[dict]
    deployedModelId: str | None = None
    model: str | None = None
    modelDisplayName: str | None = None
    modelVersionId: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    inference_mode: str
    num_items: int
    embedding_dim: int


class RecommendRequest(BaseModel):
    """Legacy recommendation request (backward compatibility)."""

    session_items: list[int]
    k: int = 10


class RecommendResponse(BaseModel):
    """Recommendation response."""

    recommendations: list[int]
    scores: list[float]
    latency_ms: float


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------


class ModelState:
    """Holds loaded model and embeddings."""

    def __init__(self):
        self.model = None
        self.ort_session = None
        self.item_embeddings = None
        self.num_items = 0
        self.embedding_dim = 0
        self.loaded = False
        self.model_version = "unknown"


state = ModelState()


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class DriftDetector:
    """Production drift detection using Evidently.

    Collects prediction statistics in a rolling window and compares
    against a reference window (first N requests) to detect distribution
    drift in prediction scores and session lengths.
    """

    def __init__(self, window_size: int = 1000, reference_size: int = 5000):
        self.window_size = window_size
        self.reference_size = reference_size
        self.score_buffer: list[float] = []
        self.session_length_buffer: list[int] = []
        self.top_item_buffer: list[int] = []
        self.reference_scores: list[float] | None = None
        self.reference_session_lengths: list[int] | None = None

    def record(self, session_length: int, top_score: float, top_item: int):
        """Record a single prediction for drift monitoring."""
        self.score_buffer.append(top_score)
        self.session_length_buffer.append(session_length)
        self.top_item_buffer.append(top_item)

        # Set reference window from first N requests
        if self.reference_scores is None and len(self.score_buffer) >= self.reference_size:
            self.reference_scores = list(self.score_buffer)
            self.reference_session_lengths = list(self.session_length_buffer)

        # Keep only recent window + reference
        max_buffer = self.window_size + self.reference_size
        if len(self.score_buffer) > max_buffer:
            self.score_buffer = self.score_buffer[-self.window_size:]
            self.session_length_buffer = self.session_length_buffer[-self.window_size:]
            self.top_item_buffer = self.top_item_buffer[-self.window_size:]

    def check_drift(self) -> dict:
        """Run Evidently drift detection on current vs reference window."""
        if self.reference_scores is None or len(self.score_buffer) < self.window_size:
            return {"drift_detected": False, "reason": "insufficient_data"}

        try:
            import pandas as pd
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            ref_df = pd.DataFrame({
                "score": self.reference_scores[:self.window_size],
                "session_length": self.reference_session_lengths[:self.window_size],
            })
            cur_df = pd.DataFrame({
                "score": self.score_buffer[-self.window_size:],
                "session_length": self.session_length_buffer[-self.window_size:],
            })

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            result = report.as_dict()
            drifted = result["metrics"][0]["result"]["dataset_drift"]
            return {"drift_detected": drifted, "details": result}
        except ImportError:
            logger.warning("Evidently not installed, drift detection unavailable")
            return {"drift_detected": False, "error": "evidently_not_installed"}
        except Exception as e:
            logger.warning("Drift check failed: %s", e)
            return {"drift_detected": False, "error": str(e)}

    def update_metrics(self):
        """Push rolling stats to Prometheus gauges."""
        window = self.window_size

        if self.score_buffer:
            recent_scores = self.score_buffer[-window:]
            PREDICTION_SCORE_MEAN.set(float(np.mean(recent_scores)))
            PREDICTION_SCORE_STD.set(float(np.std(recent_scores)))

        if self.session_length_buffer:
            recent_lengths = self.session_length_buffer[-window:]
            SESSION_LENGTH_MEAN.set(float(np.mean(recent_lengths)))

        if self.top_item_buffer:
            counts = Counter(self.top_item_buffer[-window:])
            total = sum(counts.values())
            entropy = -sum((c / total) * np.log(c / total + 1e-10) for c in counts.values())
            TOP_ITEM_ENTROPY.set(float(entropy))

        drift_result = self.check_drift()
        DRIFT_DETECTED.set(1 if drift_result.get("drift_detected", False) else 0)


drift_detector = DriftDetector(window_size=1000, reference_size=5000)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GAT-Recommendation Vertex AI Endpoint",
    description="Session-based recommendation using Graph Neural Networks",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# GCS download
# ---------------------------------------------------------------------------


def download_from_gcs(gcs_uri: str, local_path: str) -> bool:
    """Download file from GCS.

    Args:
        gcs_uri: GCS URI (gs://bucket/path/to/file)
        local_path: Local destination path

    Returns:
        True if download successful, False otherwise
    """
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return False

    try:
        from google.cloud import storage

        # Parse GCS URI
        path = gcs_uri[5:]  # Remove "gs://"
        bucket_name = path.split("/")[0]
        blob_path = "/".join(path.split("/")[1:])

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info("Downloaded %s to %s", gcs_uri, local_path)
        return True
    except Exception as e:
        logger.error("Failed to download from GCS: %s", e)
        return False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_pytorch_model():
    """Load PyTorch model for inference."""
    import torch

    from etpgt.model import create_graph_transformer_optimized

    logger.info("Loading PyTorch model from %s", MODEL_PATH)

    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # Extract config (may be stored in checkpoint or inferred from state_dict)
    config = checkpoint.get("config", {})
    model_state = checkpoint.get("model_state_dict", checkpoint)

    # Infer num_items from embedding weight shape
    embedding_key = "item_embedding.weight"
    if embedding_key in model_state:
        state.num_items = model_state[embedding_key].shape[0]
        state.embedding_dim = model_state[embedding_key].shape[1]
    else:
        state.num_items = config.get("num_items", 1000)
        state.embedding_dim = config.get("hidden_dim", 256)

    # Create model
    state.model = create_graph_transformer_optimized(
        num_items=state.num_items,
        embedding_dim=state.embedding_dim,
        hidden_dim=state.embedding_dim,
        use_laplacian_pe=False,  # PE not needed for embedding-based inference
    )

    # Filter out cached PE from state dict
    filtered_state = {k: v for k, v in model_state.items() if "_cached_pe" not in k}
    state.model.load_state_dict(filtered_state, strict=False)
    state.model.eval()

    # Cache item embeddings for fast inference
    with torch.no_grad():
        state.item_embeddings = state.model.get_item_embeddings().numpy()

    state.loaded = True
    state.model_version = str(checkpoint.get("epoch", "unknown"))

    logger.info("PyTorch model loaded: %d items, %dd embeddings", state.num_items, state.embedding_dim)


def load_onnx_model():
    """Load ONNX model for inference."""
    import onnxruntime as ort

    logger.info("Loading ONNX model from %s", MODEL_PATH)

    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"ONNX model not found: {MODEL_PATH}")

    # Load ONNX session
    state.ort_session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    # Load embeddings
    if Path(EMBEDDINGS_PATH).exists():
        state.item_embeddings = np.load(EMBEDDINGS_PATH)
    else:
        raise RuntimeError(f"Embeddings not found: {EMBEDDINGS_PATH}")

    state.num_items = state.item_embeddings.shape[0]
    state.embedding_dim = state.item_embeddings.shape[1]
    state.loaded = True

    # Load metadata if available
    metadata_path = Path(MODEL_PATH).parent / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            state.model_version = metadata.get("version", "unknown")

    logger.info("ONNX model loaded: %d items, %dd embeddings", state.num_items, state.embedding_dim)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    logger.info("[STARTUP] Starting Vertex AI endpoint (mode: %s)", INFERENCE_MODE)
    logger.info("[STARTUP] AIP_STORAGE_URI: %s", AIP_STORAGE_URI or "not set")
    logger.info("[STARTUP] GCS_ARTIFACT_SOURCE: %s", GCS_ARTIFACT_SOURCE or "not set")
    logger.info("[STARTUP] MODEL_PATH: %s", MODEL_PATH)
    logger.info("[STARTUP] EMBEDDINGS_PATH: %s", EMBEDDINGS_PATH)

    # List local model directory for debugging
    model_dir = os.path.dirname(MODEL_PATH)
    if os.path.isdir(model_dir):
        logger.info("[STARTUP] Files in %s: %s", model_dir, os.listdir(model_dir))

    try:
        # Download model files from GCS if not present locally
        if not Path(MODEL_PATH).exists() and GCS_ARTIFACT_SOURCE:
            logger.info("[STARTUP] Model not at %s, downloading from %s...", MODEL_PATH, GCS_ARTIFACT_SOURCE)
            if INFERENCE_MODE == "onnx":
                download_from_gcs(f"{GCS_ARTIFACT_SOURCE}/session_recommender.onnx", MODEL_PATH)
                download_from_gcs(f"{GCS_ARTIFACT_SOURCE}/item_embeddings.npy", EMBEDDINGS_PATH)
            else:
                download_from_gcs(f"{GCS_ARTIFACT_SOURCE}/model.pt", MODEL_PATH)

        if INFERENCE_MODE == "onnx":
            load_onnx_model()
        else:
            load_pytorch_model()
        logger.info("[STARTUP] Model loaded successfully! state.loaded=%s", state.loaded)
        MODEL_LOADED.set(1)
        MODEL_ITEMS.set(state.num_items)
    except Exception as e:
        logger.error("[STARTUP] Failed to load model: %s\n%s", e, traceback.format_exc())
        # Don't raise - let health check report not ready (503)

    # Initialize OpenTelemetry FastAPI auto-instrumentation
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry: FastAPI auto-instrumentation enabled")
    except ImportError:
        logger.warning("OpenTelemetry: FastAPI instrumentation not available")
    except Exception as e:
        logger.warning("OpenTelemetry: FastAPI instrumentation failed: %s", e)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (required by Vertex AI).

    Returns 200 when model is loaded and ready, 503 otherwise.
    """
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet",
        )
    return HealthResponse(
        status="healthy",
        model_loaded=state.loaded,
        inference_mode=INFERENCE_MODE,
        num_items=state.num_items,
        embedding_dim=state.embedding_dim,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4")


@app.get("/drift")
async def drift_status():
    """Drift detection status endpoint.

    Returns current drift detection results from Evidently analysis.
    Updates Prometheus drift metrics as a side effect.
    """
    drift_detector.update_metrics()
    return drift_detector.check_drift()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def compute_recommendations(session_items: list[int], k: int = 10) -> dict:
    """Compute recommendations for a session.

    Args:
        session_items: List of item IDs in the session
        k: Number of recommendations to return

    Returns:
        Dictionary with recommendations, scores, and latency
    """
    start_time = time.time()

    # Filter valid items
    valid_items = [i for i in session_items if 0 <= i < state.num_items]
    if not valid_items:
        raise ValueError("No valid item IDs in session")

    # Optionally wrap in OpenTelemetry span
    span = None
    if tracer is not None:
        from opentelemetry import trace as otel_trace

        span = tracer.start_span("compute_recommendations")
        span.set_attribute("session.length", len(session_items))
        span.set_attribute("session.valid_items", len(valid_items))
        span.set_attribute("recommendation.k", k)

    try:
        # Compute session embedding (mean of item embeddings)
        session_embedding = state.item_embeddings[valid_items].mean(axis=0, keepdims=True)

        # Normalize for cosine similarity
        session_norm = session_embedding / (
            np.linalg.norm(session_embedding, axis=-1, keepdims=True) + 1e-8
        )
        item_norm = state.item_embeddings / (
            np.linalg.norm(state.item_embeddings, axis=-1, keepdims=True) + 1e-8
        )

        # Compute cosine similarity scores
        scores = np.matmul(session_norm, item_norm.T).squeeze(0)

        # Exclude items already in session
        for item_id in valid_items:
            scores[item_id] = float("-inf")

        # Get top-k recommendations
        k = min(k, state.num_items - len(valid_items))
        top_indices = np.argsort(scores)[-k:][::-1]
        top_scores = scores[top_indices]

        latency_ms = (time.time() - start_time) * 1000

        # Record for drift detection
        drift_detector.record(
            session_length=len(valid_items),
            top_score=float(top_scores[0]),
            top_item=int(top_indices[0]),
        )

        if span is not None:
            span.set_attribute("recommendation.latency_ms", latency_ms)

        return {
            "recommendations": top_indices.tolist(),
            "scores": top_scores.tolist(),
            "latency_ms": round(latency_ms, 3),
        }
    finally:
        if span is not None:
            span.end()


@app.post("/predict", response_model=VertexPredictResponse)
async def predict(request: VertexPredictRequest):
    """Vertex AI prediction endpoint.

    Accepts Vertex AI standard prediction format and returns recommendations.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    predictions = []
    for instance in request.instances:
        session_items = instance.get("session_items", [])
        k = instance.get("k", 10)

        try:
            result = compute_recommendations(session_items, k)
            result["request_id"] = request_id
            predictions.append(result)
        except ValueError as e:
            predictions.append({"error": str(e), "request_id": request_id})
        except Exception as e:
            predictions.append({"error": f"Inference failed: {str(e)}", "request_id": request_id})

    latency = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
    REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()

    return VertexPredictResponse(
        predictions=predictions,
        model=f"gat-recommendation-{INFERENCE_MODE}",
        modelVersionId=state.model_version,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Legacy recommendation endpoint (backward compatibility with local dev).

    Simpler format for direct API usage.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        result = compute_recommendations(request.session_items, request.k)
        latency = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/recommend").observe(latency)
        REQUEST_COUNT.labels(endpoint="/recommend", status="success").inc()
        return RecommendResponse(
            recommendations=result["recommendations"],
            scores=result["scores"],
            latency_ms=result["latency_ms"],
        )
    except ValueError as e:
        REQUEST_COUNT.labels(endpoint="/recommend", status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend/batch")
async def recommend_batch(requests: list[RecommendRequest]):
    """Batch recommendation endpoint.

    Process multiple sessions in a single request.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    results = []
    for req in requests:
        try:
            result = compute_recommendations(req.session_items, req.k)
            results.append(
                RecommendResponse(
                    recommendations=result["recommendations"],
                    scores=result["scores"],
                    latency_ms=result["latency_ms"],
                )
            )
        except ValueError as e:
            results.append({"error": str(e)})

    latency = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/recommend/batch").observe(latency)
    REQUEST_COUNT.labels(endpoint="/recommend/batch", status="success").inc()

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
