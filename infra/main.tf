terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
locals {
  apis = [
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "compute.googleapis.com",
    "run.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)
  service  = each.value

  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# GCS Bucket
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "data" {
  name     = var.bucket_name
  location = var.region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      num_newer_versions = 3
      days_since_noncurrent_time = 30
    }
  }
}

# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = var.ar_repo
  format        = "DOCKER"
  description   = "ETP-GT Docker images for training and inference"

  depends_on = [google_project_service.apis["artifactregistry.googleapis.com"]]
}

# ---------------------------------------------------------------------------
# Service Account
# ---------------------------------------------------------------------------
resource "google_service_account" "etpgt" {
  account_id   = var.sa_name
  display_name = "ETP-GT Training and Serving SA"
  description  = "Service account for Vertex AI training and serving"
}

locals {
  sa_roles = [
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/aiplatform.user",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ]
}

resource "google_project_iam_member" "sa_roles" {
  for_each = toset(local.sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.etpgt.email}"
}
