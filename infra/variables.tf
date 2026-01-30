variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "plotpointe"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "GCS bucket for model artifacts and data"
  type        = string
  default     = "plotpointe-etpgt-data"
}

variable "ar_repo" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "etpgt"
}

variable "sa_name" {
  description = "Service account ID"
  type        = string
  default     = "etpgt-sa"
}
