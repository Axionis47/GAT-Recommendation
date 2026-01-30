output "bucket_url" {
  description = "GCS bucket URL"
  value       = "gs://${google_storage_bucket.data.name}"
}

output "ar_repo_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.etpgt.email
}
