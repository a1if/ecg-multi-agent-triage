output "frontend_url" {
  description = "The public demo"
  value       = google_cloud_run_v2_service.frontend.uri
}

output "api_url" {
  description = "Private: callable only by the frontend's service account"
  value       = google_cloud_run_v2_service.api.uri
}

output "inference_url" {
  value = var.enable_gpu ? google_cloud_run_v2_service.inference[0].uri : "(GPU service not created: enable_gpu = false)"
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

# Set these as GitHub repository variables (not secrets: they are identifiers, not credentials).
output "github_variables" {
  value = {
    GCP_PROJECT_ID   = var.project_id
    GCP_REGION       = var.region
    GCP_WIF_PROVIDER = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOYER_SA  = google_service_account.deployer.email
  }
}
