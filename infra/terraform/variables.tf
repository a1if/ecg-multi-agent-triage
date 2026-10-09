variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Region for everything. us-central1: Cloud Run GPUs are offered there and Cloud Storage's free tier applies."
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "owner/name of the GitHub repository allowed to deploy (Workload Identity Federation)"
  type        = string
  default     = "a1if/ecg-multi-agent-triage"
}

variable "enable_gpu" {
  description = "Create the GPU inference service (Gemma on an NVIDIA L4). Off: the public demo serves recorded Gemma answers on CPU only."
  type        = bool
  default     = false
}

variable "placeholder_image" {
  description = "Image used when a service is first created; the deploy workflow replaces it (Terraform then ignores the image)."
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}
