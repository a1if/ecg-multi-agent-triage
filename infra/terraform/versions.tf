terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
  }
  # State is kept locally (a single maintainer). For a team, move it to a GCS bucket:
  #   backend "gcs" { bucket = "<project>-tfstate", prefix = "ecg-triage" }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
