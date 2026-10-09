# ECG multi-agent triage on Google Cloud.
#
#   browser ──> frontend (public, Cloud Run) ──ID token──> api (private, Cloud Run) ──ID token──> inference (private,
#                                                                                                   Cloud Run + L4 GPU,
#                                                                                                   only if enable_gpu)
# Images live in Artifact Registry; GitHub Actions deploys through Workload Identity Federation (no stored keys).
# Terraform owns the infrastructure; the deploy workflow owns which image runs (images are ignored here after creation).

locals {
  apis = [
    "run.googleapis.com", "artifactregistry.googleapis.com", "iam.googleapis.com", "iamcredentials.googleapis.com",
    "sts.googleapis.com", "storage.googleapis.com", "cloudresourcemanager.googleapis.com",
  ]
  labels = { app = "ecg-triage", managed-by = "terraform" }
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

# ---------------- images ----------------
resource "google_artifact_registry_repository" "images" {
  repository_id = "ecg-triage"
  location      = var.region
  format        = "DOCKER"
  description   = "ECG triage container images"
  labels        = local.labels

  # Storage beyond the 0.5 GB free tier is billed: keep the 3 newest versions of each image, delete the rest.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-newest-3"
    action = "KEEP"
    most_recent_versions { keep_count = 3 }
  }
  cleanup_policies {
    id     = "delete-older"
    action = "DELETE"
    condition { older_than = "604800s" } # 7 days
  }
  depends_on = [google_project_service.apis]
}

# ---------------- identities: one per service, least privilege ----------------
resource "google_service_account" "frontend" {
  account_id   = "ecg-frontend"
  display_name = "ECG triage frontend (Cloud Run)"
}

resource "google_service_account" "api" {
  account_id   = "ecg-api"
  display_name = "ECG triage API (Cloud Run)"
}

resource "google_service_account" "inference" {
  account_id   = "ecg-inference"
  display_name = "ECG triage GPU inference (Cloud Run)"
}

resource "google_service_account" "deployer" {
  account_id   = "ecg-deployer"
  display_name = "GitHub Actions deployer (Workload Identity Federation)"
}

# ---------------- API (private: only the frontend may call it) ----------------
resource "google_cloud_run_v2_service" "api" {
  name                = "ecg-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false
  labels              = local.labels

  template {
    service_account                  = google_service_account.api.email
    max_instance_request_concurrency = 40
    timeout                          = "300s"
    scaling {
      min_instance_count = 0 # scale to zero when idle
      max_instance_count = 1 # one instance: the run store is in memory (backend/src/ecg_agent/api/runs.py)
    }
    containers {
      image = var.placeholder_image
      ports { container_port = 8080 }
      resources {
        limits = { cpu = "1", memory = "2Gi" }
        # The agents keep working after POST /v1/runs has answered 202, so the CPU must stay allocated between
        # requests (instance-based billing). The instance still scales to zero when idle.
        cpu_idle          = false
        startup_cpu_boost = true
      }
      env {
        name  = "RECEIVER"
        value = var.enable_gpu ? "http" : "replay"
      }
      env {
        name  = "INFERENCE_URL"
        value = var.enable_gpu ? google_cloud_run_v2_service.inference[0].uri : ""
      }
      env {
        name  = "INFERENCE_AUTH"
        value = "gcp"
      }
      env {
        name  = "DEMO_PRESETS"
        value = var.enable_gpu ? "0" : "1"
      }
      env {
        name  = "CLASSIFIER"
        value = "gemma" # with recorded answers: recorded questions route as Gemma did, others by keywords
      }
      env {
        name  = "ALLOW_UPLOADS"
        value = "0" # public demo: no health data from strangers (docs/design.md §2)
      }
      env {
        name  = "RUNS_PER_MINUTE"
        value = "6" # all visitors reach the API through the frontend, so this is a global limit
      }
      startup_probe {
        tcp_socket { port = 8080 }
        period_seconds    = 5
        failure_threshold = 24
      }
    }
  }
  lifecycle {
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "frontend_invokes_api" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.frontend.email}"
}

# ---------------- frontend (public) ----------------
resource "google_cloud_run_v2_service" "frontend" {
  name                = "ecg-frontend"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false
  labels              = local.labels

  template {
    service_account                  = google_service_account.frontend.email
    session_affinity                 = true    # Streamlit keeps each browser session on one instance
    timeout                          = "3600s" # Streamlit holds a websocket per browser tab
    max_instance_request_concurrency = 40
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = var.placeholder_image
      ports { container_port = 8080 }
      resources {
        limits            = { cpu = "1", memory = "1Gi" }
        cpu_idle          = true # request-based billing: CPU only while serving; the free tier covers a demo
        startup_cpu_boost = true
      }
      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api.uri
      }
      env {
        name  = "API_AUTH"
        value = "gcp"
      }
      startup_probe {
        tcp_socket { port = 8080 }
        period_seconds    = 5
        failure_threshold = 12
      }
    }
  }
  lifecycle {
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "public_frontend" {
  name     = google_cloud_run_v2_service.frontend.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------- GPU inference (only if enable_gpu; private: only the API may call it) ----------------
resource "google_storage_bucket" "models" {
  count                       = var.enable_gpu ? 1 : 0
  name                        = "${var.project_id}-ecg-models"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  labels                      = local.labels
}

resource "google_storage_bucket_iam_member" "inference_reads_models" {
  count  = var.enable_gpu ? 1 : 0
  bucket = google_storage_bucket.models[0].name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_cloud_run_v2_service" "inference" {
  count               = var.enable_gpu ? 1 : 0
  provider            = google
  name                = "ecg-inference"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false
  labels              = local.labels

  template {
    service_account                  = google_service_account.inference.email
    gpu_zonal_redundancy_disabled    = true # cheaper; new projects get L4 quota automatically in this mode
    max_instance_request_concurrency = 4
    timeout                          = "600s"
    scaling {
      min_instance_count = 0 # GPUs bill for the whole instance lifetime: never keep one waiting
      max_instance_count = 1 # one GPU at most, whatever the traffic
    }
    node_selector { accelerator = "nvidia-l4" }
    containers {
      image = var.placeholder_image
      ports { container_port = 8080 }
      resources {
        limits            = { cpu = "4", memory = "16Gi", "nvidia.com/gpu" = "1" }
        cpu_idle          = false # required with GPUs
        startup_cpu_boost = true
      }
      env {
        name  = "HF_HOME"
        value = "/models/hf"
      }
      volume_mounts {
        name       = "models"
        mount_path = "/models/hf"
      }
      startup_probe {
        tcp_socket { port = 8080 }
        period_seconds    = 10
        failure_threshold = 60 # loading Gemma: allow up to 10 minutes
      }
    }
    volumes {
      name = "models"
      gcs {
        bucket    = google_storage_bucket.models[0].name
        read_only = false
      }
    }
  }
  lifecycle {
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "api_invokes_inference" {
  count    = var.enable_gpu ? 1 : 0
  name     = google_cloud_run_v2_service.inference[0].name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.api.email}"
}

# ---------------- GitHub Actions: deploy without keys ----------------
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  # Only this repository's main branch can get credentials at all.
  attribute_condition = "assertion.repository == '${var.github_repository}' && assertion.ref == 'refs/heads/main'"
  oidc { issuer_uri = "https://token.actions.githubusercontent.com" }
}

resource "google_service_account_iam_member" "github_impersonates_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# What the deployer may do: push images, roll out new revisions, run them as the runtime identities. Nothing else.
resource "google_artifact_registry_repository_iam_member" "deployer_pushes" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "deployer_deploys" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_acts_as_runtime" {
  for_each           = { frontend = google_service_account.frontend.name, api = google_service_account.api.name, inference = google_service_account.inference.name }
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}
