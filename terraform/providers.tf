# Terraform Provider Configuration
#
# The backend "gcs" block below is COMMENTED OUT so a fresh clone can
# `terraform validate` and `terraform fmt -check` without errors. The
# get_started_linux.sh script rewrites this file in place with a real
# backend "gcs" block once it knows the GCP project ID, populating:
#
#   backend "gcs" {
#     bucket = "<PROJECT_ID>-tfstate"
#     prefix = "agent/state"
#   }
#
# The state bucket itself is created by get_started_linux.sh via gcloud
# (it must exist before `terraform init` because the backend uses it).
#
# If you skipped get_started_linux.sh and are wiring the backend by hand:
#
#   1. Create the state bucket:
#        gcloud storage buckets create gs://YOUR_PROJECT_ID-tfstate \
#          --project=YOUR_PROJECT_ID \
#          --location=us-central1 \
#          --uniform-bucket-level-access \
#          --public-access-prevention
#        gcloud storage buckets update gs://YOUR_PROJECT_ID-tfstate --versioning
#
#   2. Uncomment the backend block below and replace YOUR_PROJECT_ID.
#
#   3. Run: terraform init

terraform {
  required_version = ">= 1.2"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "mike-the-mechanic-prod-tfstate"
    prefix = "agent/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
