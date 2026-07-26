# Agent Infrastructure — Terraform
#
# This file manages all GCP resources for the agent EXCEPT the GCP project
# itself and the terraform state bucket. Both are bootstrapped via gcloud
# in get_started_linux.sh (the project because terraform managing its own
# project is fragile; the state bucket because the backend has to exist
# before `terraform init`).
#
# STRUCTURE:
#   Section 1: Common infrastructure (all agents)
#   Section 2: Slack-specific infrastructure (uncomment if using Slack)
#   Section 3: Google Chat-specific infrastructure
#   Section 4: Telegram-specific infrastructure
#   Section 5: Discord-specific infrastructure
#   Section 6: Scheduler MCP key (uncomment to use The Forum's scheduler MCP)
#
# Sections 2-6 are commented out by default. get_started_linux.sh uncomments
# whichever ones the operator selects. You can also uncomment them manually.
#
# Hard rule: never add cloud resources to this agent outside terraform.
# Modify this file, `terraform apply`, and commit. See AGENTS.md.

# ==============================================================================
# SECTION 1: COMMON INFRASTRUCTURE (Required for all agents)
# ==============================================================================

# The project is pre-existing — created via `gcloud projects create` in
# get_started_linux.sh. We reference it as a data source so we never own
# its lifecycle.
data "google_project" "agent_project" {
  project_id = var.project_id
}

# The Forum's project, looked up so we can resolve its default compute SA
# (the principal The Forum runs Cloud Run as) for cross-project IAM
# bindings on this agent's platform secrets.
data "google_project" "forum" {
  project_id = var.forum_project_id
}

locals {
  # The Forum's Cloud Run runs as the project's default compute SA.
  forum_runtime_sa = "${data.google_project.forum.number}-compute@developer.gserviceaccount.com"

  # Three Vertex AI service agents in the Forum project participate in
  # the agent engine lifecycle at different phases. All three need
  # iam.serviceAccountTokenCreator on the per-agent SA — missing any one
  # produces a different startup failure:
  #   - gcp-sa-aiplatform     — control-plane operations on the engine
  #                             resource (create / delete / update)
  #   - gcp-sa-aiplatform-re  — Reasoning Engine resource management;
  #                             ALSO the principal that reads the staging
  #                             bucket at cold-start to fetch the packaged
  #                             agent code (see engine_staging_reader
  #                             below — it's bound to -re only, not the
  #                             set, since -re is the only one that
  #                             actually touches GCS).
  #   - gcp-sa-aiplatform-cc  — runs the workload CONTAINER and emulates
  #                             the metadata server inside it. When this
  #                             binding is missing, the symptom is a 500
  #                             from /computeMetadata/v1/instance/
  #                             service-accounts/default/token at engine
  #                             startup with the misleading message
  #                             "Compute Engine Metadata server unavailable".
  #
  # All three are auto-provisioned by `gcloud beta services identity create
  # --service=aiplatform.googleapis.com` (one call provisions all sub-agents),
  # which the bootstrap runs in phase 5.
  forum_vertex_ai_service_agents = toset([
    "service-${data.google_project.forum.number}@gcp-sa-aiplatform.iam.gserviceaccount.com",
    "service-${data.google_project.forum.number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com",
    "service-${data.google_project.forum.number}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com",
  ])
}

# --- APIs ---
# Bootstrap APIs (serviceusage, cloudresourcemanager, secretmanager,
# and iam.googleapis.com) are already enabled by get_started_linux.sh —
# terraform would chicken-and-egg if it tried to enable them itself
# (e.g., google_service_account.agent below can't be created without
# iam.googleapis.com, and terraform's own auto-enable on first SA
# creation isn't reliable for IAM-related ops).
#
# The APIs below are the rest — declared as resources here so they're
# tracked in state and so resources that need them can depends_on them.

resource "google_project_service" "secretmanager" {
  project            = var.project_id
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

# IAM API — required by `google_service_account` and any other resource
# that talks to the IAM control plane. Listed explicitly so the SA's
# depends_on can reference it; without it, terraform sometimes races
# the first SA creation against Google's auto-enable and the apply
# fails with a confusing "service not enabled" error on the SA itself.
resource "google_project_service" "iam" {
  project            = var.project_id
  service            = "iam.googleapis.com"
  disable_on_destroy = false
}

# IAM Credentials API — required to mint short-lived tokens for service
# accounts, including the cross-project SA impersonation that Vertex
# AI's service agents do to run our per-agent SA as the engine's
# runtime identity. (Three service agents are involved at different
# phases — see the `forum_vertex_ai_service_agents` local above.)
# Without this enabled, terraform's `engine_token_creator` bindings
# succeed but the actual token-minting calls at runtime fail with a
# "Resource not found" / "API not enabled" error — manifesting as a
# Reasoning Engine that deploys cleanly and then 500s on its first
# invocation.
resource "google_project_service" "iamcredentials" {
  project            = var.project_id
  service            = "iamcredentials.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "aiplatform" {
  project            = var.project_id
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  project            = var.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# Google Workspace APIs — enabled by default so the agent can read/write
# Sheets, Docs, and Drive once the operator shares a file with the agent SA.
# If your agent doesn't need any of these, you can comment them out.
resource "google_project_service" "drive" {
  project            = var.project_id
  service            = "drive.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sheets" {
  project            = var.project_id
  service            = "sheets.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "docs" {
  project            = var.project_id
  service            = "docs.googleapis.com"
  disable_on_destroy = false
}

# --- Service account ---
# A single SA is used for everything: Google APIs (Drive/Sheets/Docs) and,
# when Google Chat is enabled (Section 3), sending Chat messages. Share
# your spreadsheets, docs, and memory docs with this SA's email; its key
# (when Section 3 is enabled) gets stored in Secret Manager.
resource "google_service_account" "agent" {
  project      = var.project_id
  account_id   = var.bot_account_id
  display_name = var.bot_name
  description  = "Service account for ${var.bot_name} (Google APIs + platform integrations)"

  depends_on = [
    google_project_service.iam,
    google_project_service.drive,
    google_project_service.sheets,
    google_project_service.docs,
  ]
}

# --- Per-agent SA: permission to USE APIs on its own project ---
# `docs_utilities.py` calls `credentials.with_quota_project(AGENT_PROJECT_ID)`
# so Workspace API calls (Docs/Drive/Sheets) bill to this agent's project
# rather than the Forum's (which doesn't enable Workspace APIs). That quota-
# project override requires the caller to have `serviceusage.services.use`
# on the chosen project. Without this binding, every Docs/Drive/Sheets call
# from the deployed engine 403s with `USER_PROJECT_DENIED` — and confusingly,
# sharing the target doc with the SA doesn't help, because the call never
# reaches the per-doc permission check.
resource "google_project_iam_member" "agent_serviceusage_consumer" {
  project = var.project_id
  role    = "roles/serviceusage.serviceUsageConsumer"
  member  = "serviceAccount:${google_service_account.agent.email}"
}

# Allow service account key creation for this project. Most orgs enforce
# `constraints/iam.disableServiceAccountKeyCreation` org-wide. Google Chat
# (Section 3) needs an SA key in Secret Manager, so we override the policy
# at the project level. If you're not using Google Chat you can remove this.
resource "google_project_organization_policy" "allow_sa_key_creation" {
  project    = var.project_id
  constraint = "constraints/iam.disableServiceAccountKeyCreation"

  boolean_policy {
    enforced = false
  }
}

# The Reasoning Engine lands in the Forum's project but RUNS AS this
# agent's per-agent SA. Cross-project SA usage is blocked by default on
# orgs created after Sep 2024 (constraint defaults to enforced). Override
# it on THIS project (the SA's home) so the per-agent SA can be attached
# as the runtime SA of an engine in another project.
#
# The constraint enforces on BOTH sides, so the Forum's project also needs
# the same override. That companion override lives in the Forum's own
# terraform — not in this repo, because the agent template shouldn't
# manage org policies on a project it doesn't own. See AGENTS.md rule #3
# for the exact snippet to give the Forum operator.
#
# Skipping either override (this one or the Forum-side one) produces the
# same failure: `adk deploy agent_engine` creates the Reasoning Engine
# cleanly, then every metadata-server token request at runtime returns
# 500 and the engine never serves a single user message.
resource "google_project_organization_policy" "allow_cross_project_sa" {
  project    = var.project_id
  constraint = "constraints/iam.disableCrossProjectServiceAccountUsage"

  boolean_policy {
    enforced = false
  }
}

# NOTE — the COMPANION Forum-side override is NOT in this repo.
# `iam.disableCrossProjectServiceAccountUsage` enforces on both sides of
# cross-project SA usage: this repo's `allow_cross_project_sa` resource (above)
# releases the agent's own project (the SA's home), but the Forum project also
# needs the same override to accept the external SA at attach time. That second
# override lives in the Forum's own terraform — see AGENTS.md rule #3 for the
# exact snippet to add there. We don't manage it from this repo because cross-
# project org-policy resources should be owned by whoever owns the project they
# target (the Forum operator), not by every agent that reaches into the Forum.
#
# If you're standing up the very first agent against a fresh Forum project and
# `adk deploy agent_engine` fails with the metadata-server 500 at runtime, the
# Forum-side override is the likely cause — ask the Forum operator to add it.

# --- Staging bucket for ADK deployments ---
# `adk deploy agent_engine` uploads the agent code here before deploying
# to Vertex AI. Lifecycle policy cleans up old uploads after 7 days.
#
# Note on project placement: the Reasoning Engine itself runs in the
# Forum's project (so all agents are administratively centralized), but
# the staging bucket lives in THIS project. That keeps the agent's code
# artifacts isolated under the agent's billing and IAM. The Forum's
# Vertex AI Service Agent gets read-only access via the cross-project
# IAM binding below (`engine_staging_reader`), so it can fetch the
# packaged agent code at cold-start time.
resource "google_storage_bucket" "staging" {
  project                     = var.project_id
  name                        = "${var.project_id}-staging"
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [
    google_project_service.storage,
  ]
}

# --- Cross-project IAM for deploying as the per-agent SA ---
#
# The Reasoning Engine lands in the Forum's project but RUNS AS this
# agent's per-agent SA (configured via .agent_engine_config.json's
# `service_account` field). Three Vertex AI service agents in the Forum
# project participate at different phases of the engine lifecycle (see
# the `forum_vertex_ai_service_agents` local at the top of this file).
#
# Two bindings flow from that:
#
#   1. Token-creator (`roles/iam.serviceAccountTokenCreator`) on the
#      per-agent SA — fans out to ALL THREE service agents below.
#      Each agent mints tokens for the per-agent SA in a different
#      phase; missing any one produces a different startup failure,
#      and the -cc one (custom container runtime) is the most load-
#      bearing — its missing binding manifests as the cryptic
#      "Compute Engine Metadata server unavailable. Response status:
#      500" that the other two pass preflight without surfacing.
#
#   2. Staging-bucket reader (`roles/storage.objectViewer`) — bound
#      ONLY to the -re agent. Only -re reads the staging bucket at
#      cold-start to fetch the packaged agent code; the other two
#      never touch GCS directly, so we don't pollute the bucket's
#      IAM policy with bindings they wouldn't use.

resource "google_service_account_iam_member" "engine_token_creator" {
  for_each           = local.forum_vertex_ai_service_agents
  service_account_id = google_service_account.agent.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${each.value}"

  # The binding succeeds without iamcredentials.googleapis.com, but the
  # token-minting call at runtime won't — depend on the API explicitly so
  # `terraform apply` enables it before any deploy attempts to use it.
  depends_on = [google_project_service.iamcredentials]
}

resource "google_storage_bucket_iam_member" "engine_staging_reader" {
  # Only the -re agent reads the staging bucket at cold-start. The other
  # two never touch GCS directly, so narrowing this binding here. (The
  # token-creator binding above stays fanned out to all three because each
  # of them mints tokens in different phases.)
  bucket = google_storage_bucket.staging.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:service-${data.google_project.forum.number}@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
}

# --- Read access to The Forum's inbound-files bucket ---
#
# When a user attaches a file, The Forum uploads it to its own GCS bucket
# and embeds a `gs://` reference in the message text it sends this agent
# (e.g. `[IMAGE: gs://<forum>-slack-files/... | image/png]`). Without this
# binding the engine can see the URI but gets a 403 fetching the object,
# which surfaces to the user as a confusing "I can't read that file".
#
# `_iam_member` (not `_iam_binding`) is required: the bucket is shared by
# every agent registered with this Forum, and `_iam_binding` is
# authoritative for the role — applying it would revoke every other
# agent's access.
#
# Scoped to the bucket rather than added to
# `forum_runtime_roles_for_agent_sa` below on purpose: a project-level
# storage grant would also expose the Forum's terraform state bucket.
#
# The name is the convention from the Forum's storage.tf
# (`${project_id}-slack-files`). If a Forum operator points
# GCS_BUCKET_NAME at a differently-named bucket, update this to match.
resource "google_storage_bucket_iam_member" "engine_inbound_files_reader" {
  bucket = "${var.forum_project_id}-slack-files"
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.agent.email}"
}

# --- Per-agent SA roles on the Forum project ---
#
# The Reasoning Engine runs in the Forum project AS the per-agent SA. For
# the engine to start and serve traffic, that SA needs baseline workload
# roles on the Forum project — without them, the runtime metadata server
# returns 500 on the first token request and the engine never reaches its
# first user message (no useful log line, just a hung first reply).
# These are the same roles the Forum's default compute SA has, which is
# why the old shared-SA setup "just worked" — the per-agent SA inherits
# nothing automatically when it moves to the Forum project.
#
# The local lives next to the resource (rather than with the other locals
# at the top of the file) because the set is specifically tied to this
# one binding — easier to maintain together when adding/removing roles.
#
# Operator IAM requirement: the user running `terraform apply` needs
# `roles/resourcemanager.projectIamAdmin` on the Forum project to grant
# these. If you don't, terraform fails here with a clear PERMISSION_DENIED
# — ask the Forum's admin to grant the bindings manually with:
#   for role in roles/aiplatform.user roles/logging.logWriter \
#               roles/monitoring.metricWriter roles/cloudtrace.agent; do
#     gcloud projects add-iam-policy-binding ${var.forum_project_id} \
#       --member="serviceAccount:${var.bot_account_id}@${var.project_id}.iam.gserviceaccount.com" \
#       --role="$role"
#   done
locals {
  forum_runtime_roles_for_agent_sa = toset([
    "roles/aiplatform.user",         # invoke Vertex AI APIs at runtime
    "roles/logging.logWriter",       # emit stdout/stderr to Cloud Logging
    "roles/monitoring.metricWriter", # emit container metrics
    "roles/cloudtrace.agent",        # emit traces. NOTE: --trace_to_cloud was removed from deploy_and_update.sh (commit b5adf67) because it triggers a metadata-proxy scope bug with cross-project SAs. We keep this role granted so re-enabling tracing later is a one-line change; remove if you've decided you'll never use it.
  ])
}

resource "google_project_iam_member" "engine_runtime_roles" {
  for_each = local.forum_runtime_roles_for_agent_sa
  project  = var.forum_project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.agent.email}"
}

# ==============================================================================
# SECTION 2: SLACK
# Uncomment to enable Slack. After `terraform apply`, populate the secret:
#   echo -n "xoxb-YOUR-TOKEN" | gcloud secrets versions add \
#     ${var.bot_account_id}-slack-token --data-file=- --project=${var.project_id}
# (get_started_linux.sh does this for you when you select Slack.)
# ==============================================================================

resource "google_secret_manager_secret" "slack_bot_token" {
  project   = var.project_id
  secret_id = "${var.bot_account_id}-slack-token"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}
#
resource "google_secret_manager_secret_iam_member" "slack_token_forum_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.slack_bot_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${local.forum_runtime_sa}"
}

# ==============================================================================
# SECTION 3: GOOGLE CHAT
# Uncomment to enable Google Chat. Each Google Chat bot needs its own GCP
# project (Google Chat API restriction) — that's already handled because
# this whole template is per-agent-project.
# ==============================================================================

# resource "google_project_service" "chat" {
#   project            = var.project_id
#   service            = "chat.googleapis.com"
#   disable_on_destroy = false
# }
#
# # Grant the agent SA permission to send Google Chat messages.
# resource "google_project_iam_member" "chat_owner" {
#   project = var.project_id
#   role    = "roles/chat.owner"
#   member  = "serviceAccount:${google_service_account.agent.email}"
# }
#
# # Container for the agent SA's key (the key value itself is populated
# # post-apply: get_started_linux.sh creates the SA key, uploads it, and
# # deletes the local file).
# resource "google_secret_manager_secret" "chat_credentials" {
#   project   = var.project_id
#   secret_id = var.chat_credentials_secret_name
#
#   replication {
#     auto {}
#   }
#
#   depends_on = [google_project_service.secretmanager]
# }
#
# resource "google_secret_manager_secret_iam_member" "chat_credentials_forum_accessor" {
#   project   = var.project_id
#   secret_id = google_secret_manager_secret.chat_credentials.secret_id
#   role      = "roles/secretmanager.secretAccessor"
#   member    = "serviceAccount:${local.forum_runtime_sa}"
# }

# ==============================================================================
# SECTION 4: TELEGRAM
# Uncomment to enable Telegram. After `terraform apply`, populate the secret:
#   echo -n "YOUR_TELEGRAM_BOT_TOKEN" | gcloud secrets versions add \
#     ${var.bot_account_id}-telegram-token --data-file=- --project=${var.project_id}
# ==============================================================================

# resource "google_secret_manager_secret" "telegram_bot_token" {
#   project   = var.project_id
#   secret_id = "${var.bot_account_id}-telegram-token"
#
#   replication {
#     auto {}
#   }
#
#   depends_on = [google_project_service.secretmanager]
# }
#
# resource "google_secret_manager_secret_iam_member" "telegram_token_forum_accessor" {
#   project   = var.project_id
#   secret_id = google_secret_manager_secret.telegram_bot_token.secret_id
#   role      = "roles/secretmanager.secretAccessor"
#   member    = "serviceAccount:${local.forum_runtime_sa}"
# }

# ==============================================================================
# SECTION 5: DISCORD
#
# Unlike Slack/Telegram, Discord uses a long-lived Gateway WebSocket rather
# than an HTTP webhook. The Forum runs a SINGLE multi-tenant discord-worker
# VM in its own project; that worker auto-discovers Discord-enabled agents
# from Firestore at runtime. To onboard a Discord agent:
#   1. Uncomment this section, `terraform apply`.
#   2. Populate ${bot_account_id}-discord-token with the bot token.
#   3. register_agent.py writes the Firestore platform block on next deploy.
#   4. Wait up to 300s (or reset the worker VM in The Forum's project) for
#      the worker to pick up the new bot.
# No terraform changes in The Forum's repo are required.
#
# Discord needs TWO cross-project secretAccessor bindings: the discord-worker
# VM SA (reads the token to open the inbound Gateway WebSocket) AND The Forum's
# Cloud Run SA (reads the token to send outbound REST replies). Granting only
# one results in DMs reaching The Forum but every reply 403ing.
# ==============================================================================

# resource "google_secret_manager_secret" "discord_bot_token" {
#   project   = var.project_id
#   secret_id = "${var.bot_account_id}-discord-token"
#
#   replication {
#     auto {}
#   }
#
#   depends_on = [google_project_service.secretmanager]
# }
#
# # (a) The Forum's discord-worker VM SA — for inbound Gateway connection.
# resource "google_secret_manager_secret_iam_member" "discord_token_worker_accessor" {
#   project   = var.project_id
#   secret_id = google_secret_manager_secret.discord_bot_token.secret_id
#   role      = "roles/secretmanager.secretAccessor"
#   member    = "serviceAccount:discord-worker@${var.forum_project_id}.iam.gserviceaccount.com"
# }
#
# # (b) The Forum's Cloud Run SA — for outbound REST replies.
# resource "google_secret_manager_secret_iam_member" "discord_token_forum_accessor" {
#   project   = var.project_id
#   secret_id = google_secret_manager_secret.discord_bot_token.secret_id
#   role      = "roles/secretmanager.secretAccessor"
#   member    = "serviceAccount:${local.forum_runtime_sa}"
# }

# ==============================================================================
# SECTION 6: SCHEDULER MCP KEY
#
# Uncomment to use The Forum's scheduler MCP server (the only MCP server
# The Forum hosts). The agent uses this to create/list/update/delete
# scheduled reminders for its users via tool calls.
#
# Three-step flow:
#   1. Uncomment + `terraform apply` (creates the empty secret container +
#      IAM binding for the Reasoning Engine SA).
#   2. From The Forum repo, run `python scripts/provision_scheduler_api_key.py
#      --agent-id YOUR_AGENT_FIRESTORE_ID`. Copy the printed plaintext.
#   3. Populate: echo -n "PLAINTEXT" | gcloud secrets versions add \
#        ${bot_account_id}-scheduler-mcp-key --data-file=- --project=PROJECT
#
# get_started_linux.sh walks you through steps 2 and 3 if you select this option.
# ==============================================================================

resource "google_secret_manager_secret" "scheduler_mcp_key" {
  project   = var.project_id
  secret_id = "${var.bot_account_id}-scheduler-mcp-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}
#
# # Grant the per-agent SA read access. The Reasoning Engine runs AS this SA
# # (set via .agent_engine_config.json's service_account field — see AGENTS.md
# # rule #3), and agent.py reads this key from Secret Manager at startup. This
# # is the same per-agent SA every other runtime secret binds to.
resource "google_secret_manager_secret_iam_member" "scheduler_mcp_key_reasoning_engine" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.scheduler_mcp_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

# --- Google OAuth user token for Drive + Sheets ---
#
# Mike acts as a real Google user for Drive work, not as this service
# account. A service account has no Drive storage quota, and in Drive the
# creator owns the file — so the SA can edit a workbook shared with it but
# cannot create a vehicle workbook, archive a receipt, or file an insurance
# PDF. Those fail `storageQuotaExceeded`.
#
# Populate the value with mint_oauth_token.py (never in terraform):
#   python mint_oauth_token.py --client-secret ~/client_secret.json \
#     | gcloud secrets versions add ${var.bot_account_id}-google-oauth \
#         --data-file=- --project=${var.project_id}
#
# Superseded once the fleet lives in a Shared Drive, where files are owned
# by the drive rather than the creator.
resource "google_secret_manager_secret" "google_oauth" {
  project   = var.project_id
  secret_id = "${var.bot_account_id}-google-oauth"

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret_iam_member" "google_oauth_reasoning_engine" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.google_oauth.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent.email}"
}

# ==============================================================================
# OUTPUTS
# ==============================================================================

output "project_id" {
  description = "GCP project ID hosting this agent"
  value       = var.project_id
}

output "service_account_email" {
  description = "Agent service account email. Share Google Sheets/Drive/Docs with this email; it also signs Google Chat messages when Section 3 is enabled."
  value       = google_service_account.agent.email
}

output "staging_bucket" {
  description = "GCS bucket for ADK deployment staging"
  value       = google_storage_bucket.staging.name
}

output "forum_runtime_sa" {
  description = "The Forum's Cloud Run service account (the principal granted secretAccessor on this agent's platform secrets)"
  value       = local.forum_runtime_sa
}
