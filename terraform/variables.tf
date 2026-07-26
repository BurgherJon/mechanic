variable "project_id" {
  description = "GCP Project ID for this agent. Must already exist and have billing linked — get_started_linux.sh verifies via `gcloud projects describe` but does not create it."
  type        = string
}

variable "region" {
  description = "Default region for resources"
  type        = string
  default     = "us-central1"
}

variable "bot_name" {
  description = "Display name for the agent (used across platforms and as the Firestore lookup key)"
  type        = string
}

variable "bot_account_id" {
  description = "Service account ID base (lowercase, hyphens only, max 30 chars). Used for the agent service account and all platform secret names: {bot_account_id}-slack-token, {bot_account_id}-telegram-token, etc."
  type        = string
}

variable "bot_description" {
  description = "Description of what the agent does (used for Google Chat configuration)"
  type        = string
  default     = "AI assistant powered by Vertex AI on Comites.ai's The Forum"
}

variable "bot_avatar_url" {
  description = "URL for the bot's avatar image (used for Google Chat, optional)"
  type        = string
  default     = ""
}

variable "chat_credentials_secret_name" {
  description = "Name for the Google Chat service account key secret in Secret Manager. Only used when Section 3 (Google Chat) is uncommented."
  type        = string
  default     = "agent-chat-credentials"
}

variable "forum_project_id" {
  description = "The GCP project ID where The Forum (slack-vertex-ai-middleware) is deployed. Used for cross-project IAM bindings so The Forum's Cloud Run SA can read this agent's platform secrets."
  type        = string
}

variable "discord_application_id" {
  description = "Discord application ID from the Developer Portal (General Information → Application ID). Required only if Section 5 (Discord) is uncommented and the Discord secret is populated; register_agent.py writes it onto the Firestore platform block for traceability. Leave empty if not using Discord."
  type        = string
  default     = ""
}
