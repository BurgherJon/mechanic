#!/usr/bin/env python3
"""
Register this agent with The Forum's Firestore, with multi-platform
configuration auto-detected from Secret Manager.

Platforms are auto-detected by checking which secrets exist in the agent's
GCP project. Secret IDs follow the same naming convention that
terraform/main.tf uses, so adding a platform is a matter of `terraform
apply` + storing a token + re-running this script (which `deploy_and_update.sh`
does automatically).

Each detected platform is validated via its native API before being
written to Firestore, so a broken token won't silently land in the
agent's document:

  - Slack:       auth.test         -> captures bot user_id
  - Telegram:    getMe             -> captures bot username
  - Discord:     users/@me         -> captures bot user_id; requires
                                      discord_application_id in tfvars
  - Google Chat: parses the SA key JSON for required fields
                 (no native auth-ping endpoint)

Usage:
    python register_agent.py \\
        --agent-name "My Agent" \\
        --vertex-ai-agent-id projects/PROJ/locations/REGION/reasoningEngines/ID \\
        [--tfvars terraform/terraform.tfvars] \\
        [--firestore-project your-forum-project-id]

`deploy_and_update.sh` calls this for you. You usually only invoke it
directly if you're testing the registration step in isolation.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from google.api_core import exceptions as google_exc
from google.cloud import firestore, secretmanager

# MIME types Mike can read, written to the agent's Firestore record. The Forum
# gates attachments on this before invoking the agent, so anything not listed
# here is rejected with a friendly message and never reaches us.
#
# This list REPLACES the Forum's default image allowlist rather than extending
# it — so the image types have to be spelled out alongside application/pdf, or
# Mike would stop receiving photographed receipts the moment PDFs were added.
ACCEPTED_FILE_TYPES = [
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/heic",
    "image/heif",
    "application/pdf",
]


def parse_tfvars(path: Path) -> dict:
    """Parse simple `key = "value"` lines from a terraform.tfvars file."""
    if not path.exists():
        sys.exit(f"terraform.tfvars not found at {path}")
    pattern = re.compile(r'^\s*(\w+)\s*=\s*"([^"]*)"')
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def secret_exists(sm_client, project_id: str, secret_id: str) -> bool:
    name = f"projects/{project_id}/secrets/{secret_id}"
    try:
        sm_client.get_secret(request={"name": name})
        return True
    except google_exc.NotFound:
        return False


def access_secret(sm_client, project_id: str, secret_id: str) -> str:
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = sm_client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


def validate_slack(token: str) -> str:
    """Call Slack auth.test. Returns the bot user_id."""
    req = urllib.request.Request(
        "https://slack.com/api/auth.test",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        sys.exit(f"  [x] Slack auth.test failed: {data.get('error')}")
    print(f"  [OK] Slack:       bot @{data.get('user')} (user_id {data['user_id']})")
    return data["user_id"]


def validate_telegram(token: str) -> dict:
    """Call Telegram getMe. Returns the bot info dict."""
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"  [x] Telegram getMe failed: HTTP {e.code} {e.reason}")
    if not data.get("ok"):
        sys.exit(f"  [x] Telegram getMe failed: {data.get('description')}")
    bot = data["result"]
    print(f"  [OK] Telegram:    bot @{bot.get('username')} (id {bot.get('id')})")
    return bot


def validate_discord(token: str) -> str:
    """Call Discord users/@me. Returns the bot user_id (snowflake)."""
    # Discord's API requires a descriptive User-Agent or it returns 403.
    req = urllib.request.Request(
        "https://discord.com/api/v10/users/@me",
        headers={
            "Authorization": f"Bot {token}",
            "User-Agent": "ComitesAgentRegistrar/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"  [x] Discord users/@me failed: HTTP {e.code} {e.reason}")
    bot_id = data.get("id")
    if not bot_id:
        sys.exit(f"  [x] Discord users/@me returned unexpected payload: {data}")
    print(f"  [OK] Discord:     bot @{data.get('username')} (id {bot_id})")
    return bot_id


def validate_google_chat(key_json: str) -> str:
    """Parse the SA key JSON and verify required fields. Returns client_email."""
    try:
        key = json.loads(key_json)
    except json.JSONDecodeError as e:
        sys.exit(f"  [x] Google Chat credentials secret is not valid JSON: {e}")
    missing = [f for f in ("client_email", "private_key", "project_id") if f not in key]
    if missing:
        sys.exit(f"  [x] Google Chat credentials missing required fields: {missing}")
    print(f"  [OK] Google Chat: SA {key['client_email']}")
    return key["client_email"]


def build_platforms(tfvars: dict, sm_client) -> tuple[list[dict], dict]:
    """Probe Secret Manager for each platform's expected secret and validate.

    Returns (platforms_array_for_firestore, top_level_legacy_fields).
    """
    agent_project = tfvars["project_id"]
    bot_account_id = tfvars["bot_account_id"]
    chat_secret_name = tfvars.get("chat_credentials_secret_name")
    forum_project = tfvars.get("forum_project_id", "your-forum-project-id")
    discord_app_id = tfvars.get("discord_application_id", "")

    slack_secret = f"{bot_account_id}-slack-token"
    telegram_secret = f"{bot_account_id}-telegram-token"
    discord_secret = f"{bot_account_id}-discord-token"

    platforms = []
    legacy = {}

    if secret_exists(sm_client, agent_project, slack_secret):
        token = access_secret(sm_client, agent_project, slack_secret)
        bot_id = validate_slack(token)
        platforms.append({
            "platform": "slack",
            "enabled": True,
            "slack_bot_id": bot_id,
            "slack_bot_token_secret": slack_secret,
            "slack_bot_token_project_id": agent_project,
        })
        # The Forum's legacy lookup path queries by top-level slack_bot_id.
        legacy["slack_bot_id"] = bot_id

    if chat_secret_name and secret_exists(sm_client, agent_project, chat_secret_name):
        key_json = access_secret(sm_client, agent_project, chat_secret_name)
        validate_google_chat(key_json)
        platforms.append({
            "platform": "google_chat",
            "enabled": True,
            "google_chat_service_account_secret": chat_secret_name,
            "google_chat_project_id": agent_project,
        })

    if secret_exists(sm_client, agent_project, telegram_secret):
        token = access_secret(sm_client, agent_project, telegram_secret)
        validate_telegram(token)
        platforms.append({
            "platform": "telegram",
            "enabled": True,
            "telegram_bot_token_secret": telegram_secret,
            "telegram_bot_token_project_id": agent_project,
        })

    if secret_exists(sm_client, agent_project, discord_secret):
        # The Forum's discord-worker authorizes incoming events using the
        # SA email we write here; it must exactly match the SA on the
        # worker VM. See The Forum's terraform output
        # `discord_worker_service_account`.
        worker_sa = f"discord-worker@{forum_project}.iam.gserviceaccount.com"
        try:
            token = access_secret(sm_client, agent_project, discord_secret)
        except google_exc.FailedPrecondition:
            print(f"  [!] Discord:      secret exists but has no version yet — skipping. "
                  f"Add the token with `gcloud secrets versions add {discord_secret}` and rerun.")
        else:
            validate_discord(token)
            if not discord_app_id:
                sys.exit("  [x] Discord secret is populated but discord_application_id is empty "
                         "in terraform.tfvars. Set it from the Discord Developer Portal "
                         "(General Information -> Application ID) and rerun.")
            platforms.append({
                "platform": "discord",
                "enabled": True,
                "discord_bot_token_secret": discord_secret,
                "discord_bot_token_project_id": agent_project,
                "discord_application_id": discord_app_id,
                "discord_worker_service_account": worker_sa,
            })

    return platforms, legacy


def main():
    parser = argparse.ArgumentParser(description="Register agent in The Forum's Firestore with auto-detected platforms")
    parser.add_argument("--agent-name", required=True,
                        help='Display name (e.g., "My Agent"); used as Firestore lookup key')
    parser.add_argument("--vertex-ai-agent-id", required=True,
                        help="Full Reasoning Engine resource name from `adk deploy`")
    parser.add_argument("--tfvars", default=None,
                        help="Path to terraform.tfvars (default: terraform/terraform.tfvars next to this script)")
    parser.add_argument("--firestore-project", default="your-forum-project-id",
                        help="Project hosting The Forum's Firestore (default: your-forum-project-id)")
    args = parser.parse_args()

    tfvars_path = Path(args.tfvars) if args.tfvars else Path(__file__).parent / "terraform" / "terraform.tfvars"
    tfvars = parse_tfvars(tfvars_path)

    for required in ("project_id", "bot_account_id"):
        if required not in tfvars:
            sys.exit(f"terraform.tfvars missing required key: {required}")

    print(f"Registering agent: {args.agent_name}")
    print(f"  Vertex AI:     {args.vertex_ai_agent_id}")
    print(f"  Agent project: {tfvars['project_id']}")
    print(f"  Firestore in:  {args.firestore_project}")
    print()
    print("Detecting platforms from Secret Manager...")

    sm_client = secretmanager.SecretManagerServiceClient()
    platforms, legacy = build_platforms(tfvars, sm_client)

    if not platforms:
        sys.exit("\nNo platform secrets found in the agent project. "
                 "Uncomment a platform section in terraform/main.tf, run `terraform apply`, "
                 "populate the secret, and re-run.")

    db = firestore.Client(project=args.firestore_project)
    query = db.collection("agents").where("display_name", "==", args.agent_name).limit(1)
    existing = list(query.stream())

    agent_data = {
        "display_name": args.agent_name,
        "vertex_ai_agent_id": args.vertex_ai_agent_id,
        "platforms": platforms,
        "accepted_file_types": ACCEPTED_FILE_TYPES,
        "updated_at": datetime.now(timezone.utc),
        **legacy,
    }

    if existing:
        doc_id = existing[0].id
        update_data = dict(agent_data)
        # Phase out plaintext slack_bot_token written by older deploy paths.
        # Secret references in the platforms array are the supported path.
        update_data["slack_bot_token"] = firestore.DELETE_FIELD
        db.collection("agents").document(doc_id).update(update_data)
        print(f"\n[OK] Updated existing agent doc: {doc_id}")
    else:
        agent_data["created_at"] = datetime.now(timezone.utc)
        _, doc_ref = db.collection("agents").add(agent_data)
        doc_id = doc_ref.id
        print(f"\n[OK] Created new agent doc: {doc_id}")

    print(f"\nAGENT_ID={doc_id}")
    print("(Set this in your .env if your agent needs to know its own Firestore ID, then redeploy.)")


if __name__ == "__main__":
    main()
