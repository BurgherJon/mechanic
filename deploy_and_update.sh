#!/usr/bin/env bash
#
# deploy_and_update.sh — Deploy this agent to Vertex AI, smoke-test it,
# register it (with all enabled platforms) in The Forum's Firestore via
# register_agent.py, and delete the previous Reasoning Engine.
#
# Usage:
#   ./deploy_and_update.sh
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - ADK installed (path in ADK_BIN, or on PATH)
#   - terraform/terraform.tfvars present (used by register_agent.py)
#   - At least one platform secret populated in the agent project's Secret
#     Manager (slack token, telegram token, discord token, or chat SA key)
#
# Configuration is read from .env in this directory. Required keys:
#   GOOGLE_CLOUD_PROJECT (= the Forum's project; the Reasoning Engine
#     lives in this project administratively, alongside every other agent),
#   AGENT_PROJECT_ID (= this agent's own project; the per-agent SA, secrets,
#     and staging bucket live here),
#   AGENT_DISPLAY_NAME, FORUM_PROJECT_ID (same as GOOGLE_CLOUD_PROJECT),
#   ADK_BIN, ADK_PYTHON.
#
# The Reasoning Engine RUNS AS the per-agent SA via the `service_account`
# field in .agent_engine_config.json (next to this script). Without that
# config file the engine would inherit the Forum project's default
# compute SA — shared with every other agent — which defeats per-agent
# isolation. Confirm the file is present and points at the right SA.

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    echo "Loading environment from ${SCRIPT_DIR}/.env..."
    # Parse line-by-line so unquoted values with spaces (e.g.
    # AGENT_DISPLAY_NAME=Agent Demo) don't get re-tokenized by the shell.
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        # strip one layer of surrounding quotes if present
        [[ "$value" == \"*\" && "$value" == *\" ]] && value="${value:1:-1}"
        [[ "$value" == \'*\' && "$value" == *\' ]] && value="${value:1:-1}"
        export "$key=$value"
    done < "${SCRIPT_DIR}/.env"
else
    echo "ERROR: .env not found at ${SCRIPT_DIR}/.env"
    echo "Run ./get_started_linux.sh to generate it, or copy .env.example to .env and edit."
    exit 1
fi

# Deploy target = the Forum's project. The Reasoning Engine runs there
# (alongside every other agent), but uses this agent's per-agent SA as
# its runtime identity (see .agent_engine_config.json).
FORUM_PROJECT_ID="${FORUM_PROJECT_ID:?FORUM_PROJECT_ID must be set in .env}"

# Agent's own project = where the SA, secrets, and staging bucket live.
AGENT_PROJECT_ID="${AGENT_PROJECT_ID:?AGENT_PROJECT_ID must be set in .env}"

# GOOGLE_CLOUD_PROJECT and FORUM_PROJECT_ID should normally match in .env
# (both = the Forum's project), but tolerate one being unset.
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-$FORUM_PROJECT_ID}"

REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
AGENT_DISPLAY_NAME="${AGENT_DISPLAY_NAME:?AGENT_DISPLAY_NAME must be set in .env}"
ADK_BIN="${ADK_BIN:-$(command -v adk 2>/dev/null || echo adk)}"
ADK_PYTHON="${ADK_PYTHON:-$(dirname "$ADK_BIN")/python3}"
AGENT_DIR="${SCRIPT_DIR}"

# Hard-fail if the SA-assignment config is missing — without it the engine
# silently inherits the Forum's compute SA (shared with every other agent)
# and per-agent secret/doc isolation breaks. get_started_linux.sh generates
# this file; if it's gone, something went wrong.
if [[ ! -f "${SCRIPT_DIR}/.agent_engine_config.json" ]]; then
    echo "ERROR: .agent_engine_config.json missing in ${SCRIPT_DIR}."
    echo "  Without this file the deployed engine would run as the Forum's"
    echo "  default compute SA instead of this agent's per-agent SA, and"
    echo "  every other agent's per-doc / per-secret IAM would apply to it."
    echo "  Re-run ./get_started_linux.sh to regenerate it, or recreate from"
    echo "  the template at the same path."
    exit 1
fi

# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------
log()  { echo -e "\n\033[1;34m> $*\033[0m"; }
ok()   { echo -e "\033[1;32m  [OK] $*\033[0m"; }
err()  { echo -e "\033[1;31m  [xx] $*\033[0m" >&2; }
warn() { echo -e "\033[1;33m  [!!] $*\033[0m"; }

get_existing_agent_id() {
    "$ADK_PYTHON" -c "
import vertexai
from vertexai.preview import reasoning_engines
vertexai.init(project='${FORUM_PROJECT_ID}', location='${REGION}')
for e in reasoning_engines.ReasoningEngine.list():
    if '${AGENT_DISPLAY_NAME}' in (e.display_name or ''):
        print(e.resource_name.split('/')[-1])
        break
" 2>/dev/null || true
}

get_agent_resource_name() {
    echo "projects/${FORUM_PROJECT_ID}/locations/${REGION}/reasoningEngines/$1"
}

# ------------------------------------------------------------------
# Pre-flight
# ------------------------------------------------------------------
log "Pre-flight"

# We deploy to the Forum's project. The agent's own project is only
# touched indirectly (via the SA assignment, staging bucket reads, and
# cross-project secret IAM bindings already wired by terraform).
gcloud config set project "$FORUM_PROJECT_ID" --quiet
ok "gcloud project set to $FORUM_PROJECT_ID (deploy target)"
ok "Per-agent SA + secrets live in: $AGENT_PROJECT_ID"

if ! command -v "$ADK_BIN" >/dev/null 2>&1 && [[ ! -x "$ADK_BIN" ]]; then
    err "ADK binary not found at $ADK_BIN"
    echo "  Install with: pip install google-adk"
    echo "  Then set ADK_BIN in .env to the full path (e.g. /path/to/venv/bin/adk)."
    exit 1
fi
ok "ADK binary: $ADK_BIN"

# ------------------------------------------------------------------
# Pre-flight: deploy bundle size guard
# ------------------------------------------------------------------
# `adk deploy agent_engine` tars AGENT_DIR (minus .ae_ignore matches) and
# uploads it. Vertex AI rejects payloads over 8 MB with an opaque 400
# (INVALID_ARGUMENT ... payload size exceeds), and only AFTER a full upload
# attempt. The usual cause is a virtualenv or data dir that .ae_ignore didn't
# exclude. Simulate the bundle (same basename-fnmatch rules ADK applies via
# shutil.ignore_patterns) and fail fast with the offenders BEFORE uploading.
# Override the cap with MAX_BUNDLE_MB=<n> for a legitimately large bundle.
log "Pre-flight: checking deploy bundle size against the 8 MB Vertex limit..."
BUNDLE_REPORT=$("$ADK_PYTHON" - "$AGENT_DIR" <<'PYEOF'
import os, sys, fnmatch

agent_dir = sys.argv[1]
limit_mb = float(os.environ.get("MAX_BUNDLE_MB", "8"))
limit = limit_mb * 1024 * 1024

# .ae_ignore is fed to shutil.ignore_patterns: fnmatch on basenames, not
# gitignore semantics. Mirror that exactly so the estimate matches the upload.
patterns = []
ae = os.path.join(agent_dir, ".ae_ignore")
if os.path.exists(ae):
    with open(ae) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)

def ignored(name):
    return any(fnmatch.fnmatch(name, p) for p in patterns)

def human(n):
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024

total = 0
by_top = {}
for root, dirs, files in os.walk(agent_dir):
    dirs[:] = [d for d in dirs if not ignored(d)]   # prune ignored dirs at every depth
    for f in files:
        if ignored(f):
            continue
        fp = os.path.join(root, f)
        try:
            sz = os.path.getsize(fp)
        except OSError:
            continue
        total += sz
        top = os.path.relpath(fp, agent_dir).split(os.sep)[0]
        by_top[top] = by_top.get(top, 0) + sz

if total > limit:
    print("TOOBIG")
    print(f"Bundle is {human(total)} after .ae_ignore (Vertex AI payload limit is {limit_mb:.0f} MB).")
    print("Biggest entries:")
    for name, sz in sorted(by_top.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {human(sz):>10}  {name}")
else:
    print(f"OK {human(total)}")
PYEOF
) || {
    warn "Bundle size pre-check could not run (Python error). Proceeding without it."
    BUNDLE_REPORT=""
}

BUNDLE_STATUS=$(echo "$BUNDLE_REPORT" | head -1)
if [[ "$BUNDLE_STATUS" == "TOOBIG" ]]; then
    err "Deploy bundle exceeds the Vertex AI payload limit — NOT deploying."
    echo "$BUNDLE_REPORT" | tail -n +2 | sed 's/^/  /' >&2
    err "Almost always a venv or data dir that .ae_ignore didn't exclude."
    err "Fix .ae_ignore (basename fnmatch — e.g. add a bare 'venv' line), or if the"
    err "bundle is legitimately large: MAX_BUNDLE_MB=<n> ./deploy_and_update.sh"
    exit 1
elif [[ "$BUNDLE_STATUS" == OK* ]]; then
    ok "Deploy bundle size: ${BUNDLE_STATUS#OK }"
fi

# ------------------------------------------------------------------
# Step 1: Look for the existing Reasoning Engine (for blue/green)
# ------------------------------------------------------------------
log "Step 1: Looking for existing '${AGENT_DISPLAY_NAME}' Reasoning Engine..."
OLD_AGENT_ID=$(get_existing_agent_id)
if [[ -n "$OLD_AGENT_ID" ]]; then
    ok "Found: $(get_agent_resource_name "$OLD_AGENT_ID")"
else
    warn "No existing Reasoning Engine found. Will create a new one."
fi

# ------------------------------------------------------------------
# Step 2: Deploy the new Reasoning Engine
# ------------------------------------------------------------------
log "Step 2: Deploying agent to Vertex AI Agent Engine..."
echo "  Engine project:   $FORUM_PROJECT_ID  (Forum project — where every agent lives)"
echo "  Staging bucket:   gs://${AGENT_PROJECT_ID}-staging  (in this agent's project)"
echo "  Region:           $REGION"
echo "  Source:           $AGENT_DIR"
echo "  Runtime SA:       (from .agent_engine_config.json — should be this agent's per-agent SA)"

AGENT_PARENT_DIR="$(dirname "$AGENT_DIR")"
AGENT_PACKAGE_NAME="$(basename "$AGENT_DIR")"

DEPLOY_EXIT=0
DEPLOY_OUTPUT=$(cd "$AGENT_PARENT_DIR" && "$ADK_BIN" deploy agent_engine \
    --project "$FORUM_PROJECT_ID" \
    --region "$REGION" \
    --staging_bucket "gs://${AGENT_PROJECT_ID}-staging" \
    --display_name "$AGENT_DISPLAY_NAME" \
    --agent_engine_config_file "${AGENT_DIR}/.agent_engine_config.json" \
    "$AGENT_PACKAGE_NAME" 2>&1) || DEPLOY_EXIT=$?
echo "$DEPLOY_OUTPUT"

# Decide success from adk's exit code AND its output — NEVER from a scraped
# engine ID. adk can exit 0 while the engine failed to start, and that failure
# output can still contain a reasoningEngines/<id> path, so a scraped ID is not
# evidence of a working deploy. Require all three: a clean exit, no failure
# marker in the output, and a printed Reasoning Engine resource name. Even then
# we don't cut over — Step 3 verifies against the live API first.
if [[ "$DEPLOY_EXIT" -ne 0 ]]; then
    err "Deployment failed (adk exited $DEPLOY_EXIT). Old agent untouched."
    err "Check logs:"
    err "  gcloud logging read 'resource.type=\"aiplatform.googleapis.com/ReasoningEngine\"' --project=$FORUM_PROJECT_ID --limit=50"
    exit 1
fi
if echo "$DEPLOY_OUTPUT" | grep -qiE "Deploy failed:|failed to start|does not exist|Traceback \(most recent call last\)"; then
    err "adk exited 0 but its output reports a failure. Old agent untouched."
    err "Check logs:"
    err "  gcloud logging read 'resource.type=\"aiplatform.googleapis.com/ReasoningEngine\"' --project=$FORUM_PROJECT_ID --limit=50"
    exit 1
fi
if ! echo "$DEPLOY_OUTPUT" | grep -qE "reasoningEngines/[0-9]+"; then
    err "Deploy output has no Reasoning Engine resource name — treating as failed."
    err "Old agent untouched. adk prints the created engine's resource name on a"
    err "real deploy; its absence means the deploy did not complete. See output above."
    exit 1
fi

NEW_AGENT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oP 'reasoningEngines/\K[0-9]+' | tail -1)
NEW_RESOURCE_NAME=$(get_agent_resource_name "$NEW_AGENT_ID")
ok "adk reported a deploy: $NEW_RESOURCE_NAME (verifying against the live API next)"

# ------------------------------------------------------------------
# Step 3: Verify the new engine is actually live (exists + serves)
# ------------------------------------------------------------------
# This is the gate that protects the cutover. Two authoritative checks against
# the live API, both fatal:
#   1. Existence — the engine must appear in the Reasoning Engine list. A deploy
#      whose container failed to start is auto-deleted by adk, so a dead deploy
#      will NOT be listed (the ID scraped in Step 2 is meaningless on its own).
#   2. Liveness — it must create a session. A 404 / "does not exist" here means
#      it cannot serve traffic.
# Steps 4-6 (repoint Firestore, delete the old agent) run ONLY if this passes,
# so a failed deploy can never repoint to a dead engine or delete the fallback.
log "Step 3: Verifying new engine exists and serves traffic..."
VERIFY_RESULT=$("$ADK_PYTHON" -c "
import sys
import vertexai
from vertexai.preview import reasoning_engines
vertexai.init(project='${FORUM_PROJECT_ID}', location='${REGION}')

target_id = '${NEW_AGENT_ID}'
resource_name = '${NEW_RESOURCE_NAME}'

# 1) Existence: the engine must be in the live list. A failed-to-start deploy
#    is auto-deleted by adk, so it won't appear here even though Step 2 scraped
#    its (now-dead) ID from the deploy output.
live_ids = [e.resource_name.split('/')[-1] for e in reasoning_engines.ReasoningEngine.list()]
if target_id not in live_ids:
    print(f'NOT_FOUND: engine {target_id} is not in the live engine list (deploy likely failed and was auto-deleted)')
    sys.exit(1)

# 2) Liveness: it must actually create a session.
agent = reasoning_engines.ReasoningEngine(resource_name)
session = agent.create_session(user_id='smoke-test')
print(f'Session created: {session[\"id\"]}')
print('VERIFIED_OK')
" 2>&1) || true

if echo "$VERIFY_RESULT" | grep -q "VERIFIED_OK"; then
    ok "New engine verified live (exists + creates sessions)."
else
    err "New engine verification FAILED — it does not exist or cannot serve traffic."
    err "NOT repointing Firestore and NOT deleting the old agent; the previous"
    err "engine (if any) is untouched and still serving."
    echo "$VERIFY_RESULT" | tail -15
    err "Check logs:"
    err "  gcloud logging read 'resource.type=\"aiplatform.googleapis.com/ReasoningEngine\" AND resource.labels.reasoning_engine_id=\"$NEW_AGENT_ID\"' --project=$FORUM_PROJECT_ID --limit=50"
    exit 1
fi

# ------------------------------------------------------------------
# Step 4: Register with The Forum
# ------------------------------------------------------------------
log "Step 4: Registering agent in The Forum's Firestore..."

"$ADK_PYTHON" -m pip install --quiet google-cloud-firestore google-cloud-secret-manager 2>/dev/null || true

"$ADK_PYTHON" "${SCRIPT_DIR}/register_agent.py" \
    --agent-name "$AGENT_DISPLAY_NAME" \
    --vertex-ai-agent-id "$NEW_RESOURCE_NAME" \
    --firestore-project "$FORUM_PROJECT_ID" || {
    err "Agent registration failed!"
    echo "  New Reasoning Engine is live at: $NEW_RESOURCE_NAME"
    echo "  You can re-run register_agent.py manually after fixing the issue."
    exit 1
}
ok "The Forum's Firestore updated."

# ------------------------------------------------------------------
# Step 5: Clear stale sessions for this agent
# ------------------------------------------------------------------
log "Step 5: Clearing stale sessions..."
SESSIONS_DELETED=$("$ADK_PYTHON" -c "
from google.cloud import firestore
db = firestore.Client(project='${FORUM_PROJECT_ID}')

agents = db.collection('agents').where('display_name', '==', '${AGENT_DISPLAY_NAME}').stream()
agent_doc_id = None
for agent in agents:
    agent_doc_id = agent.id
    break

if not agent_doc_id:
    print('0')
else:
    deleted = 0
    for session in db.collection('sessions').stream():
        if agent_doc_id in session.id:
            db.collection('sessions').document(session.id).delete()
            deleted += 1
    print(deleted)
" 2>/dev/null) || SESSIONS_DELETED="0"

if [[ "$SESSIONS_DELETED" -gt 0 ]]; then
    ok "Cleared $SESSIONS_DELETED stale session(s)."
else
    ok "No stale sessions to clear."
fi

# ------------------------------------------------------------------
# Step 6: Delete the old Reasoning Engine
# ------------------------------------------------------------------
if [[ -n "$OLD_AGENT_ID" && "$OLD_AGENT_ID" != "$NEW_AGENT_ID" ]]; then
    log "Step 6: Cleaning up old Reasoning Engine ($OLD_AGENT_ID)..."
    OLD_RESOURCE_NAME=$(get_agent_resource_name "$OLD_AGENT_ID")

    ACCESS_TOKEN=$(gcloud auth print-access-token)
    if curl -s -X DELETE \
        "https://${REGION}-aiplatform.googleapis.com/v1beta1/${OLD_RESOURCE_NAME}?force=true" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        | grep -q '"done": true'; then
        ok "Old Reasoning Engine deleted: $OLD_RESOURCE_NAME"
    else
        warn "Could not auto-delete old Reasoning Engine $OLD_RESOURCE_NAME — delete it manually if not needed."
    fi
else
    log "Step 6: No old Reasoning Engine to clean up."
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo
echo "==========================================================="
echo "  Deployment complete!"
echo "==========================================================="
echo "  Agent:        $AGENT_DISPLAY_NAME"
echo "  New engine:   $NEW_RESOURCE_NAME"
echo "  The Forum:    Updated in Firestore (project=$FORUM_PROJECT_ID)"
if [[ -n "${OLD_AGENT_ID:-}" && "$OLD_AGENT_ID" != "$NEW_AGENT_ID" ]]; then
    echo "  Old engine:   Deleted ($(get_agent_resource_name "$OLD_AGENT_ID"))"
fi
echo
echo "  Test it by sending a DM to your bot on one of its enabled platforms."
echo "==========================================================="
