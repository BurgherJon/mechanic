# AGENTS.md

Hard rules and guidance for AI coding agents (Claude Code, Cursor, Copilot, etc.) working in this repository. This file follows the [agents.md](https://agents.md/) convention.

This agent is built on **[The Forum](https://github.com/Comites-ai/the-forum)** — [Comites.ai](https://comites.ai)'s open-source platform that bridges messaging platforms (Slack, Google Chat, Telegram, Discord) to AI agents on Vertex AI. Detailed background on how The Forum works lives in The Forum repo at [`docs/FOR_AGENT_DEVELOPERS.md`](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md) — read it before doing anything non-trivial.

## Hard rules

These are invariants. Breaking them silently breaks the agent. Don't.

### 1. Infrastructure changes go through terraform

Never create or modify GCP resources for this agent via the Cloud Console, `gcloud` one-liners, or the GCP web UI. Always:

1. Edit `terraform/main.tf`.
2. `terraform apply`.
3. Commit the change.

The platform sections (Slack/Google Chat/Telegram/Discord/Scheduler MCP) in `main.tf` are intentionally commented out. **Uncomment** them when enabling a platform — never delete them. The two exceptions to "everything in terraform" are: (a) the GCP project itself, and (b) the terraform state bucket — both bootstrapped by `get_started_linux.sh` via gcloud. Don't try to bring those into terraform; the chicken-and-egg with the GCS backend isn't worth it.

### 2. Secrets live in GCP Secret Manager — never in code, .env, or terraform state

- Plaintext secrets must not appear in `.env`, `terraform.tfvars`, code, comments, commit messages, or PR descriptions.
- Use `secret_utilities.get_secret_from_secret_manager(project_id, secret_id)` to fetch them at runtime.
- Each secret must have an IAM binding granting `roles/secretmanager.secretAccessor` to whatever principal needs to read it (The Forum's Cloud Run SA for platform tokens; the agent's Reasoning Engine SA for things the agent itself reads). The IAM binding lives in `terraform/main.tf` next to the secret container.
- If you see a `403 Permission Denied` on a secret read, the fix is almost always to add or correct that IAM binding in terraform. Don't grant the permission via `gcloud secrets add-iam-policy-binding` and walk away — terraform will drift.

### 3. The Reasoning Engine deploys to THE FORUM's project, but RUNS AS this agent's per-agent SA

There are two projects in play:

- **The Forum's project** (`FORUM_PROJECT_ID` = `GOOGLE_CLOUD_PROJECT` in `.env`): Where The Forum runs, and where every agent's Reasoning Engine physically lives. Administratively centralized so The Forum can list/route to all agents.
- **This agent's own project** (`AGENT_PROJECT_ID` in `.env` = `project_id` in `terraform/terraform.tfvars`): Where the per-agent SA, secrets, and ADK staging bucket live.

The Reasoning Engine's runtime identity is the per-agent SA (`BOT_ACCOUNT_ID@AGENT_PROJECT_ID.iam.gserviceaccount.com`), set via `.agent_engine_config.json`'s `service_account` field. **That's the SA you share Google Docs / Sheets / Drive files with** — not the Forum's compute SA, and not your own user account. Sharing with anyone else won't grant the deployed agent access.

**Do not add comment fields to `.agent_engine_config.json`.** ADK validates the file with `AgentEngineConfig`, a Pydantic model with `extra=forbid`, so any unknown key (including `_comment` or anything else not in the schema) fails the deploy at validation time with a misleading error long before reaching Vertex AI. Keep the file minimal — currently just `service_account`. If you need to add another field, verify it's in ADK's schema first (`src/google/adk/cli/cli_deploy.py` and the underlying `agent_engines.create(config=...)` shape).

The cross-project IAM that makes this work is provisioned by terraform: three of the Forum's Vertex AI service agents (`gcp-sa-aiplatform`, `gcp-sa-aiplatform-re`, and `gcp-sa-aiplatform-cc`) each get `roles/iam.serviceAccountTokenCreator` on the per-agent SA (to mint runtime tokens) and `roles/storage.objectViewer` on the staging bucket (to fetch the packaged agent code at cold start). The most load-bearing of the three is `-cc` (the custom container runtime that emulates the metadata server inside the engine) — if its binding is missing the engine deploys cleanly and then 500s on the first invocation with the cryptic "Compute Engine Metadata server unavailable. Response status: 500." Don't deploy the Reasoning Engine to the agent's own project — that wastes the cross-project IAM and means The Forum can't see your agent in its routing lookups.

**Cross-project IAM dependency:** The Forum project must have the Vertex AI service identities provisioned before this agent's terraform can apply, because the IAM bindings reference all three service-agent emails (`service-${FORUM_PROJECT_NUMBER}@gcp-sa-aiplatform[.|-re|-cc].iam.gserviceaccount.com`), which only auto-exist once Vertex AI has been used in that project. **`get_started_linux.sh` handles this automatically** (phase 5 runs `gcloud beta services identity create --service=aiplatform.googleapis.com --project=$FORUM_PROJECT_ID` — a single call provisions all three sub-agents and is idempotent, no-op if they already exist). If you skipped the bootstrap or are applying terraform by hand and hit a "principal does not exist" error on `engine_token_creator[...]` or `engine_staging_reader[...]`, run that same gcloud command and re-apply. The Forum's admin needs `roles/serviceusage.serviceUsageAdmin` on the Forum project to run it.

**Cross-project SA usage org policy (two overrides — only one is in this repo):** GCP enforces `constraints/iam.disableCrossProjectServiceAccountUsage` by default (and the default has been *enforced* on all orgs created since Sep 2024). The constraint applies on BOTH sides of cross-project SA usage:

- The project that *owns* the SA — must release it for use elsewhere.
- The project where the SA is being *attached* — must accept external SAs.

Skipping either side produces the same failure: `adk deploy agent_engine` creates the Reasoning Engine cleanly, and then every metadata-server token request at runtime returns 500 with no useful log line — the engine never serves a single user message.

This template's terraform handles the **agent-side override** (`google_project_organization_policy.allow_cross_project_sa`, on `var.project_id`) — that's the override for the SA's home project. The operator needs `roles/orgpolicy.policyAdmin` on the agent's own project for that resource to apply.

The **Forum-side override** is the Forum operator's responsibility and belongs in the Forum's own terraform, NOT in this repo. The agent template shouldn't manage org policies on a project it doesn't own — each agent reaching into the Forum's project to set the same policy would be both a permission overreach (every agent operator would need `orgpolicy.policyAdmin` on the Forum) and a footgun (any agent's `terraform destroy` would revert the policy and break every other agent). The Forum operator should add this once to the Forum's terraform:

```hcl
# Reasoning Engines created here run as per-agent SAs from other projects.
# Without disabling this constraint, those cross-project SAs cannot be
# attached at runtime and the engine 500s on every metadata token request.
resource "google_project_organization_policy" "allow_cross_project_sa_runtime" {
  project    = "your-forum-project-id"  # or whatever variable the Forum uses
  constraint = "constraints/iam.disableCrossProjectServiceAccountUsage"

  boolean_policy {
    enforced = false
  }
}
```

If you're standing up the very first agent against a fresh Forum project and the metadata-server 500 happens at runtime, the Forum-side override is the likely cause — ask the Forum operator to add the snippet above.

**Per-agent SA needs workload roles on the Forum project:** Cross-project impersonation gets the SA *into* the Forum project as a runtime identity, but the SA still has zero permissions there by default. Without baseline roles, the Reasoning Engine deploys cleanly and then 500s on the first user message — the runtime metadata server can't mint tokens for the SA, no useful log line, just a hung first reply. The terraform grants the per-agent SA four roles on the Forum project via `google_project_iam_member.engine_runtime_roles` (`aiplatform.user`, `logging.logWriter`, `monitoring.metricWriter`, `cloudtrace.agent`) — the same set the Forum's default compute SA has, which is why the old shared-SA setup "just worked." The operator running `terraform apply` needs `roles/resourcemanager.projectIamAdmin` on the Forum project to grant these. If you hit PERMISSION_DENIED on `engine_runtime_roles[...]`, ask the Forum's admin to either grant you that role or apply the bindings manually (the resource's comment in `terraform/main.tf` includes the exact gcloud commands).

### 4. Always use `deploy_and_update.sh` to deploy

Don't run `adk deploy agent_engine` directly. The script does blue/green deploy + smoke test + Firestore registration + stale-session cleanup + old-engine deletion. Skipping the script means:

- Firestore still points at the old Reasoning Engine → users get the old agent.
- No smoke test → a broken deploy goes live and there's no quick rollback.
- Old engine isn't cleaned up → wasted spend.

### 5. Platforms are registered by auto-detection — don't hand-edit Firestore

`register_agent.py` (which `deploy_and_update.sh` calls) detects enabled platforms by probing Secret Manager for the expected secret IDs (`{bot_account_id}-slack-token`, `{bot_account_id}-telegram-token`, etc.). To enable a new platform:

1. Uncomment its section in `terraform/main.tf`.
2. `terraform apply`.
3. Populate the secret value via `gcloud secrets versions add`.
4. Re-run `./deploy_and_update.sh`.

Don't add platform entries directly in the Firestore Console. They'll be overwritten on the next deploy.

### 6. User IDs are names — not platform IDs

The Forum sends the user's actual name (e.g. `"Alice Chen"`) as `user_id` to the agent, and prefixes incoming messages with `[From: Name]` (or `[From: Name | platform_id: ...]` for scheduled jobs). Your agent should treat `user_id` as a human name, not a Slack `U...` ID, Telegram numeric ID, etc. This is how cross-platform identity works: the same person on Slack and Telegram gets the same `user_id`.

### 7. The scheduler MCP URL must have a trailing slash

When wiring `MCPToolset(StreamableHTTPConnectionParams(url=...))`, the URL must end in `/api/v1/mcp/scheduler/` — with the trailing slash. The Forum's FastAPI route is registered as `/scheduler/`; the bare form 307-redirects POST → GET, which silently breaks the MCP JSON-RPC handshake.

### 8. Multimodal input requires explicit handling

If your agent should accept images from users, you must override the input handling to extract the `images` parameter The Forum sends alongside `message`. The default ADK `Agent` ignores it and produces empty responses on image messages. See [The Forum's `FOR_AGENT_DEVELOPERS.md` §"Receiving Images from Slack"](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md#receiving-images-from-slack) for the pattern.

### 9. Don't reach into The Forum repo to change The Forum's code

This repo and The Forum repo coordinate over Firestore documents and Secret Manager — never via shared code. If you find yourself wanting to modify The Forum's behavior from here, you're solving the wrong problem. Either:

- Use The Forum's existing extension points (the scheduler MCP, the Firestore platform array, the per-agent webhook routes).
- Open a PR against The Forum.

### 10. Sessions are stateful — clear them when prompts change

The Forum's Firestore `sessions` collection holds running conversations. If you change `agent.py`'s prompt in a way that's incompatible with mid-conversation state, sessions from before the change can produce weird responses. `deploy_and_update.sh` step 5 clears stale sessions automatically — let it do its job.

The template relies on managed Agent Engine sessions plus the Forum's Firestore — you don't run your own session store, so nothing here needs tending. If you ever swap in a custom `BaseSessionService` (e.g. your own SQL backend), be aware its persisted session schema is tied to the ADK version that wrote it; don't point a different ADK version at an existing custom session table without confirming the schema still matches.

### 11. `get_started_linux.sh` is single-use and self-deletes

The bootstrap script runs once at repo setup and deletes itself. Don't try to re-run it to "regenerate" `.env` or `terraform.tfvars` — edit those files directly, or modify them via `terraform.tfvars.example` + a fresh clone.

### 12. Model calls are forced to the `global` endpoint at import time — keep it there

`agent.py` sets `os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'` as its very first statement, *above* the `google.adk` / `google.genai` imports. This is deliberate and load-bearing: the Reasoning Engine deploys to a regional location (`us-central1`), but the Gemini preview models the template defaults to are only served on the `global` endpoint. The Google libraries read `GOOGLE_CLOUD_LOCATION` at import, so the override has to come before they're imported. If you move that line below the imports — or drop it — preview models start failing with `404 / NOT_FOUND` while regional models keep working, which makes it look like a model-name typo rather than an endpoint problem.

## Building your agent

When you're filling in actual agent behavior:

### Defining the agent's purpose

`agent.py`'s `root_agent` has three fields that shape the LLM's behavior:

- `description`: a short summary of what the agent does — shown to other agents that use this one as a sub-agent.
- `instruction`: the system prompt. This is where the bulk of agent behavior lives.
- `tools`: the list of `FunctionTool` / `AgentTool` / `MCPToolset` the LLM can call.

Replace the stub `STUB_INSTRUCTION` with your real prompt. Keep prompts specific about *what tools to call and when*, especially if the agent has multiple tools that overlap.

### Adding a function tool

1. Define the function in `custom_functions.py` with a clear docstring (the LLM sees the docstring as the tool description).
2. Import it in `agent.py`: `from .custom_functions import my_tool`.
3. Wrap and add: `tools=[FunctionTool(my_tool), ...]`.

### Adding a sub-agent

1. Define the sub-agent in `custom_agents.py` (its own `Agent(...)` with its own model + prompt).
2. Import it in `agent.py`: `from .custom_agents import my_subagent`.
3. Wrap and add: `tools=[AgentTool(agent=my_subagent), ...]`.

### Adding an MCP toolset

Two transports:

- **stdio** (most public MCP servers): `MCPToolset(connection_params=StdioServerParameters(command="npx", args=["-y", "@org/server"], env={...}))`. Requires `npx`/`uvx` in the Reasoning Engine container.
- **Streamable HTTP / SSE** (hosted servers, including The Forum's scheduler): `MCPToolset(connection_params=StreamableHTTPConnectionParams(url="...", headers={"X-API-Key": ...}))`.

The scheduler MCP wiring is already stubbed in `agent.py` — uncomment after you've completed the three-step setup: enable Section 6 in `terraform/main.tf`, `terraform apply`, then follow the provisioning instructions in [The Forum's `FOR_AGENT_DEVELOPERS.md` §"Scheduler MCP Server"](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md#scheduler-mcp-server). The post-setup README's "Add MCP toolsets" section walks through the same flow.

### Using external APIs that need a key

1. Add the secret container + IAM binding in `terraform/main.tf` (follow the pattern of the existing platform secrets).
2. `terraform apply`.
3. Populate the secret value: `echo -n "API_KEY_VALUE" | gcloud secrets versions add my-secret --data-file=- --project=$GOOGLE_CLOUD_PROJECT`.
4. Fetch in the agent code: `secret_utilities.get_secret_from_secret_manager(project_id, "my-secret")`. Do this at module load time so cold-start latency is paid once per container instance, not per request.

### Local development

`adk web` from the repo root spins up a local web UI for the agent (talks to Vertex AI for the model, runs `agent.py` locally). Useful for iterating on prompts and tools without a full Reasoning Engine deploy. The Forum routing/platform stuff is bypassed in this mode — to test platform integration end-to-end, you have to deploy.

Before deploying after a meaningful change, exercise the real agent in-process — `InMemoryRunner` driving `root_agent` with `RunConfig(streaming_mode=SSE)`, a couple of messages that actually trigger your tools. The deployed engine hides model/tool errors behind a generic "failed to start" or an empty reply (e.g. a 400 from a malformed tool-call history shows up only as "0 chunks"); the in-process runner surfaces the real exception. This is the fastest way to catch a bad tool wiring or model issue before it costs you a deploy cycle.

### Redeploying

`./deploy_and_update.sh`. See rule 4.

## Testing

The template ships a minimal sanity test in `test.md` (for template maintainers — gets deleted by `get_started_linux.sh` during initial setup). Once you're building your own agent, add your own tests under a `tests/` directory using `pytest` (already implied by `requirements.txt`). When you wire up CI, model it on The Forum's [`.github/workflows/ci.yml`](https://github.com/Comites-ai/the-forum/blob/main/.github/workflows/ci.yml).
