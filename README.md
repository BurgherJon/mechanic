# Mike the Mechanic

Mike manages your cars.

Built on **[The Forum](https://github.com/Comites-ai/the-forum)** — [Comites.ai](https://comites.ai)'s open-source middleware that routes messages from Slack, Google Chat, Telegram, and Discord to AI agents running on Vertex AI.

## What Mike does

Mike keeps a maintenance record for each of your vehicles in Google Drive:

- **Service history from receipts.** Photograph or upload a receipt; Mike reads the date, shop, work performed, and total off it, files the original in that car's Service Receipts folder, and adds a row to Repair History.
- **Odometer tracking.** Readings are recorded whenever you mention them, and used to project current mileage — which is what makes mileage-based reminders (oil change at 112,000) fire *before* they come due rather than after.
- **Documents.** Insurance and registration PDFs live in each car's Documents folder, and Mike chases you for them when they're missing or expiring.
- **A weekly check**, Saturdays at 10am, one scheduled job per vehicle.

## Fleet data model

One Drive folder per vehicle, inside the folder named by `FLEET_FOLDER_ID`:

```
<Fleet folder>/
├── Memory                     <- the agent's memory doc (not a vehicle)
└── <Vehicle name>/
    ├── <Vehicle name>         <- workbook, 4 tabs
    ├── Documents/             <- Insurance.pdf, Registration.pdf
    └── Service Receipts/      <- receipts as originally uploaded
```

| Tab | Shape | Columns |
|---|---|---|
| Info | vertical key/value | Year, Manufacturer, Model, Vin, License Plate, Insurance Company, Policy Number, Insurance Expiration, Title, Registration Expiration, Projected Mileage |
| Repair History | table | Date, Description, Receipt File |
| Reminders | table | Date Required, Mileage Required, Task |
| Odometer Readings | table | Date, Reading |

The schema lives in [`fleet_utilities.py`](fleet_utilities.py) and is what `add_vehicle` stamps into every new car's workbook. Fields and columns are located **by label, not by position**, so reordering rows in the Sheets UI won't break anything. `Projected Mileage` is computed — don't edit it by hand.

## Google credentials

Mike talks to Drive and Sheets as a **real Google user**, via an OAuth refresh token in Secret Manager (`mike-the-mechanic-google-oauth`), minted by [`mint_oauth_token.py`](mint_oauth_token.py).

This is deliberate, and it is not how the rest of the agent authenticates. A service account has no Drive storage quota, and in Drive the *creator* owns the file — so the agent's SA can edit a workbook shared with it but cannot create a vehicle workbook, archive a receipt, or file a PDF. Those fail with `storageQuotaExceeded` (and, via the Sheets API, with a misleading "The caller does not have permission").

> **Mint the token for a dedicated secondary Google account — not the account that owns the Drive the fleet folder lives in.**
>
> The token carries the full `drive` scope, which cannot be narrowed to one folder. Whoever holds it can reach everything that account can reach. Share the fleet folder — and only the fleet folder — with a second account, and mint as that account. Then a leaked or misused token exposes the fleet and nothing else.
>
> The cost of that containment is that Mike's files are owned by the secondary account and count against its quota, so don't delete it. The permanent fix is a Shared Drive, where files are owned by the drive rather than the creator; tracked as PLAT-38.

To mint or rotate:

```bash
python mint_oauth_token.py --client-secret ~/client_secret.json \
  | gcloud secrets versions add mike-the-mechanic-google-oauth \
      --data-file=- --project=mike-the-mechanic-prod
```

Everything else — Secret Manager, Vertex AI, the memory doc, the scheduler MCP — continues to use the agent's own service account.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│   Slack                         │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  The Forum (Cloud Run, project: vertex-ai-middleware-prod)│
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Mike the Mechanic (Vertex AI Reasoning Engine) │
│  Project: mike-the-mechanic-prod                   │
│  SA:      mike-the-mechanic@mike-the-mechanic-prod.iam.gserviceaccount.com
└──────────────────────────────────────────────────────┘
```

## Local development

```bash
# In a venv with this repo's dependencies installed
adk web
```

That launches a local UI for chatting with the agent. The Forum routing is bypassed — to test the platform integration end-to-end, deploy.

## Deploy

```bash
./deploy_and_update.sh
```

The script does blue/green: deploys a new Reasoning Engine, smoke-tests it, updates The Forum's Firestore to point at the new engine, clears stale sessions, then deletes the old engine. Safe to re-run; if anything fails partway through, the old engine is untouched.

## Next steps

The agent works end-to-end as soon as you've deployed once (it'll respond as Junius Rusticus on whichever platform you enabled — proving the pipeline works before you write any real agent logic). From there, the typical buildup is:

### 1. Define what your agent does

Edit [`agent.py`](agent.py):

- Replace `STUB_INSTRUCTION` with the system prompt that defines your agent's persona, the tasks it handles, and how it should call tools.
- Update the `description` field — that's what shows up if another agent ever uses yours as a sub-agent, and it's what registration writes to Firestore.
- Pick the model that fits the work. `HIGH_QUALITY_AGENT_MODEL` in `.env` is what `root_agent` uses by default; `QUICK_AGENT_MODEL` is the convention for cheaper sub-agents.

### 2. Use the persistent memory that's already wired up

`get_agent_memory()` and `update_agent_memory()` are already registered in `root_agent.tools`. They read and write a Google Doc whose ID came from `AGENT_MEMORY_DOC_ID` in `.env` (you set this in the bootstrap; the doc is shared with this agent's runtime SA).

To use them, instruct the model in your prompt:

- *"Before every response, call `get_agent_memory` and use the contents to personalize your reply."*
- *"At the end of each session, call `update_agent_memory(...)` with the complete updated memory text — distill what you learned, don't just append."*

If you'd rather not use memory: delete the two `FunctionTool(get_agent_memory)` / `FunctionTool(update_agent_memory)` entries from `root_agent.tools` in `agent.py`, and leave `AGENT_MEMORY_DOC_ID` blank.

### 3. Add function tools

Edit [`custom_functions.py`](custom_functions.py):

- Define plain Python functions with clear docstrings (the docstring is shown to the LLM as the tool description).
- Type-hint the args and return value.
- Import and wrap in `agent.py`:
  ```python
  from .custom_functions import my_tool
  ...
  tools=[FunctionTool(my_tool), ...]
  ```

### 4. Add sub-agents (optional)

Edit [`custom_agents.py`](custom_agents.py):

- Define a separate `Agent(...)` with its own narrow prompt for specialized work (web search, classification, etc.).
- Wrap in `AgentTool(agent=my_subagent)` and add to `root_agent.tools`.

### 5. Add MCP toolsets (optional)

- For The Forum's hosted scheduler MCP: uncomment Section 6 in [`terraform/main.tf`](terraform/main.tf), `terraform apply`, then follow the three-step provisioning in [The Forum's `FOR_AGENT_DEVELOPERS.md` §"Scheduler MCP Server"](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md#scheduler-mcp-server). Then uncomment the `scheduler_toolset` block in [`agent.py`](agent.py) and add it to `root_agent.tools`.
- For other MCP servers (GitHub, Garmin, etc.): add directly in [`agent.py`](agent.py) using `MCPToolset(...)`. See the "Adding MCP Servers" section in [`FOR_AGENT_DEVELOPERS.md`](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md#adding-mcp-servers-to-your-agent-adk-native).

### 6. Add external API integration with secrets

1. Add the secret container + IAM binding for the Reasoning Engine SA in [`terraform/main.tf`](terraform/main.tf) (follow the pattern of the existing platform secrets).
2. `terraform apply`.
3. Populate the value: `echo -n "API_KEY" | gcloud secrets versions add my-secret --data-file=- --project=$GOOGLE_CLOUD_PROJECT`.
4. Read at module load in your tool code: `secret_utilities.get_secret_from_secret_manager(project_id, "my-secret")`.

### 7. Add image / multimodal support (optional)

The Forum forwards images alongside text in the `images` parameter. The default `Agent` doesn't process them — you need to override input handling. See [`FOR_AGENT_DEVELOPERS.md` §"Receiving Images"](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md#receiving-images-from-slack).

### 8. Enable additional platforms

To add Telegram once Slack is already working (or any other combination):

1. Uncomment the relevant section in [`terraform/main.tf`](terraform/main.tf).
2. `terraform apply` (creates the secret container + IAM binding).
3. Populate the token: `echo -n "TOKEN" | gcloud secrets versions add mike-the-mechanic-{platform}-token --data-file=- --project=$GOOGLE_CLOUD_PROJECT`.
4. For Telegram / Discord, do the platform-side webhook / Gateway setup per [`FOR_AGENT_DEVELOPERS.md`](https://github.com/Comites-ai/the-forum/blob/main/docs/FOR_AGENT_DEVELOPERS.md).
5. `./deploy_and_update.sh` — `register_agent.py` auto-detects the new secret and adds the platform to your Firestore record.

## Operating rules

See [AGENTS.md](AGENTS.md) for the invariants (where infrastructure changes go, how secrets work, how deploys work, etc.). Same rules apply whether you or an AI coding assistant is doing the work.

## License

This agent was bootstrapped from the [Comites.ai Agent Template](https://github.com/Comites-ai/agent-template), which is MIT-licensed. This repo ships without a license file, so it defaults to "all rights reserved" — add your own `LICENSE` if you intend to distribute it (keep it permissive with MIT/Apache-2.0, copyleft with AGPL-3.0, or proprietary — your choice). Separately, the Comites.ai [trademark policy](https://github.com/Comites-ai/agent-template/blob/main/TRADEMARK.md) still applies regardless of your code license: don't use "Comites", "The Forum", or related names in your project's name.

## Acknowledgements

Bootstrapped from the [Comites.ai Agent Template](https://github.com/Comites-ai/agent-template) (MIT) and runs on [The Forum](https://github.com/Comites-ai/the-forum) (AGPL-3.0).
