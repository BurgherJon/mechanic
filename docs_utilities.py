"""
Google Docs read/write helpers for agent memory.

The template wires up persistent memory by giving the agent read/write
access to a single Google Doc. The doc's contents become whatever the
agent puts there — notes about the user, conversation summaries, goals,
ongoing tasks, etc.

How auth works
--------------
We use **Application Default Credentials** (ADC). When the agent runs on
Vertex AI Agent Engine, ADC resolves to the project's default compute
service account (PROJECT_NUMBER-compute@developer.gserviceaccount.com).
When you run locally via `adk web`, ADC resolves to whatever
`gcloud auth application-default login` set up — usually your own
account.

There is **no service-account key** in Secret Manager for this. The doc
must be explicitly shared (Editor access) with whichever identity ADC
resolves to:

- Production (deployed): the compute SA email shown by `get_started_linux.sh`
- Local dev: the email from `gcloud config get-value account`

`get_started_linux.sh` prints the production SA email and walks you
through the share step.

What this module is NOT
-----------------------
- It doesn't manage authentication keys or rotation.
- It doesn't grant project-level IAM bindings — Docs API access is
  granted per-document via sharing, not via roles.
- It doesn't paginate or stream — `read_doc` and `write_doc` are simple
  one-shot calls suitable for memory documents under a few thousand
  words. For larger docs, use the Docs API directly.
"""
import os
from typing import Any, Dict, Optional

import google.auth
from google.auth.credentials import Credentials
from googleapiclient.discovery import build

from .secret_utilities import retry_on_transient_error


# Scopes the agent needs to read and write a Google Doc.
# `documents` covers the Docs API; `drive` is needed because Docs uses
# Drive permissions under the hood (and writes touch revision metadata).
_DOCS_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]


class GoogleDocsConnector:
    """
    Wrapper around the Google Docs API for read/write of a single doc.

    Construct once per agent process (use `get_docs_connector()` for a
    cached singleton). Methods retry transient network and 5xx errors
    via the `retry_on_transient_error` decorator from secret_utilities.
    """

    def __init__(self, credentials: Optional[Credentials] = None):
        if credentials is None:
            # ADC: returns whichever identity is configured in the runtime
            # environment (per-agent SA on Reasoning Engine, gcloud user
            # locally). The returned project_id is ignored — Docs API
            # doesn't care about project.
            credentials, _ = google.auth.default(scopes=_DOCS_SCOPES)
            # Bill Docs API calls to the agent's own project (where the
            # Docs API is enabled by terraform), not the Forum project
            # where the engine runs. The Forum doesn't enable Workspace
            # APIs — without this override, calls fail with
            # USER_PROJECT_DENIED. Scoped to this client only so Vertex
            # AI / Firestore / Secret Manager keep their default quota
            # project, which is correct for them.
            agent_project = os.environ.get("AGENT_PROJECT_ID")
            if agent_project:
                credentials = credentials.with_quota_project(agent_project)
        self._credentials = credentials
        self._docs_service = build("docs", "v1", credentials=credentials)

    @retry_on_transient_error()
    def read_doc(self, document_id: str) -> str:
        """
        Return the full plain text of a Google Doc.

        Strips out structure (headings, formatting) — returns just the
        concatenated text content. Good enough for memory docs; if your
        agent needs richer structure, call the Docs API directly.

        Raises googleapiclient.errors.HttpError with status 403 if the
        doc hasn't been shared with the agent's service account.
        """
        doc = self._docs_service.documents().get(documentId=document_id).execute()
        parts = []
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue
            for run in paragraph.get("elements", []):
                text_run = run.get("textRun")
                if text_run:
                    parts.append(text_run.get("content", ""))
        return "".join(parts)

    @retry_on_transient_error()
    def write_doc(self, document_id: str, content: str) -> Dict[str, Any]:
        """
        Replace the entire body of a Google Doc with `content`.

        Atomic from the user's perspective — the Docs API processes the
        delete+insert as a single batchUpdate. If the agent crashes
        between read and write, the doc keeps its previous contents.
        """
        # Find the current end-of-body index. Docs always reserves index 1
        # for the document start; the trailing newline at end_index is
        # part of the body and can't be deleted, so we delete up to
        # end_index - 1. An empty doc reports end_index = 2 (just the
        # undeletable newline) — skip the delete in that case, since a
        # (1, 1) range is rejected as empty by the Docs API.
        doc = self._docs_service.documents().get(documentId=document_id).execute()
        body_content = doc.get("body", {}).get("content", [])
        end_index = 1
        for element in body_content:
            if "endIndex" in element:
                end_index = element["endIndex"]

        requests = []
        if end_index > 2:
            requests.append({
                "deleteContentRange": {
                    "range": {"startIndex": 1, "endIndex": end_index - 1}
                }
            })
        requests.append({
            "insertText": {"location": {"index": 1}, "text": content}
        })

        return self._docs_service.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests},
        ).execute()


# Process-wide cached connector. The first call builds the service
# client (one network round-trip for OAuth metadata); subsequent calls
# reuse it.
_connector: Optional[GoogleDocsConnector] = None


def get_docs_connector() -> GoogleDocsConnector:
    """Return a process-wide cached `GoogleDocsConnector`."""
    global _connector
    if _connector is None:
        _connector = GoogleDocsConnector()
    return _connector
