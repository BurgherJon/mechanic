"""
Custom function tools for your agent.

Each function in this file is wrappable in `google.adk.tools.FunctionTool`
and added to `root_agent.tools` in `agent.py`. The function's docstring
is shown to the LLM as the tool description, so write clear docstrings
that explain what the tool does, what arguments it takes, and what it
returns.

Pattern for adding a new tool:

    def my_tool(some_arg: str) -> dict:
        \"\"\"
        Short one-liner that explains what this does.

        Args:
            some_arg: Description.

        Returns:
            Description of the return structure.
        \"\"\"
        return {"result": some_arg}

Then in agent.py:

    from .custom_functions import my_tool
    ...
    tools=[FunctionTool(my_tool)],
"""
import os
from datetime import date as _date, timedelta
from typing import Any, Dict, List

from .docs_utilities import get_docs_connector
from . import fleet_utilities as fleet
from .fleet_utilities import get_fleet_connector


# ============================================================================
# Persistent memory via Google Docs (wired up by default in agent.py)
#
# The template ships with these two memory tools already registered in
# `root_agent.tools`. The Google Doc ID comes from AGENT_MEMORY_DOC_ID in
# .env (get_started_linux.sh prompts for it). The doc must be shared
# (Editor access) with the per-agent SA email — that's the SA the
# Reasoning Engine runs as (see .agent_engine_config.json).
#
# If you don't want memory:
#   1. Remove the two FunctionTool entries from root_agent.tools in agent.py
#   2. Delete these two functions (or leave them — they'll just go unused)
#   3. Leave AGENT_MEMORY_DOC_ID unset (the tools no-op via the raise)
# ============================================================================

def get_agent_memory() -> str:
    """
    Retrieve the agent's persistent memory from the configured Google Doc.

    The doc ID comes from AGENT_MEMORY_DOC_ID in .env. The doc must be
    shared (Editor access) with the agent's runtime service account
    (BOT_ACCOUNT_ID@AGENT_PROJECT_ID.iam.gserviceaccount.com).

    Returns:
        The full text content of the memory document. May be an empty
        string if the doc has no content yet.

    Raises:
        ValueError: if AGENT_MEMORY_DOC_ID is not set in the environment.
        googleapiclient.errors.HttpError (403): if the doc hasn't been
            shared with the agent's runtime service account.
    """
    doc_id = os.environ.get("AGENT_MEMORY_DOC_ID")
    if not doc_id:
        raise ValueError(
            "AGENT_MEMORY_DOC_ID is not set. Either set it in .env (and on the "
            "deployed Reasoning Engine) or remove the memory tools from "
            "root_agent.tools in agent.py."
        )
    return get_docs_connector().read_doc(doc_id)


def update_agent_memory(updated_memory: str) -> Dict[str, Any]:
    """
    Replace the agent's persistent memory with the provided text.

    Use this at the end of a session (or whenever the agent has new
    information worth persisting) to write back updated notes. The
    write replaces the entire document body.

    Args:
        updated_memory: Complete new memory document text. This replaces
            the existing content — pass the full updated memory, not just
            the changes.

    Returns:
        API response confirming the update.

    Raises:
        ValueError: if AGENT_MEMORY_DOC_ID is not set in the environment.
        googleapiclient.errors.HttpError (403): if the doc hasn't been
            shared with the agent's runtime service account.
    """
    doc_id = os.environ.get("AGENT_MEMORY_DOC_ID")
    if not doc_id:
        raise ValueError(
            "AGENT_MEMORY_DOC_ID is not set. Either set it in .env (and on the "
            "deployed Reasoning Engine) or remove the memory tools from "
            "root_agent.tools in agent.py."
        )
    return get_docs_connector().write_doc(doc_id, updated_memory)


# ============================================================================
# Fleet tools
#
# These wrap fleet_utilities for the LLM. Two conventions throughout:
#
#   1. They return {"error": "..."} instead of raising for anything the user
#      can fix (unknown vehicle, missing reading). An exception surfaces to
#      the user as "I appear to have a broken tool"; a returned error lets
#      Mike explain the actual problem and ask for what he needs.
#   2. Dates are accepted as free text and normalized here, so the model
#      never has to guess a format.
# ============================================================================

# How far ahead the weekly check looks for upcoming obligations.
LOOKAHEAD_DAYS = 62
# An odometer reading older than this is considered stale and worth asking about.
ODOMETER_STALE_DAYS = 60
# A vehicle with no service logged in this long gets a gentle nudge.
SERVICE_NUDGE_DAYS = 365


def _resolve(vehicle_name: str):
    """Look up a vehicle, returning (vehicle, error_dict)."""
    conn = get_fleet_connector()
    vehicle = conn.find_vehicle(vehicle_name)
    if not vehicle:
        known = [v["name"] for v in conn.list_vehicles()]
        return None, {"error": f"No vehicle matching {vehicle_name!r}.",
                      "known_vehicles": known}
    if not vehicle.get("spreadsheet_id"):
        return None, {"error": f"{vehicle['name']} has no workbook in its folder."}
    return vehicle, None


def _download(gcs_uri: str) -> bytes:
    """
    Fetch an object The Forum staged for us in its bucket.

    The quota project must be pinned to the agent's own project. The engine
    runs inside the Forum's project, so plain ADC bills GCS calls there —
    and this agent's SA has `serviceusage.services.use` only on its own
    project (see the serviceUsageConsumer binding in terraform). Without
    the override the download fails 403 "does not have
    serviceusage.services.use access", which reads like a bucket
    permission problem and is not one; the object ACL is never consulted.
    """
    import google.auth
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {gcs_uri}")
    bucket, _, blob = gcs_uri[5:].partition("/")

    credentials, _ = google.auth.default()
    agent_project = os.environ.get("AGENT_PROJECT_ID")
    if agent_project and hasattr(credentials, "with_quota_project"):
        credentials = credentials.with_quota_project(agent_project)
    client = storage.Client(project=agent_project or None, credentials=credentials)
    return client.bucket(bucket).blob(blob).download_as_bytes()


def list_vehicles() -> Dict[str, Any]:
    """
    List every vehicle in the fleet.

    Returns:
        {"vehicles": [{"name": ...}, ...]}. An empty list means no cars
        have been added yet — offer to add one.
    """
    try:
        return {"vehicles": [{"name": v["name"]} for v in get_fleet_connector().list_vehicles()]}
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
        return {"error": f"Could not list vehicles: {exc}"}


def add_vehicle(name: str) -> Dict[str, Any]:
    """
    Create a new vehicle: its folder, Documents and Service Receipts
    subfolders, and a workbook with the standard four tabs.

    The vehicle starts with every Info field blank. After calling this,
    ask the user for the details (year, manufacturer, model, VIN, plate,
    insurance company, policy number, insurance expiration, title,
    registration expiration) and record them with update_vehicle_detail.
    Then offer to take their insurance document, registration document,
    and any past service receipts.

    Args:
        name: What to call the car, e.g. "Tacoma" or "Sarah's Civic".

    Returns:
        The created vehicle, including a link to its new workbook.
    """
    try:
        created = get_fleet_connector().create_vehicle(name)
        return {"created": created["name"],
                "spreadsheet_url": created["spreadsheet_url"],
                "next_step": "Ask the user for the Info fields, then for the "
                             "insurance and registration documents."}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not create the vehicle: {exc}"}


def remove_vehicle(name: str) -> Dict[str, Any]:
    """
    Move a vehicle's entire folder to the Drive trash.

    This removes the workbook, documents, and receipts together. Always
    confirm with the user before calling this, naming the car explicitly.
    Recoverable from the Drive trash for 30 days.

    Args:
        name: The vehicle to remove.
    """
    try:
        removed = get_fleet_connector().delete_vehicle(name)
        return {"removed": removed["name"],
                "note": "Moved to Drive trash; recoverable for 30 days."}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        # Only a file's owner can trash it. Vehicles created by a different
        # account (or added to the fleet folder by hand) can be read and
        # edited but not removed, and the raw API message for that is
        # "insufficient permissions", which invites a pointless retry.
        if "insufficientFilePermissions" in str(exc) or "sufficient permissions" in str(exc):
            return {"error": f"I can edit {name}'s records but I'm not the owner of "
                             f"its folder, so I can't delete it. Whoever owns that "
                             f"folder in Drive needs to remove it."}
        return {"error": f"Could not remove the vehicle: {exc}"}


def get_vehicle_details(name: str) -> Dict[str, Any]:
    """
    Everything known about one vehicle: its Info fields, current mileage
    estimate, which required fields and documents are missing, and counts
    of service records and reminders.

    Args:
        name: The vehicle to look up.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    conn = get_fleet_connector()
    try:
        info = conn.read_info(vehicle)
        readings = conn.list_odometer(vehicle)
        missing_fields = [f for f in fleet.REQUIRED_INFO_FIELDS if not info.get(f, "").strip()]
        return {
            "name": vehicle["name"],
            "info": info,
            "projected_mileage": fleet.project_mileage(readings),
            "last_odometer": (
                {"reading": readings[-1]["reading"],
                 "date": fleet.format_date(readings[-1]["date"])}
                if readings else None
            ),
            "missing_info_fields": missing_fields,
            "missing_documents": conn.missing_documents(vehicle),
            "service_record_count": len(conn.list_repairs(vehicle)),
            "reminder_count": len(conn.list_reminders(vehicle)),
            "spreadsheet_url": vehicle.get("spreadsheet_url", ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not read {vehicle['name']}: {exc}"}


def update_vehicle_detail(name: str, field: str, value: str) -> Dict[str, Any]:
    """
    Set one field on a vehicle's Info tab.

    Call once per field. Use the exact field labels: Year, Manufacturer,
    Model, Vin, License Plate, Insurance Company, Policy Number,
    Insurance Expiration, Title, Registration Expiration.

    Do not set Projected Mileage by hand — it is computed from odometer
    readings by record_odometer and the weekly check.

    Args:
        name: The vehicle.
        field: The Info label to set.
        value: The value to write. Dates as M/D/YYYY.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    try:
        get_fleet_connector().update_info(vehicle, {field: value})
        return {"updated": {field: value}, "vehicle": vehicle["name"]}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not update {field}: {exc}"}


def record_odometer(name: str, reading: int, reading_date: str = "") -> Dict[str, Any]:
    """
    Record an odometer reading and refresh the vehicle's mileage estimate.

    Call this any time the user mentions their mileage, even in passing —
    readings are what make mileage-based reminders work.

    Args:
        name: The vehicle.
        reading: The odometer value, e.g. 108070.
        reading_date: When it was read. Defaults to today. Accepts
            M/D/YYYY or YYYY-MM-DD.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    when = fleet.parse_date(reading_date) or fleet.today()
    try:
        conn = get_fleet_connector()
        conn.add_odometer(vehicle, when, int(reading))
        readings = conn.list_odometer(vehicle)
        projected = fleet.project_mileage(readings)
        if projected is not None:
            conn.update_info(vehicle, {"Projected Mileage": str(projected)})
        rate = fleet.daily_mileage_rate(readings)
        return {
            "recorded": {"vehicle": vehicle["name"], "reading": int(reading),
                         "date": fleet.format_date(when)},
            "projected_mileage": projected,
            "miles_per_week": round(rate * 7, 1) if rate else None,
            "readings_on_file": len(readings),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not record the reading: {exc}"}


def read_uploaded_file(gcs_uri: str, mime_type: str, purpose: str = "receipt") -> Dict[str, Any]:
    """
    Read a file the user just uploaded and extract what it says.

    When a user's message begins with a token like
    `[IMAGE: gs://... | image/jpeg]` or `[FILE: gs://... | application/pdf]`,
    pass that URI and MIME type here. Handles photos and PDFs alike.

    For a service receipt this returns the service date, vendor, odometer
    if printed, work performed, and total — use those to fill in
    log_service rather than asking the user to retype them.

    Args:
        gcs_uri: The gs:// URI from the message token.
        mime_type: The MIME type from the message token.
        purpose: "receipt", "insurance", "registration", or "other" —
            controls what the extraction focuses on.
    """
    try:
        from google import genai
        from google.genai import types
        data = _download(gcs_uri)
    except Exception as exc:  # noqa: BLE001
        # Report the failure verbatim. An earlier version asserted this was
        # always a missing storage.objectViewer grant, which sent debugging
        # down the wrong path when the real cause was the quota project.
        return {"error": f"Could not fetch the uploaded file: {exc}"}

    prompts = {
        "receipt": (
            "This is a vehicle service receipt or invoice. Extract, as plain "
            "labelled lines: service date; vendor/shop name; odometer reading "
            "if printed; every service or repair performed (itemized); parts "
            "replaced; and the total charged. If a value is not present, say "
            "'not shown' rather than guessing. Quote the date exactly as printed."
        ),
        "insurance": (
            "This is a vehicle insurance document. Extract: insurance company, "
            "policy number, effective date, expiration date, insured vehicle "
            "(year/make/model), and VIN if shown. Say 'not shown' for anything absent."
        ),
        "registration": (
            "This is a vehicle registration document. Extract: registered owner, "
            "plate number, VIN, registration expiration date, and the issuing "
            "state. Say 'not shown' for anything absent."
        ),
    }
    prompt = prompts.get(purpose, "Describe this document and transcribe any text it contains.")

    try:
        from .model_utils import generate_vision

        # VISION_MODEL (Gemini, global endpoint) with transient retry —
        # deliberately decoupled from the root agent's Claude model.
        extracted = generate_vision([
            types.Part.from_bytes(data=data, mime_type=mime_type),
            types.Part.from_text(text=prompt),
        ])
        return {"extracted": extracted,
                "gcs_uri": gcs_uri, "mime_type": mime_type}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not read the file: {exc}"}


def log_service(name: str, description: str, service_date: str = "",
                receipt_gcs_uri: str = "", receipt_mime_type: str = "",
                odometer_reading: int = 0) -> Dict[str, Any]:
    """
    Add a row to a vehicle's Repair History, archiving the receipt.

    If a receipt URI is given, the file is copied out of The Forum's
    staging bucket into that vehicle's Service Receipts folder and linked
    from the new row. This matters: the staged copy is deleted after a
    day, so a receipt that isn't archived during this conversation is
    gone for good.

    Args:
        name: The vehicle.
        description: What was done. Include the vendor and the total.
        service_date: Date on the receipt. Defaults to today.
        receipt_gcs_uri: gs:// URI from the upload token, if any.
        receipt_mime_type: MIME type from the upload token.
        odometer_reading: Odometer printed on the receipt, if any — pass
            it and it gets logged as a reading too.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    conn = get_fleet_connector()
    when = fleet.parse_date(service_date) or fleet.today()
    result: Dict[str, Any] = {"vehicle": vehicle["name"],
                              "date": fleet.format_date(when)}

    receipt_link = ""
    if receipt_gcs_uri:
        try:
            data = _download(receipt_gcs_uri)
            ext = {"application/pdf": "pdf", "image/png": "png",
                   "image/heic": "heic", "image/webp": "webp"}.get(receipt_mime_type, "jpg")
            safe = "".join(c for c in description[:40] if c.isalnum() or c in " -_").strip()
            filename = f"{when.isoformat()} {safe or 'service'}.{ext}"
            stored = conn.store_file(vehicle, data, filename,
                                     receipt_mime_type or "application/octet-stream",
                                     fleet.RECEIPTS_FOLDER)
            receipt_link = stored.get("webViewLink", "")
            result["receipt_archived_as"] = stored.get("name")
        except Exception as exc:  # noqa: BLE001
            # Log the service anyway — losing the row as well as the file
            # would be the worse outcome.
            result["receipt_warning"] = f"Could not archive the receipt: {exc}"

    try:
        conn.add_repair(vehicle, fleet.format_date(when), description, receipt_link)
        result["logged"] = description
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not write the service record: {exc}"}

    if odometer_reading:
        odo = record_odometer(vehicle["name"], int(odometer_reading), fleet.format_date(when))
        result["odometer"] = odo.get("recorded") or odo.get("error")
    return result


def store_vehicle_document(name: str, gcs_uri: str, mime_type: str,
                           document_name: str) -> Dict[str, Any]:
    """
    File an uploaded document into a vehicle's Documents folder.

    Use for insurance cards and registration documents, which the weekly
    check expects to find there. Name them "Insurance" or "Registration"
    so the check recognizes them.

    Args:
        name: The vehicle.
        gcs_uri: gs:// URI from the upload token.
        mime_type: MIME type from the upload token.
        document_name: What to call it, e.g. "Insurance" or "Registration".
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    try:
        data = _download(gcs_uri)
        ext = {"application/pdf": "pdf", "image/png": "png"}.get(mime_type, "jpg")
        stored = get_fleet_connector().store_file(
            vehicle, data, f"{document_name}.{ext}",
            mime_type or "application/octet-stream", fleet.DOCUMENTS_FOLDER)
        return {"stored": stored.get("name"), "vehicle": vehicle["name"],
                "url": stored.get("webViewLink", "")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not file the document: {exc}"}


def list_service_history(name: str) -> Dict[str, Any]:
    """
    Every service and repair recorded for a vehicle, newest first.

    Args:
        name: The vehicle.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    try:
        rows = get_fleet_connector().list_repairs(vehicle)
        rows.sort(key=lambda r: fleet.parse_date(r.get("Date")) or _date.min, reverse=True)
        return {"vehicle": vehicle["name"], "service_records": rows}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not read the service history: {exc}"}


def list_reminders(name: str) -> Dict[str, Any]:
    """
    The recurring and one-off items tracked for a vehicle.

    Args:
        name: The vehicle.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    try:
        return {"vehicle": vehicle["name"],
                "reminders": get_fleet_connector().list_reminders(vehicle)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not read the reminders: {exc}"}


def add_reminder(name: str, task: str, date_required: str = "",
                 mileage_required: str = "") -> Dict[str, Any]:
    """
    Add an item to a vehicle's Reminders tab.

    Give a date, a mileage, or both. A mileage-based reminder (e.g. an oil
    change at 112000) is projected against the vehicle's mileage trend, so
    the weekly check can raise it before it comes due.

    Args:
        name: The vehicle.
        task: What needs doing, e.g. "Oil change" or "Renew registration".
        date_required: Due date as M/D/YYYY, if date-based.
        mileage_required: Odometer value it's due at, if mileage-based.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    try:
        when = fleet.parse_date(date_required)
        get_fleet_connector().add_reminder(
            vehicle, task, fleet.format_date(when) if when else "",
            str(mileage_required or ""))
        return {"added": task, "vehicle": vehicle["name"],
                "date_required": fleet.format_date(when) if when else "",
                "mileage_required": mileage_required or ""}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not add the reminder: {exc}"}


def weekly_check(name: str) -> Dict[str, Any]:
    """
    Run the Saturday review for one vehicle and report what needs attention.

    Checks, in order: reminders coming due by date in the next two months;
    mileage-based reminders projected to come due in that window; whether
    the odometer reading is stale; whether insurance or registration is
    expiring or already expired; whether those documents are on file;
    which required Info fields are blank; and whether it has been a year
    since any service was logged. Also refreshes Projected Mileage.

    Returns a "findings" list. An empty list means nothing needs attention —
    say so briefly rather than inventing something to report.

    Args:
        name: The vehicle to check.
    """
    vehicle, err = _resolve(name)
    if err:
        return err
    conn = get_fleet_connector()
    now = fleet.today()
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)
    findings: List[Dict[str, Any]] = []

    try:
        info = conn.read_info(vehicle)
        readings = conn.list_odometer(vehicle)
        projected = fleet.project_mileage(readings)

        # Keep the sheet's estimate current — this is the weekly refresh.
        if projected is not None:
            conn.update_info(vehicle, {"Projected Mileage": str(projected)})

        # 1 + 2. Reminders, by date and by projected mileage.
        for row in conn.list_reminders(vehicle):
            task = (row.get("Task") or "").strip()
            if not task:
                continue
            due = fleet.parse_date(row.get("Date Required"))
            if due and due <= horizon:
                findings.append({
                    "type": "reminder_overdue" if due < now else "reminder_due",
                    "task": task, "due": fleet.format_date(due),
                    "days_away": (due - now).days,
                })
                continue
            target = fleet.parse_int(row.get("Mileage Required"))
            if target and projected is not None:
                eta = fleet.date_mileage_due(readings, target, now)
                if projected >= target:
                    findings.append({"type": "reminder_overdue", "task": task,
                                     "due_at_mileage": target,
                                     "projected_mileage": projected})
                elif eta and eta <= horizon:
                    findings.append({"type": "reminder_due", "task": task,
                                     "due_at_mileage": target,
                                     "projected_mileage": projected,
                                     "estimated_date": fleet.format_date(eta)})

        # 3. Stale or absent odometer.
        if not readings:
            findings.append({"type": "odometer_missing",
                             "detail": "No odometer readings on file yet."})
        else:
            age = (now - readings[-1]["date"]).days
            if age > ODOMETER_STALE_DAYS:
                findings.append({"type": "odometer_stale", "days_old": age,
                                 "last_reading": readings[-1]["reading"],
                                 "last_read_on": fleet.format_date(readings[-1]["date"])})

        # 4. Insurance and registration expiry.
        for label, what in (("Insurance Expiration", "insurance"),
                            ("Registration Expiration", "registration")):
            expires = fleet.parse_date(info.get(label, ""))
            if expires and expires <= horizon:
                findings.append({
                    "type": f"{what}_expired" if expires < now else f"{what}_expiring",
                    "expires": fleet.format_date(expires),
                    "days_away": (expires - now).days,
                })

        # 5. Required documents on file.
        for missing in conn.missing_documents(vehicle):
            findings.append({"type": "document_missing", "document": missing})

        # 6. Blank required Info fields.
        blank = [f for f in fleet.REQUIRED_INFO_FIELDS if not info.get(f, "").strip()]
        if blank:
            findings.append({"type": "info_incomplete", "fields": blank})

        # 7. Long gap since any service.
        repairs = conn.list_repairs(vehicle)
        dates = [d for d in (fleet.parse_date(r.get("Date")) for r in repairs) if d]
        if dates:
            gap = (now - max(dates)).days
            if gap > SERVICE_NUDGE_DAYS:
                findings.append({"type": "no_recent_service", "days_since": gap,
                                 "last_service": fleet.format_date(max(dates))})
        elif not repairs:
            findings.append({"type": "no_service_history",
                             "detail": "No service records logged yet."})

        return {"vehicle": vehicle["name"], "checked_on": fleet.format_date(now),
                "projected_mileage": projected, "findings": findings}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not complete the check for {vehicle['name']}: {exc}"}
