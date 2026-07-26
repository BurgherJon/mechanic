"""
Drive + Sheets access for Mike's vehicle fleet.

Layout owned by this module (one folder per vehicle, inside FLEET_FOLDER_ID):

    <Fleet folder>/
        Memory                      <- the agent memory doc, not a vehicle
        <Vehicle name>/
            <Vehicle name>          <- spreadsheet, 4 tabs (see SCHEMA below)
            Documents/              <- Insurance.pdf, Registration.pdf, ...
            Service Receipts/       <- receipt images/PDFs as uploaded

`Info` is a vertical key/value tab (label in column A, value in column B) so
that adding a field never renumbers the others. Everything else is a normal
header-row table. Reads locate fields and columns *by label*, never by index,
so a human reordering rows in the UI cannot break the agent.

Authentication mirrors docs_utilities: ADC (the per-agent SA on Reasoning
Engine, your gcloud identity locally) with the quota project pinned to
AGENT_PROJECT_ID, because the Workspace APIs are enabled in the agent's
project rather than the Forum project where the engine actually runs.
"""
import io
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import google.auth
import google.auth.transport.requests
from google.auth.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

from .secret_utilities import retry_on_transient_error

_FLEET_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

FOLDER_MIME = "application/vnd.google-apps.folder"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"

# Subfolders created inside every vehicle folder.
DOCUMENTS_FOLDER = "Documents"
RECEIPTS_FOLDER = "Service Receipts"

# Documents the weekly check expects to find, matched case-insensitively on
# the filename stem so "Insurance.pdf" and "insurance 2026.pdf" both count.
REQUIRED_DOCUMENTS = ["Insurance", "Registration"]

# --- Workbook schema -------------------------------------------------------
# Field order here is the order written into a new vehicle's Info tab.
INFO_FIELDS = [
    "Year",
    "Manufacturer",
    "Model",
    "Vin",
    "License Plate",
    "Insurance Company",
    "Policy Number",
    "Insurance Expiration",
    "Title",
    "Registration Expiration",
    "Projected Mileage",
]

# Info fields Mike should chase the user for when they're blank. Projected
# Mileage is excluded: Mike computes it, the user never supplies it.
REQUIRED_INFO_FIELDS = [
    "Year", "Manufacturer", "Model", "Vin", "License Plate",
    "Insurance Company", "Policy Number", "Insurance Expiration",
    "Registration Expiration",
]

TAB_INFO = "Info"
TAB_REPAIRS = "Repair History"
TAB_REMINDERS = "Reminders"
TAB_ODOMETER = "Odometer Readings"

TABLE_TABS = {
    TAB_REPAIRS: ["Date", "Description", "Receipt File"],
    TAB_REMINDERS: ["Date Required", "Mileage Required", "Task"],
    TAB_ODOMETER: ["Date", "Reading"],
}

# Sheets renders these as M/D/YYYY; write the same so the columns stay
# visually consistent whether a human or Mike added the row.
DATE_FORMAT = "%-m/%-d/%Y"


def today() -> date:
    """Today in UTC. Callers wanting the user's local date pass it in."""
    return datetime.now(timezone.utc).date()


def parse_date(value: Any) -> Optional[date]:
    """
    Parse the date formats that realistically appear in these sheets.

    Returns None rather than raising: a malformed cell should degrade one
    row of a report, not fail the whole weekly check.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def format_date(value: Optional[date]) -> str:
    return value.strftime(DATE_FORMAT) if value else ""


def parse_int(value: Any) -> Optional[int]:
    """Pull an integer out of '108,070', '108070 mi', etc."""
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def _load_oauth_credentials() -> Credentials:
    """
    Build refreshing user credentials from the OAuth token in Secret Manager.

    Mike acts as a real Google user rather than as his own service account,
    because a service account has no Drive storage quota: in Drive the
    creator owns the file, so an SA can edit a workbook it has been given
    access to but cannot create one, upload a receipt, or file a PDF —
    all of those fail `storageQuotaExceeded`. The token is minted once by
    mint_oauth_token.py; google-auth refreshes the access token from the
    stored refresh token on each call.

    Falls back to ADC when the secret isn't configured, so read-only work
    still functions and the failure mode stays legible.
    """
    from google.oauth2.credentials import Credentials as UserCredentials

    project_id = (os.environ.get("AGENT_SECRET_PROJECT")
                  or os.environ.get("AGENT_PROJECT_ID")
                  or os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    secret_id = os.environ.get(
        "FLEET_OAUTH_SECRET_ID",
        f"{os.environ.get('BOT_ACCOUNT_ID', 'mike-the-mechanic')}-google-oauth")
    try:
        from .secret_utilities import get_secret_from_secret_manager
        raw = get_secret_from_secret_manager(project_id, secret_id)
        creds = UserCredentials.from_authorized_user_info(json.loads(raw), _FLEET_SCOPES)
        if not creds.valid:
            creds.refresh(google.auth.transport.requests.Request())
        return creds
    except Exception as exc:  # noqa: BLE001
        # Reads keep working under the service account; writes that create
        # files will fail with a storage-quota error that points here.
        import logging
        logging.getLogger(__name__).warning(
            "Falling back to ADC for Drive/Sheets — could not load OAuth "
            "credentials from secret %r in %r: %s. Creating files will fail "
            "with storageQuotaExceeded until mint_oauth_token.py has been run.",
            secret_id, project_id, exc)
        adc, _ = google.auth.default(scopes=_FLEET_SCOPES)
        return adc


class FleetConnector:
    """Drive + Sheets operations over the vehicle fleet."""

    def __init__(self, credentials: Optional[Credentials] = None,
                 fleet_folder_id: Optional[str] = None):
        if credentials is None:
            credentials = _load_oauth_credentials()
        # Applied whether or not the caller supplied credentials: the
        # Workspace APIs are enabled in the agent's project, not the Forum
        # project the engine runs in, and without the override the calls
        # bill the wrong project and come back 403 "caller does not have
        # permission" — which reads like an auth problem and isn't one.
        agent_project = os.environ.get("AGENT_PROJECT_ID")
        if agent_project and hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(agent_project)
        self._credentials = credentials
        self._drive = build("drive", "v3", credentials=credentials)
        self._sheets = build("sheets", "v4", credentials=credentials)
        self._fleet_folder_id = fleet_folder_id or os.environ.get("FLEET_FOLDER_ID", "")
        if not self._fleet_folder_id:
            raise RuntimeError(
                "FLEET_FOLDER_ID is not set. Point it at the Drive folder that "
                "holds the per-vehicle folders and share that folder with this "
                "agent's service account as Editor."
            )

    # --- Drive primitives ------------------------------------------------
    @retry_on_transient_error()
    def _children(self, parent_id: str, mime: Optional[str] = None) -> List[Dict[str, Any]]:
        q = f"'{parent_id}' in parents and trashed=false"
        if mime:
            q += f" and mimeType='{mime}'"
        out, page = [], None
        while True:
            resp = self._drive.files().list(
                q=q, fields="nextPageToken, files(id,name,mimeType,webViewLink,modifiedTime)",
                supportsAllDrives=True, includeItemsFromAllDrives=True,
                pageToken=page,
            ).execute()
            out.extend(resp.get("files", []))
            page = resp.get("nextPageToken")
            if not page:
                return out

    @retry_on_transient_error()
    def _create_folder(self, name: str, parent_id: str) -> str:
        meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
        return self._drive.files().create(
            body=meta, fields="id", supportsAllDrives=True).execute()["id"]

    @retry_on_transient_error()
    def _upload_bytes(self, data: bytes, name: str, mime_type: str,
                      parent_id: str) -> Dict[str, str]:
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)
        created = self._drive.files().create(
            body={"name": name, "parents": [parent_id]},
            media_body=media, fields="id,name,webViewLink",
            supportsAllDrives=True,
        ).execute()
        return created

    # --- Vehicle discovery -----------------------------------------------
    def list_vehicles(self) -> List[Dict[str, Any]]:
        """
        Every vehicle folder in the fleet root.

        Only folders count, so the Memory doc sitting alongside them is
        ignored without needing to special-case its name.
        """
        vehicles = []
        for folder in self._children(self._fleet_folder_id, FOLDER_MIME):
            vehicles.append({
                "name": folder["name"],
                "folder_id": folder["id"],
                "url": folder.get("webViewLink", ""),
            })
        return sorted(vehicles, key=lambda v: v["name"].lower())

    def find_vehicle(self, name: str) -> Optional[Dict[str, Any]]:
        """Case-insensitive lookup, tolerating partial names ('tacoma')."""
        wanted = (name or "").strip().lower()
        if not wanted:
            return None
        vehicles = self.list_vehicles()
        for v in vehicles:
            if v["name"].lower() == wanted:
                return self._hydrate(v)
        matches = [v for v in vehicles if wanted in v["name"].lower()]
        return self._hydrate(matches[0]) if len(matches) == 1 else None

    def _hydrate(self, vehicle: Dict[str, Any]) -> Dict[str, Any]:
        """Attach the spreadsheet id and subfolder ids to a vehicle record."""
        children = self._children(vehicle["folder_id"])
        for child in children:
            if child["mimeType"] == SHEET_MIME:
                vehicle["spreadsheet_id"] = child["id"]
                vehicle["spreadsheet_url"] = child.get("webViewLink", "")
            elif child["mimeType"] == FOLDER_MIME:
                if child["name"].lower() == DOCUMENTS_FOLDER.lower():
                    vehicle["documents_folder_id"] = child["id"]
                elif child["name"].lower() == RECEIPTS_FOLDER.lower():
                    vehicle["receipts_folder_id"] = child["id"]
        return vehicle

    # --- Vehicle lifecycle -----------------------------------------------
    def create_vehicle(self, name: str) -> Dict[str, Any]:
        """Create the folder tree and workbook for a new vehicle."""
        name = name.strip()
        if not name:
            raise ValueError("Vehicle name is required.")
        if self.find_vehicle(name):
            raise ValueError(f"A vehicle named {name!r} already exists.")

        folder_id = self._create_folder(name, self._fleet_folder_id)
        docs_id = self._create_folder(DOCUMENTS_FOLDER, folder_id)
        receipts_id = self._create_folder(RECEIPTS_FOLDER, folder_id)
        spreadsheet_id = self._create_workbook(name, folder_id)
        return {
            "name": name,
            "folder_id": folder_id,
            "documents_folder_id": docs_id,
            "receipts_folder_id": receipts_id,
            "spreadsheet_id": spreadsheet_id,
            "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
        }

    @retry_on_transient_error()
    def _create_workbook(self, name: str, parent_id: str) -> str:
        """
        Build the 4-tab workbook from the schema, inside the vehicle folder.

        The spreadsheet is created via the *Drive* API rather than
        `sheets.spreadsheets().create()`. Sheets' create call places the new
        file in the caller's own My Drive, which a service account in this
        Workspace is not permitted to do — it fails 403 "The caller does not
        have permission", which looks like an auth misconfiguration but
        isn't. Creating through Drive puts the file straight into the shared
        vehicle folder, where the agent demonstrably can write.
        """
        spreadsheet_id = self._drive.files().create(
            body={"name": name, "mimeType": SHEET_MIME, "parents": [parent_id]},
            fields="id", supportsAllDrives=True,
        ).execute()["id"]

        # A Drive-created spreadsheet has one default sheet. Rename it to
        # Info and add the table tabs alongside it.
        meta = self._sheets.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="sheets(properties(sheetId,title,index))").execute()
        default_sheet_id = meta["sheets"][0]["properties"]["sheetId"]

        requests: List[Dict[str, Any]] = [{
            "updateSheetProperties": {
                "properties": {"sheetId": default_sheet_id, "title": TAB_INFO, "index": 0},
                "fields": "title,index",
            }
        }]
        for i, tab in enumerate(TABLE_TABS):
            requests.append({"addSheet": {"properties": {"title": tab, "index": i + 1}}})
        self._sheets.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}).execute()

        data = [{"range": f"{TAB_INFO}!A1:A{len(INFO_FIELDS)}",
                 "values": [[f] for f in INFO_FIELDS]}]
        for tab, headers in TABLE_TABS.items():
            data.append({"range": f"'{tab}'!A1", "values": [headers]})
        self._sheets.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "USER_ENTERED", "data": data},
        ).execute()
        return spreadsheet_id

    @retry_on_transient_error()
    def delete_vehicle(self, name: str) -> Dict[str, Any]:
        """
        Trash a vehicle's folder (recoverable for 30 days).

        Deliberately a trash rather than a permanent delete: this is
        reachable from a chat message, and an irreversible delete one typo
        away from 'Tacoma' is not a risk worth taking.
        """
        vehicle = self.find_vehicle(name)
        if not vehicle:
            raise ValueError(f"No vehicle named {name!r}.")
        self._drive.files().update(
            fileId=vehicle["folder_id"], body={"trashed": True},
            supportsAllDrives=True,
        ).execute()
        return vehicle

    # --- Sheet values ----------------------------------------------------
    @retry_on_transient_error()
    def _get_values(self, spreadsheet_id: str, rng: str) -> List[List[Any]]:
        return self._sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=rng,
        ).execute().get("values", [])

    @retry_on_transient_error()
    def _append_row(self, spreadsheet_id: str, tab: str, row: List[Any]) -> None:
        self._sheets.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range=f"'{tab}'!A1",
            valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS",
            body={"values": [row]},
        ).execute()

    def read_info(self, vehicle: Dict[str, Any]) -> Dict[str, str]:
        rows = self._get_values(vehicle["spreadsheet_id"], f"{TAB_INFO}!A1:B100")
        return {r[0].strip(): (r[1].strip() if len(r) > 1 else "")
                for r in rows if r and r[0].strip()}

    @retry_on_transient_error()
    def update_info(self, vehicle: Dict[str, Any], updates: Dict[str, str]) -> List[str]:
        """
        Set Info fields by label, appending any label that doesn't exist yet.

        Returns the labels actually written.
        """
        rows = self._get_values(vehicle["spreadsheet_id"], f"{TAB_INFO}!A1:B100")
        index = {r[0].strip().lower(): i + 1 for i, r in enumerate(rows) if r and r[0].strip()}
        data, written, next_row = [], [], len(rows) + 1
        for label, value in updates.items():
            row_num = index.get(label.strip().lower())
            if row_num is None:
                row_num = next_row
                next_row += 1
                data.append({"range": f"{TAB_INFO}!A{row_num}", "values": [[label]]})
            data.append({"range": f"{TAB_INFO}!B{row_num}", "values": [[value]]})
            written.append(label)
        if data:
            self._sheets.spreadsheets().values().batchUpdate(
                spreadsheetId=vehicle["spreadsheet_id"],
                body={"valueInputOption": "USER_ENTERED", "data": data},
            ).execute()
        return written

    def _table(self, vehicle: Dict[str, Any], tab: str) -> List[Dict[str, str]]:
        """Read a header-row table into dicts keyed by the header labels."""
        rows = self._get_values(vehicle["spreadsheet_id"], f"'{tab}'!A1:Z1000")
        if not rows:
            return []
        headers = [h.strip() for h in rows[0]]
        out = []
        for row in rows[1:]:
            if not any(str(c).strip() for c in row):
                continue
            out.append({headers[i]: (row[i] if i < len(row) else "")
                        for i in range(len(headers))})
        return out

    # --- Repair history ---------------------------------------------------
    def list_repairs(self, vehicle: Dict[str, Any]) -> List[Dict[str, str]]:
        return self._table(vehicle, TAB_REPAIRS)

    def add_repair(self, vehicle: Dict[str, Any], when: str, description: str,
                   receipt_link: str = "") -> None:
        self._append_row(vehicle["spreadsheet_id"], TAB_REPAIRS,
                         [when, description, receipt_link])

    # --- Odometer ---------------------------------------------------------
    def list_odometer(self, vehicle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Readings sorted oldest first, with unparseable rows dropped."""
        readings = []
        for row in self._table(vehicle, TAB_ODOMETER):
            when, value = parse_date(row.get("Date")), parse_int(row.get("Reading"))
            if when and value is not None:
                readings.append({"date": when, "reading": value})
        return sorted(readings, key=lambda r: r["date"])

    def add_odometer(self, vehicle: Dict[str, Any], when: date, reading: int) -> None:
        self._append_row(vehicle["spreadsheet_id"], TAB_ODOMETER,
                         [format_date(when), reading])

    # --- Reminders --------------------------------------------------------
    def list_reminders(self, vehicle: Dict[str, Any]) -> List[Dict[str, str]]:
        return self._table(vehicle, TAB_REMINDERS)

    def add_reminder(self, vehicle: Dict[str, Any], task: str,
                     date_required: str = "", mileage_required: str = "") -> None:
        self._append_row(vehicle["spreadsheet_id"], TAB_REMINDERS,
                         [date_required, mileage_required, task])

    # --- Documents --------------------------------------------------------
    def list_documents(self, vehicle: Dict[str, Any]) -> List[Dict[str, str]]:
        folder_id = vehicle.get("documents_folder_id")
        if not folder_id:
            return []
        return [{"name": f["name"], "id": f["id"], "url": f.get("webViewLink", "")}
                for f in self._children(folder_id)]

    def missing_documents(self, vehicle: Dict[str, Any]) -> List[str]:
        present = " ".join(d["name"].lower() for d in self.list_documents(vehicle))
        return [want for want in REQUIRED_DOCUMENTS if want.lower() not in present]

    def store_file(self, vehicle: Dict[str, Any], data: bytes, filename: str,
                   mime_type: str, into: str) -> Dict[str, str]:
        """Upload bytes into this vehicle's Documents or Service Receipts folder."""
        key = "documents_folder_id" if into == DOCUMENTS_FOLDER else "receipts_folder_id"
        folder_id = vehicle.get(key)
        if not folder_id:
            # Older folders may predate a subfolder; create it on demand
            # rather than failing the upload the user just made.
            folder_id = self._create_folder(into, vehicle["folder_id"])
            vehicle[key] = folder_id
        return self._upload_bytes(data, filename, mime_type, folder_id)


# --- Mileage projection ----------------------------------------------------
def daily_mileage_rate(readings: List[Dict[str, Any]]) -> Optional[float]:
    """
    Miles per day, from the oldest and newest readings.

    Endpoints rather than a regression on purpose: with the handful of
    readings these sheets realistically hold, the span between first and
    last is the honest estimate, and it degrades gracefully as readings
    accumulate. Returns None when there isn't enough data to say anything.
    """
    if len(readings) < 2:
        return None
    first, last = readings[0], readings[-1]
    days = (last["date"] - first["date"]).days
    miles = last["reading"] - first["reading"]
    if days <= 0 or miles < 0:
        return None
    return miles / days


def project_mileage(readings: List[Dict[str, Any]],
                    as_of: Optional[date] = None) -> Optional[int]:
    """Estimate today's odometer by extending the trend past the last reading."""
    if not readings:
        return None
    as_of = as_of or today()
    rate = daily_mileage_rate(readings)
    last = readings[-1]
    if rate is None:
        # One reading only: the best estimate is that reading itself.
        return last["reading"]
    elapsed = (as_of - last["date"]).days
    return int(round(last["reading"] + rate * max(elapsed, 0)))


def date_mileage_due(readings: List[Dict[str, Any]], target_mileage: int,
                     as_of: Optional[date] = None) -> Optional[date]:
    """When the odometer is projected to reach `target_mileage`."""
    rate = daily_mileage_rate(readings)
    if not rate or not readings:
        return None
    as_of = as_of or today()
    current = project_mileage(readings, as_of)
    if current is None:
        return None
    remaining = target_mileage - current
    if remaining <= 0:
        return as_of
    return as_of + timedelta(days=int(round(remaining / rate)))


_connector: Optional[FleetConnector] = None


def get_fleet_connector() -> FleetConnector:
    """Cached singleton, built on first use."""
    global _connector
    if _connector is None:
        _connector = FleetConnector()
    return _connector
