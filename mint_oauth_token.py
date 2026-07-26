"""
Mint the OAuth user credentials Mike uses for Drive and Sheets.

Why user credentials rather than the agent's service account: a service
account has no Drive storage quota of its own, and in Drive the *creator*
owns the file. So an SA can edit workbooks it's been given access to, but
cannot create a new vehicle workbook, archive a receipt, or file an
insurance PDF — those all fail with `storageQuotaExceeded`. Running as a
real user sidesteps that; files are owned by that user and count against
their quota.

(The longer-term fix is a Shared Drive, where files are owned by the drive
rather than by whoever created them. See the Shared Drive story in Linear.
When that lands, this can go away and fleet_utilities can return to ADC.)

WHICH ACCOUNT TO MINT THIS FOR
------------------------------
Use a dedicated, secondary Google account — NOT the account that owns the
Drive these folders live in.

This token carries the full `drive` scope, which cannot be narrowed to a
single folder: whoever holds it can read and write everything that account
can reach. Minting it for your primary account puts your entire Drive
behind a refresh token sitting in Secret Manager, read by a long-running
agent. Minting it for a second account that has been shared *only* the
fleet folder means a leaked token exposes exactly that folder and nothing
else.

Setup for the recommended arrangement:
  1. Create (or pick) a secondary Google account for the agent to act as.
  2. From the account that owns the fleet folder, share that one folder
     with the secondary account as Editor. Share nothing else.
  3. Run this script signed in as the secondary account.

The trade-off: files Mike creates are owned by the secondary account, so
they count against its quota and disappear if it is deleted. That is the
same containment working as intended — keep the account, or move to a
Shared Drive.

Per repo convention, secrets never live in code or terraform. This script
prints the token JSON to stdout; you pipe it into Secret Manager.

Prerequisites:
  A Desktop-app OAuth client in the agent's GCP project:
    APIs & Services → Credentials → Create credentials → OAuth client ID
    → Desktop app. Download the client_secret JSON.
  The project's OAuth consent screen must list the Drive and Sheets scopes
  below, and the account you mint for must be a test user (or the app
  published).

Usage:
    python mint_oauth_token.py --client-secret ~/client_secret.json \\
        | gcloud secrets versions add mike-the-mechanic-google-oauth \\
            --data-file=- --project=mike-the-mechanic-prod

Re-run to rotate. The old refresh token keeps working until revoked, so
add the new version first and verify before revoking anything.
"""
import argparse
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

# Drive is needed to create folders/files and upload receipts; spreadsheets
# to build and edit the workbooks. Both are minted in a single flow —
# authorizing them separately would invalidate the earlier grant.
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--client-secret",
        required=True,
        help="Path to the Desktop-app OAuth client_secret JSON.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port for the consent redirect (0 = pick a free one).",
    )
    args = parser.parse_args()

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secret, SCOPES)
    # prompt='consent' forces a refresh token even if this account has
    # already authorized the client — without it a re-mint can come back
    # with an access token only, and the engine dies when it expires.
    # The default prompt message is printed to stdout by google-auth-oauthlib,
    # which would corrupt the token JSON when this script is piped straight
    # into `gcloud secrets versions add`. Suppress it; the browser opens on
    # its own, and any real error still goes to stderr.
    creds = flow.run_local_server(
        port=args.port,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message="",
    )

    if not creds.refresh_token:
        print("ERROR: no refresh token returned. Re-run; if it persists, "
              "revoke the client's access at myaccount.google.com/permissions "
              "and try again.", file=sys.stderr)
        return 1

    # Token JSON → stdout, for piping into `gcloud secrets versions add`.
    print(creds.to_json())
    print("\nMinted for scopes:\n  " + "\n  ".join(SCOPES), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
