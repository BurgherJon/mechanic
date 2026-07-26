"""
Helpers for reading secrets from Google Cloud Secret Manager.

Use these whenever your agent needs an API key, OAuth token, or any other
secret value. Never hard-code secrets or store them in .env — they belong
in Secret Manager, with an IAM binding granting the Reasoning Engine's
service account secretAccessor on the secret.

See AGENTS.md for the full rule on secret handling.
"""
import logging
import ssl
import time
from functools import wraps

from google.cloud import secretmanager
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


# Transient errors that should be retried with backoff.
TRANSIENT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionError,
    TimeoutError,
    ssl.SSLError,
)


def retry_on_transient_error(max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a function on transient network errors with exponential backoff.

    Also retries on HTTP 5xx responses from Google APIs. Use this as a
    decorator on any function that makes a network call to a Google API.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except TRANSIENT_ERRORS as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Transient error in {func.__name__}: {e}. "
                            f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise
                except HttpError as e:
                    if e.resp.status >= 500:
                        last_exception = e
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(
                                f"Server error in {func.__name__}: {e}. "
                                f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


def get_secret_from_secret_manager(
    project_id: str,
    secret_id: str,
    version_id: str = "latest",
) -> str:
    """
    Fetch a secret value from Google Cloud Secret Manager.

    Args:
        project_id: The GCP project hosting the secret. For per-agent secrets
            (Slack tokens, MCP keys, etc.) this is the agent's own project
            (BOT_ACCOUNT_ID and AGENT_SECRET_PROJECT in .env).
        secret_id: The Secret Manager secret ID (not the resource name).
        version_id: Secret version. Defaults to "latest".

    Returns:
        The decoded secret payload as a string.

    Raises:
        google.api_core.exceptions.PermissionDenied: if the calling identity
            doesn't have secretAccessor. Most commonly fixed by uncommenting
            the IAM binding for this secret in terraform/main.tf.
        google.api_core.exceptions.NotFound: if the secret container doesn't
            exist or has no enabled versions.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
