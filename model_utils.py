"""Model construction: three tiers of model behind one routing, resilience
wrapper.

Tiers (2026-10-04; Gemini Pro and the first-party Anthropic path are retired):

    LOCAL_AGENT_MODEL          local/agent-large                  free; tried first when it is up and the request fits
    HIGH_QUALITY_AGENT_MODEL   openrouter/xiaomi/mimo-v2.6-pro    root's hosted model: whatever local cannot take
    SPECIALIST_AGENT_MODEL     openrouter/xiaomi/mimo-v2.6-pro    sub-agents' hosted model
    FALLBACK_AGENT_MODEL       gemini-3.8-flash                   last resort when the hosted model fails
    SEARCH_AGENT_MODEL         gemini-3.8-flash                   sub-agents using google_search/url_context
    VISION_MODEL               gemini-3.8-flash                   direct genai extraction calls

Model ids name their provider by prefix, and each provider has one builder
in `_PROVIDERS`:

* `local/<alias>`: Jonathan's self-hosted models (Qwen3.8 27B behind the
  alias agent-large, 64k context) on the comites.ai LLM server, an
  OpenAI-compatible LiteLLM proxy at COMITES_LLM_BASE_URL behind a
  Cloudflare tunnel, reached through ADK's LiteLLM class. The server is
  shared and sometimes busy or off, so it is pinged (no key, no model call)
  before it is used. Its key is `comites-llm-key` in this agent's project.
  See the "comites.ai LLM Connection" page for the server.
* `openrouter/<maker>/<model>`: an open-weight model through OpenRouter on
  ADK's LiteLLM class, with this agent's OpenRouter key from Secret Manager
  (`{BOT_ACCOUNT_ID}-openrouter-key`, or OPENROUTER_SECRET_ID). Ported from
  the Comites template (Nina the Nanny, AGE-17): provider routing pinned on
  every request, reasoning excluded from the response, prompt caching is
  the serving provider's own and automatic.
* `gemini-*`: ADK's native Gemini class on Vertex AI. Keeps context caching
  and the built-in Google Search tool, which adapter layers lose, so search
  and vision stay here.

Rules this module encodes, each learned in production:

1. Never hand a bare model string to Agent(model=...). Every model goes
   through a factory here so the wrapper below is always in the path, and
   an id with no provider fails at import rather than mid-conversation (a
   stale `claude-*` or `gemini-3.1-pro-*` in .env is the likely one).
2. Never subclass ADK model internals (Sam's June-2026 TypeError came from
   a subclass written against an older ADK). Everything here composes at
   the `BaseLlm` boundary, which is a stable public contract.
3. Agent Engine runs every request on a fresh event loop in its own
   thread. An async HTTP client cached at import binds to the first loop
   and dies on the second request ("attached to a different loop"). Inner
   models are therefore built per event loop and cached in a small map.
4. Non-streaming completions only: streaming teardown mid-turn kills turns
   that use MCP toolsets, and the Forum consumes the full stream before
   delivering anyway, so nothing is lost.
5. Route each call: local when the endpoint answers its health check and
   the request fits the local window with headroom, else the hosted model.
   One model per turn: a turn that starts on the hosted model, or that
   escalates to it, stays there for the rest of its tool calls, so the
   reply keeps one voice (seen 2026-09-27: a Flash backup finished a Pro
   turn in a different voice). A local call that fails, times out or runs
   out of output tokens escalates; it is never retried at length, because
   the Forum gives a whole turn 300 seconds.
6. Retry transient serving errors (408 / 429 / 5xx) on the hosted model
   before failing a turn, honouring a Retry-After. An empty final response
   gets one retry of the hosted model after a short pause, then the
   fallback, which runs UNCACHED: ADK's context cache rewrites the request
   in place for the model that owns it, and Vertex refuses that cache for
   another model. So the request is snapshotted before the first call and
   restored before every further call.
7. Heal orphaned tool calls before every model call, so a turn the platform
   cut off cannot poison the rest of the session.
8. No thought part ever reaches the Forum or the history. The Forum joins
   every text part of the reply, thoughts included (five paragraphs of
   MiMo's deliberation reached a Discord ahead of Nina's reply,
   2026-09-29), so thought parts are stripped from every response, and the
   history is cleaned of any left by an earlier model (Claude's signed
   thinking blocks fail Gemini with 400 "Invalid thought signature").
   OpenRouter is also asked not to return reasoning at all.
9. Stamp unsigned function calls with Gemini's dummy thought signature
   right before a Gemini call: Vertex rejects an unsigned call in the turn
   in progress with 400 "Function call is missing a thought_signature",
   which is every call a local or OpenRouter model made.
10. Log one `model usage` line per completed call with the prompt, cached,
   thought and output token counts, plus this module's own estimate of the
   prompt (`est=`) so the local routing threshold can be tuned against the
   counts the serving model reports.
11. Log the whole error, not its first 200 characters: ADK's text opens
   with a preamble, and the status, quota and Retry-After come after it.
"""
import asyncio
import collections
import functools
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr
from typing_extensions import override

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "local/agent-large"
DEFAULT_HIGH_QUALITY_MODEL = "openrouter/xiaomi/mimo-v2.6-pro"
DEFAULT_SPECIALIST_MODEL = "openrouter/xiaomi/mimo-v2.6-pro"
DEFAULT_FALLBACK_MODEL = "gemini-3.8-flash"
DEFAULT_VISION_MODEL = "gemini-3.8-flash"
# Sub-agents that use ADK's google_search / url_context are gated to Gemini.
DEFAULT_SEARCH_MODEL = "gemini-3.8-flash"
DEFAULT_SEARCH_BACKUP_MODEL = "gemini-3.5-flash"

# Output cap for hosted calls. These calls are non-streaming, so the cap must
# leave the response inside the HTTP request timeout; 16384 covers the
# longest outputs in the estate (Sam's ~4000-word memory rewrite).
MAX_OUTPUT_TOKENS = 16384

# An agent's `effort` maps onto Gemini's thinking level (the fallback) and
# onto OpenRouter's reasoning effort. 3.8 Flash rejects MINIMAL, so the map
# stops at LOW; unknown values fall back to the model's own default.
_THINKING_LEVELS = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}

# Substrings that mark a transient serving failure worth retrying. Matched
# against str(exception) because google-genai, httpx and LiteLLM each raise
# their own types. 500/INTERNAL were absent before 2026-09-06 and let a
# transient Vertex 500 kill a deploy verification.
_RETRYABLE_MARKERS = (
    "429", "RESOURCE_EXHAUSTED", "rate_limit",
    "500", "INTERNAL",
    "502", "503", "UNAVAILABLE", "504",
    "RateLimitError", "ServiceUnavailableError", "InternalServerError", "APIConnectionError",
)
# Status codes the SDKs attach to their exceptions (google.genai APIError.code,
# LiteLLM's status_code); checked ahead of the text markers.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
_RETRY_ATTEMPTS = 3
_ERROR_TEXT_LIMIT = 2000
_RETRY_AFTER_CAP_SECONDS = 30.0
# Pause before the one retry of a model that returned an empty final response.
_DEGENERATE_RETRY_PAUSE_SECONDS = 2.0
_INNER_CACHE_CAP = 8


# ---------------------------------------------------------------------------
# Model ids and config
# ---------------------------------------------------------------------------

def local_model_id() -> Optional[str]:
    """The local model, or None when LOCAL_AGENT_MODEL is "off" (or empty).

    "off" rather than an empty value is the documented switch: a deploy path
    that drops empty variables would otherwise bring the default back on.
    """
    value = os.environ.get("LOCAL_AGENT_MODEL", DEFAULT_LOCAL_MODEL).strip()
    return None if value.lower() in ("", "off", "none") else value


def high_quality_model_id() -> str:
    return os.environ.get("HIGH_QUALITY_AGENT_MODEL", DEFAULT_HIGH_QUALITY_MODEL)


def specialist_model_id() -> str:
    return os.environ.get("SPECIALIST_AGENT_MODEL", DEFAULT_SPECIALIST_MODEL)


def fallback_model_id() -> str:
    return os.environ.get("FALLBACK_AGENT_MODEL", DEFAULT_FALLBACK_MODEL)


def vision_model_id() -> str:
    return os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL)


def search_model_id() -> str:
    return os.environ.get("SEARCH_AGENT_MODEL", DEFAULT_SEARCH_MODEL)


def _agent_config(effort: Optional[str]) -> types.GenerateContentConfig:
    """Per-agent generation config: the output cap plus the effort as a
    Gemini thinking level (read only by the Gemini fallback; the LiteLLM
    models take effort from the wrapper instead).

    Never set include_thoughts: the Forum joins every text part of the
    stream into the reply, so thought text would reach the user.
    """
    level = _THINKING_LEVELS.get((effort or "").lower())
    return types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=level) if level else None,
    )


def high_quality_config(effort: str = "high") -> types.GenerateContentConfig:
    return _agent_config(effort)


def specialist_config(effort: str = "medium") -> types.GenerateContentConfig:
    return _agent_config(effort)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

PROVIDER_GEMINI = "gemini"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_LOCAL = "local"
OPENROUTER_PREFIX = "openrouter/"
LOCAL_PREFIX = "local/"


class UnsupportedModelError(ValueError):
    """The model id names a provider this module does not wire up."""


def provider_for(model_id: str) -> str:
    """Name the provider that serves `model_id`; raise a clear error otherwise."""
    if model_id.startswith("gemini-") and not model_id.startswith("gemini-3.1-pro"):
        return PROVIDER_GEMINI
    if model_id.startswith(OPENROUTER_PREFIX) and len(model_id) > len(OPENROUTER_PREFIX):
        return PROVIDER_OPENROUTER
    if model_id.startswith(LOCAL_PREFIX) and len(model_id) > len(LOCAL_PREFIX):
        return PROVIDER_LOCAL
    raise UnsupportedModelError(
        f"model {model_id!r}: this module serves local/<alias> ids on the comites.ai "
        "LLM server, openrouter/<maker>/<model> ids through OpenRouter, and gemini-* "
        "ids on Vertex AI (Gemini Pro and Claude are retired). Fix LOCAL_AGENT_MODEL / "
        "HIGH_QUALITY_AGENT_MODEL / SPECIALIST_AGENT_MODEL / FALLBACK_AGENT_MODEL in "
        ".env (SEARCH_AGENT_MODEL and VISION_MODEL must be gemini-*)."
    )


def _agent_project() -> str:
    project = os.environ.get("AGENT_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
    if not project:
        raise RuntimeError("AGENT_PROJECT_ID is not set; it names the project holding this agent's model keys")
    return project


@functools.lru_cache(maxsize=8)
def _secret(project_id: str, secret_id: str) -> str:
    """A secret's latest value, blocking, once per container per secret.

    A failed read is not cached, so a key added after the deploy is picked
    up on a later call. The value is never logged.
    """
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    value = client.access_secret_version(request={"name": name}).payload.data.decode("utf-8").strip()
    if not value:
        raise RuntimeError(f"secret {secret_id} in {project_id} has an empty value")
    return value


def _build_gemini(model_id: str, effort: Optional[str]) -> BaseLlm:
    from google.adk.models.google_llm import Gemini

    return Gemini(model=model_id)


# --- OpenRouter (ported from the Comites template, AGE-17) -------------------

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# OpenRouter's provider tags for MiMo: GMICloud serves it in full bf16,
# DeepInfra in fp8; Xiaomi's own endpoint is in the PRC and is left out. An
# agent on another model overrides these in .env.
DEFAULT_OPENROUTER_PROVIDER_ORDER = "gmicloud,deepinfra"
DEFAULT_OPENROUTER_PROVIDER_IGNORE = "xiaomi"


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def openrouter_routing() -> Dict[str, Any]:
    """The `provider` object sent with every OpenRouter request.

    `order` is tried first, in order, and keeping it fixed is what lets the
    provider's automatic prompt cache hit turn after turn; `allow_fallbacks`
    reaches any other provider when those are down; `ignore` names providers
    never to use; `data_collection: deny` rules out any provider that may
    keep prompts for training; `require_parameters` rules out any provider
    that cannot take every parameter sent, tools included.
    """
    routing: Dict[str, Any] = {
        "order": _csv(os.environ.get("OPENROUTER_PROVIDER_ORDER", DEFAULT_OPENROUTER_PROVIDER_ORDER)),
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
    }
    ignore = _csv(os.environ.get("OPENROUTER_PROVIDER_IGNORE", DEFAULT_OPENROUTER_PROVIDER_IGNORE))
    if ignore:
        routing["ignore"] = ignore
    return routing


def openrouter_secret_id() -> str:
    secret_id = os.environ.get("OPENROUTER_SECRET_ID") or (
        f"{os.environ['BOT_ACCOUNT_ID']}-openrouter-key" if os.environ.get("BOT_ACCOUNT_ID") else ""
    )
    if not secret_id:
        raise RuntimeError("an openrouter/ model needs BOT_ACCOUNT_ID or OPENROUTER_SECRET_ID in .env")
    return secret_id


def _build_openrouter(model_id: str, effort: Optional[str]) -> BaseLlm:
    """ADK's LiteLLM class on OpenRouter, with the key, the routing and the effort.

    `reasoning.exclude` keeps the model's reasoning out of the response: it
    still thinks, and its thoughts are not returned (rule 8).
    """
    from google.adk.models.lite_llm import LiteLlm

    reasoning: Dict[str, Any] = {"exclude": True}
    if effort in _THINKING_LEVELS:
        reasoning["effort"] = effort
    return LiteLlm(
        model=model_id,
        api_key=_secret(_agent_project(), openrouter_secret_id()),
        api_base=OPENROUTER_API_BASE,
        extra_body={"provider": openrouter_routing(), "reasoning": reasoning},
        drop_params=True,
    )


# --- The comites.ai LLM server (local models) --------------------------------

DEFAULT_COMITES_LLM_BASE_URL = "https://llm.jonathancavell.com/v1"
DEFAULT_COMITES_LLM_SECRET_ID = "comites-llm-key"
# agent-large's window. The server's guide asks for ~20% headroom over
# everything that shares it: prompt, tools, history, reasoning and output.
DEFAULT_LOCAL_CONTEXT_TOKENS = 65536
_LOCAL_HEADROOM = 0.2
# Output reserved inside the window for a local call (reasoning included).
DEFAULT_LOCAL_MAX_OUTPUT_TOKENS = 8192
# The server reads ~800 prompt tokens/s and writes ~40-65 tokens/s, and the
# Forum gives a whole turn 300 s, so a local call that runs past this
# escalates while there is still time for the hosted model to answer.
DEFAULT_LOCAL_TIMEOUT_SECONDS = 120.0
# Mixed prompts (prose, tool schemas, JSON) run ~3 characters a token on the
# server's tokenizers; prose runs higher, so this over-counts, the safe way.
_CHARS_PER_TOKEN = 3.0
_IMAGE_TOKENS = 1500
_PER_MESSAGE_TOKENS = 4
# The health check: answered by the server's proxy without a key and without
# loading a model. Remembered for a minute either way.
_PING_TIMEOUT_SECONDS = 3.0
_PING_TTL_SECONDS = 60.0


def comites_llm_base_url() -> str:
    return os.environ.get("COMITES_LLM_BASE_URL", DEFAULT_COMITES_LLM_BASE_URL).rstrip("/")


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def local_max_output_tokens() -> int:
    return int(_env_number("LOCAL_MAX_OUTPUT_TOKENS", DEFAULT_LOCAL_MAX_OUTPUT_TOKENS))


def local_prompt_budget() -> int:
    """Estimated prompt tokens a local call may carry: the window less headroom and the output reserve."""
    window = _env_number("LOCAL_CONTEXT_TOKENS", DEFAULT_LOCAL_CONTEXT_TOKENS)
    return int(window * (1 - _LOCAL_HEADROOM)) - local_max_output_tokens()


def _local_litellm_id(model_id: str) -> str:
    # The server speaks OpenAI's API; LiteLLM's openai/ prefix sends the
    # alias as the model name to COMITES_LLM_BASE_URL.
    return "openai/" + model_id[len(LOCAL_PREFIX):]


def _build_local(model_id: str, effort: Optional[str]) -> BaseLlm:
    """ADK's LiteLLM class on the comites.ai LLM server.

    The server's models think by default; effort "none" turns that off.
    Their reasoning arrives as `reasoning_content`, which ADK turns into
    thought parts, and the wrapper strips those from the response (rule 8).
    """
    from google.adk.models.lite_llm import LiteLlm

    kwargs: Dict[str, Any] = {}
    if effort == "none":
        kwargs["reasoning_effort"] = "none"
    return LiteLlm(
        model=_local_litellm_id(model_id),
        api_key=_secret(_agent_project(), os.environ.get("COMITES_LLM_SECRET_ID", DEFAULT_COMITES_LLM_SECRET_ID)),
        api_base=comites_llm_base_url(),
        timeout=_env_number("LOCAL_TIMEOUT_SECONDS", DEFAULT_LOCAL_TIMEOUT_SECONDS),
        drop_params=True,
        **kwargs,
    )


_local_health: Dict[str, float] = {}


def mark_local_down(reason: str) -> None:
    """Skip the local server until the next health check is due."""
    if _local_health.get("ok", 0.0):
        logger.warning("Local LLM server marked unavailable for %gs: %s", _PING_TTL_SECONDS, reason)
    _local_health.update(ok=0.0, at=time.monotonic())


async def local_available() -> bool:
    """True when the comites.ai LLM server answers its health check.

    `GET /health/readiness` on the server's LiteLLM proxy: no key, no model
    call, ~0.1 s. It proves the machine, the tunnel and the proxy are up,
    not that a model is loaded; a model failure behind a healthy proxy
    shows up as an error on the call itself, which escalates. The answer is
    remembered for `_PING_TTL_SECONDS` so a turn's calls ping once.
    """
    now = time.monotonic()
    if "at" in _local_health and now - _local_health["at"] < _PING_TTL_SECONDS:
        return bool(_local_health["ok"])
    import httpx

    url = comites_llm_base_url().removesuffix("/v1") + "/health/readiness"
    ok = False
    try:
        async with httpx.AsyncClient(timeout=_PING_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
        ok = response.status_code == 200 and response.json().get("status") == "healthy"
        if not ok:
            logger.info("Local LLM health check: HTTP %s %s", response.status_code, response.text[:200])
    except Exception as exc:  # noqa: BLE001 — any failure means "not now"
        logger.info("Local LLM health check failed: %s", describe_error(exc))
    if ok != bool(_local_health.get("ok", 0.0)) and "at" in _local_health:
        logger.warning("Local LLM server is now %s", "available" if ok else "unavailable")
    _local_health.update(ok=1.0 if ok else 0.0, at=now)
    return ok


def _part_chars(part: types.Part) -> Tuple[int, int]:
    """(characters, images) a part adds to a prompt."""
    chars, images = 0, 0
    if part.text:
        chars += len(part.text)
    if part.function_call is not None:
        chars += len(part.function_call.name or "") + len(json.dumps(part.function_call.args or {}, default=str))
    if part.function_response is not None:
        chars += len(part.function_response.name or "") + len(json.dumps(part.function_response.response or {}, default=str))
    if part.inline_data is not None or part.file_data is not None:
        images += 1
    return chars, images


def estimate_request_tokens(llm_request: LlmRequest) -> int:
    """A conservative estimate of the prompt: system instruction, tools and
    the whole history, tool calls and results included (ADK's own estimate
    counts text only, and tool output is what makes these agents large).
    """
    chars, images, messages = 0, 0, 0
    config = llm_request.config
    if config is not None and config.system_instruction:
        instruction = config.system_instruction
        if isinstance(instruction, str):
            chars += len(instruction)
        else:
            for part in getattr(instruction, "parts", None) or []:
                chars += _part_chars(part)[0]
    if config is not None and config.tools:
        for tool in config.tools:
            if isinstance(tool, types.Tool):
                chars += len(json.dumps(tool.model_dump(exclude_none=True), default=str))
    for content in llm_request.contents or []:
        messages += 1
        for part in content.parts or []:
            c, i = _part_chars(part)
            chars += c
            images += i
    return int(chars / _CHARS_PER_TOKEN) + images * _IMAGE_TOKENS + messages * _PER_MESSAGE_TOKENS


_PROVIDERS: Dict[str, Callable[[str, Optional[str]], BaseLlm]] = {
    PROVIDER_GEMINI: _build_gemini,
    PROVIDER_OPENROUTER: _build_openrouter,
    PROVIDER_LOCAL: _build_local,
}


def _build_inner(model_id: str, effort: Optional[str]) -> BaseLlm:
    return _PROVIDERS[provider_for(model_id)](model_id, effort)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _status_code(exc: BaseException) -> Optional[int]:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_transient(exc: BaseException) -> bool:
    if _status_code(exc) in _RETRYABLE_STATUS_CODES:
        return True
    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def _is_timeout(exc: BaseException) -> bool:
    return "Timeout" in type(exc).__name__ or "timed out" in str(exc).lower()


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The `Retry-After` header on the exception's HTTP response, in seconds, if any."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
        seconds = float(value) if value else None
    except Exception:  # noqa: BLE001 — an odd header is no reason to fail a retry
        return None
    return seconds if seconds is not None and seconds >= 0 else None


def describe_error(exc: BaseException, limit: int = _ERROR_TEXT_LIMIT) -> str:
    """The error as one log line: type, the whole message, the structured fields, Retry-After."""
    bits = [f"{type(exc).__name__}: {' '.join(str(exc).split())}"]
    for attr in ("code", "status_code", "status", "llm_provider"):
        value = getattr(exc, attr, None)
        if value not in (None, "") and not callable(value):
            bits.append(f"{attr}={value}")
    details = getattr(exc, "details", None)
    if details and not callable(details):
        try:
            rendered = json.dumps(details, default=str)
        except (TypeError, ValueError):
            rendered = str(details)
        bits.append(f"details={rendered[:800]}")
    retry_after = retry_after_seconds(exc)
    if retry_after is not None:
        bits.append(f"Retry-After={retry_after:g}s")
    text = " | ".join(bits)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# History repair, applied before every model call
# ---------------------------------------------------------------------------

_INTERRUPTED_RESULT: Dict[str, Any] = {
    "error": (
        "interrupted: no result was recorded for this call (platform timeout "
        "or cutoff). Treat its side effects as unconfirmed and re-check before "
        "assuming they happened."
    )
}


def heal_orphaned_tool_calls(contents: List[types.Content]) -> int:
    """Insert a synthetic function_response for every function_call that
    never got one, so a cut-off turn cannot poison the whole session.

    Found live on 2026-09-09: the platform cut a turn off after the
    function_call event was persisted but before the tool replied. Every
    later turn replayed that history and the provider rejected it (a tool
    call without a tool result), on primary and backup alike, until the
    session was reset. ADK already moves each function_response to sit right
    after its call, so by the time contents reach the model any call still
    unanswered is a true orphan.

    Mutates `contents` in place. If the content right after the orphaned
    call already carries function responses (a sibling call that DID get
    answered), the synthetic result is added to that content, because a
    strict provider requires every call of a message to be answered in the
    very next message. Calls without an id (some Gemini histories) cannot be
    matched and are left alone. Returns the number of calls healed.
    """
    answered = set()
    for content in contents:
        for part in content.parts or []:
            if part.function_response is not None and part.function_response.id:
                answered.add(part.function_response.id)

    healed = 0
    i = 0
    while i < len(contents):
        orphans = [
            part.function_call
            for part in (contents[i].parts or [])
            if part.function_call is not None
            and part.function_call.id
            and part.function_call.id not in answered
        ]
        if orphans:
            parts = [
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response=dict(_INTERRUPTED_RESULT)
                    )
                )
                for call in orphans
            ]
            nxt = contents[i + 1] if i + 1 < len(contents) else None
            if nxt is not None and any(
                p.function_response is not None for p in (nxt.parts or [])
            ):
                nxt.parts = list(nxt.parts or []) + parts
            else:
                contents.insert(i + 1, types.Content(role="user", parts=parts))
                i += 1
            for call in orphans:
                answered.add(call.id)
                logger.warning(
                    "Healed orphaned tool call %s (id=%s): no result was recorded",
                    call.name, call.id,
                )
            healed += len(orphans)
        i += 1
    return healed


def drop_thought_parts(contents: List[types.Content]) -> int:
    """Remove every thought part from `contents`, and any content left empty.

    Sessions that ran on Claude hold its thinking blocks as parts with
    `thought=True` and Claude's signature in `thought_signature`; ADK replays
    them, and Gemini 3 rejects the signature with 400 "Invalid thought
    signature", on every later turn of that session. Gemini itself returns
    thought parts only when `include_thoughts` is on, which nothing here
    sets; its own signatures ride on function-call and text parts, which
    are kept. A thought part carrying a function call or response is kept
    too. Mutates in place; returns the number of parts dropped.
    """
    dropped = 0
    kept_contents: List[types.Content] = []
    for content in contents:
        parts = content.parts or []
        kept = [
            p for p in parts
            if not p.thought or p.function_call is not None or p.function_response is not None
        ]
        dropped += len(parts) - len(kept)
        if len(kept) != len(parts):
            content.parts = kept
        if kept or not parts:
            kept_contents.append(content)
    contents[:] = kept_contents
    return dropped


def strip_response_thoughts(responses: List[LlmResponse]) -> int:
    """Remove thought parts from model responses before they reach the Forum (rule 8)."""
    dropped = 0
    for response in responses:
        content = response.content
        if content is None or not content.parts:
            continue
        kept = [
            p for p in content.parts
            if not p.thought or p.function_call is not None or p.function_response is not None
        ]
        dropped += len(content.parts) - len(kept)
        if len(kept) != len(content.parts):
            content.parts = kept
    return dropped


# Gemini 3 validates a thought signature on every function call in the
# current turn. A call made by another model carries none, so the turn is
# rejected with 400 "Function call is missing a thought_signature". Google's
# documented escape for history that came from elsewhere is this dummy
# value, which tells the validator to skip the part.
SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


def stamp_thought_signatures(contents: List[types.Content]) -> int:
    """Give every unsigned function call in `contents` the dummy signature.

    Only model-authored function-call parts without a signature are touched;
    a real signature is never overwritten and nothing else changes. Mutates
    in place; returns the number of parts stamped.
    """
    stamped = 0
    for content in contents:
        if content.role != "model":
            continue
        for part in content.parts or []:
            if part.function_call is not None and not part.thought_signature:
                part.thought_signature = SKIP_THOUGHT_SIGNATURE
                stamped += 1
    return stamped


class _RequestSnapshot:
    """The parts of an `LlmRequest` a model call rewrites in place.

    ADK's context cache manager applies an active cache by stripping the
    system instruction, tools and tool config from `config`, setting
    `config.cached_content` and cutting the cached prefix off `contents`;
    the Gemini model also appends a user content when the history ends on
    the model's side. A second call on the same object, whether a retry, an
    escalation or the fallback, must start from the request as the flow
    built it, so it is captured here before the first call and put back
    before every later one.
    """

    def __init__(self, llm_request: LlmRequest):
        self.contents = list(llm_request.contents or [])
        self.config = llm_request.config.model_copy() if llm_request.config is not None else None
        self.cache_config = llm_request.cache_config
        self.cache_metadata = llm_request.cache_metadata
        self.cacheable_contents_token_count = llm_request.cacheable_contents_token_count

    def restore(self, llm_request: LlmRequest, *, uncached: bool = False) -> None:
        """Put the request back; `uncached` also drops every trace of the cache."""
        llm_request.contents = list(self.contents)
        llm_request.config = self.config.model_copy() if self.config is not None else None
        if uncached:
            llm_request.cache_config = None
            llm_request.cache_metadata = None
            llm_request.cacheable_contents_token_count = None
            if llm_request.config is not None:
                llm_request.config.cached_content = None
        else:
            llm_request.cache_config = self.cache_config
            llm_request.cache_metadata = self.cache_metadata
            llm_request.cacheable_contents_token_count = self.cacheable_contents_token_count


def usage_line(model_id: str, responses: List[Any], estimate: Optional[int] = None) -> Optional[str]:
    """One line of token accounting for a completed call, or None.

    Cached tokens are the prompt prefix the host served from its cache
    (Vertex's, the OpenRouter provider's or the local server's); `est` is
    this module's estimate of the same prompt, for tuning the local budget.
    """
    usage = None
    for response in reversed(list(responses)):
        if getattr(response, "usage_metadata", None) is not None:
            usage = response.usage_metadata
            break
    if usage is None:
        return None
    line = (
        f"model usage {model_id}: prompt={usage.prompt_token_count or 0} "
        f"cached={usage.cached_content_token_count or 0} "
        f"thoughts={usage.thoughts_token_count or 0} "
        f"output={usage.candidates_token_count or 0}"
    )
    return line + (f" est={estimate}" if estimate is not None else "")


def _turn_key(contents: List[types.Content]) -> Optional[str]:
    """Identify the turn a request belongs to: the user's latest message and
    everything they said before it. Every call of one turn's tool loop
    shares it; the next message starts a new one."""
    last = None
    for i, content in enumerate(contents):
        if content.role == "user" and any(p.text and not p.thought for p in content.parts or []):
            last = i
    if last is None:
        return None
    digest = hashlib.sha1()
    for content in contents[: last + 1]:
        if content.role == "user":
            for part in content.parts or []:
                if part.text:
                    digest.update(part.text.encode("utf-8", "ignore"))
    return f"{last}:{digest.hexdigest()}"


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

ROUTE_LOCAL = "local"
ROUTE_HOSTED = "hosted"
_TURN_CACHE_CAP = 512


class ResilientLlm(BaseLlm):
    """Non-streaming, routing, retrying, per-event-loop wrapper.

    Each call goes to `local_model` when it is set, the server is up and the
    request fits (rule 5), else to `primary_model` (the hosted model), then
    to `backup_model` (the Gemini fallback). `model` (the BaseLlm field ADK
    copies into llm_request.model, and what ADK's tool gates inspect) is the
    primary's real id; the request is re-pointed at whichever model runs.
    """

    primary_model: str
    backup_model: Optional[str] = None
    local_model: Optional[str] = None
    max_tokens: int = MAX_OUTPUT_TOKENS
    # The reasoning effort asked of a LiteLLM model ("low", "medium",
    # "high"; "none" turns local thinking off). Gemini's thinking rides on
    # the agent's GenerateContentConfig instead.
    effort: Optional[str] = None

    _inner: Dict[Tuple[Any, str], BaseLlm] = PrivateAttr(default_factory=dict)
    _turns: "collections.OrderedDict[str, str]" = PrivateAttr(default_factory=collections.OrderedDict)

    def __init__(
        self,
        *,
        primary_model: str,
        backup_model: Optional[str] = None,
        local_model: Optional[str] = None,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        effort: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        for model_id in (primary_model, backup_model, local_model):
            if model_id:
                provider_for(model_id)
        if local_model and provider_for(local_model) != PROVIDER_LOCAL:
            raise UnsupportedModelError(f"LOCAL_AGENT_MODEL must be a local/<alias> id, got {local_model!r}")
        super().__init__(
            model=primary_model,
            primary_model=primary_model,
            backup_model=backup_model,
            local_model=local_model,
            max_tokens=max_tokens,
            effort=effort,
            **kwargs,
        )

    @classmethod
    @override
    def supported_models(cls) -> List[str]:
        # Never registered by pattern; always constructed explicitly.
        return []

    def _inner_for(self, model_id: str) -> BaseLlm:
        try:
            loop_key: Any = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_key = None
        key = (loop_key, model_id)
        inner = self._inner.get(key)
        if inner is None:
            if len(self._inner) >= _INNER_CACHE_CAP:
                # Dead loops leave entries behind; loops are few, so a
                # small cap bounds growth over a long-lived engine.
                self._inner.clear()
            inner = _build_inner(model_id, self.effort)
            self._inner[key] = inner
        return inner

    def _remember(self, turn: Optional[str], route: str) -> None:
        if turn is None:
            return
        self._turns[turn] = route
        self._turns.move_to_end(turn)
        while len(self._turns) > _TURN_CACHE_CAP:
            self._turns.popitem(last=False)

    async def _collect(
        self,
        model_id: str,
        llm_request: LlmRequest,
        snapshot: _RequestSnapshot,
        *,
        uncached: bool = False,
        estimate: Optional[int] = None,
    ) -> List[LlmResponse]:
        """Run one complete non-streaming call with transient retry.

        Every attempt starts from the snapshot (see `_RequestSnapshot`). A
        local call is tried at most twice and never after a timeout (rule 5).
        """
        provider = provider_for(model_id)
        inner = self._inner_for(model_id)
        attempts = 2 if provider == PROVIDER_LOCAL else _RETRY_ATTEMPTS
        for attempt in range(attempts):
            snapshot.restore(llm_request, uncached=uncached or provider != PROVIDER_GEMINI)
            llm_request.model = _local_litellm_id(model_id) if provider == PROVIDER_LOCAL else model_id
            if llm_request.config is not None:
                if llm_request.config.max_output_tokens is None:
                    llm_request.config.max_output_tokens = self.max_tokens
                if provider == PROVIDER_LOCAL:
                    llm_request.config.max_output_tokens = min(
                        llm_request.config.max_output_tokens, local_max_output_tokens()
                    )
            if provider == PROVIDER_GEMINI and llm_request.contents:
                stamp_thought_signatures(llm_request.contents)
            try:
                responses: List[LlmResponse] = []
                async for response in inner.generate_content_async(llm_request, stream=False):
                    responses.append(response)
                line = usage_line(model_id, responses, estimate)
                if line:
                    logger.info(line)
                strip_response_thoughts(responses)
                return responses
            except Exception as exc:  # noqa: BLE001 — inspect, re-raise non-transient
                last = attempt == attempts - 1
                if not _is_transient(exc) or last or (provider == PROVIDER_LOCAL and _is_timeout(exc)):
                    raise
                retry_after = retry_after_seconds(exc)
                pause = min(retry_after, _RETRY_AFTER_CAP_SECONDS) if retry_after is not None else 2.0 * (attempt + 1)
                logger.warning(
                    "Transient error from %s (attempt %d/%d, retrying in %gs): %s",
                    model_id, attempt + 1, attempts, pause, describe_error(exc),
                )
                await asyncio.sleep(pause)
        return []  # unreachable; keeps type checkers calm

    @staticmethod
    def _is_degenerate(responses: List[LlmResponse]) -> bool:
        """True when the final response carries neither text nor a tool call."""
        if not responses:
            return True
        final = responses[-1]
        if final.content and final.content.parts:
            for part in final.content.parts:
                if part.function_call:
                    return False
                if part.text and part.text.strip() and not part.thought:
                    return False
        return True

    async def _try_local(
        self, llm_request: LlmRequest, snapshot: _RequestSnapshot, turn: Optional[str], estimate: int
    ) -> Optional[List[LlmResponse]]:
        """The local leg of rule 5: the responses when local answered, else None."""
        if not self.local_model:
            return None
        budget = local_prompt_budget()
        if self._turns.get(turn) == ROUTE_HOSTED:
            return None
        if estimate > budget:
            reason = f"estimated prompt {estimate} > local budget {budget}"
        elif not await local_available():
            reason = "local server unavailable"
        else:
            try:
                responses = await self._collect(self.local_model, llm_request, snapshot, uncached=True, estimate=estimate)
            except Exception as exc:  # noqa: BLE001 — escalate on any local failure
                reason = describe_error(exc)
                if "APIConnectionError" in reason or _is_timeout(exc) or _status_code(exc) in (502, 503, 504, 524, 530):
                    mark_local_down(reason[:200])
            else:
                final = responses[-1] if responses else None
                if final is not None and final.finish_reason == types.FinishReason.MAX_TOKENS:
                    reason = f"ran out of output tokens ({local_max_output_tokens()})"
                elif self._is_degenerate(responses):
                    reason = "empty reply"
                else:
                    self._remember(turn, ROUTE_LOCAL)
                    return responses
            logger.warning("Local %s did not answer (%s); escalating to %s", self.local_model, reason, self.primary_model)
            self._remember(turn, ROUTE_HOSTED)
            return None
        logger.info("Routing to %s: %s", self.primary_model, reason)
        self._remember(turn, ROUTE_HOSTED)
        return None

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ):
        primary_error: Optional[BaseException] = None
        responses: List[LlmResponse] = []
        # Rules 7 and 8: repaired once here, so every model sees a clean history.
        if llm_request.contents:
            heal_orphaned_tool_calls(llm_request.contents)
            if drop_thought_parts(llm_request.contents):
                logger.info("Dropped thought parts from the history")
        # Taken after the repair, before any model sees the request.
        snapshot = _RequestSnapshot(llm_request)
        turn = _turn_key(llm_request.contents or [])
        estimate = estimate_request_tokens(llm_request)

        local = await self._try_local(llm_request, snapshot, turn, estimate)
        if local is not None:
            for response in local:
                yield response
            return

        try:
            for attempt in range(2):
                responses = await self._collect(self.primary_model, llm_request, snapshot, estimate=estimate)
                if not self._is_degenerate(responses):
                    for response in responses:
                        yield response
                    return
                finish = responses[-1].finish_reason if responses else None
                if attempt == 0:
                    # Rule 6: an empty STOP is usually a fluke; one more try on
                    # the model whose voice and judgement the agent is tuned to.
                    logger.warning(
                        "Primary %s returned an empty final response (finish_reason=%s); retrying it once",
                        self.primary_model, finish,
                    )
                    await asyncio.sleep(_DEGENERATE_RETRY_PAUSE_SECONDS)
                else:
                    logger.warning(
                        "Primary %s returned an empty final response again (finish_reason=%s)",
                        self.primary_model, finish,
                    )
        except Exception as exc:  # noqa: BLE001 — fall through to backup
            primary_error = exc
            logger.warning("Primary %s failed: %s", self.primary_model, describe_error(exc))

        if not self.backup_model:
            if primary_error is not None:
                raise primary_error
            for response in responses:
                yield response
            return

        logger.warning("Falling back to %s (uncached)", self.backup_model)
        try:
            responses = await self._collect(self.backup_model, llm_request, snapshot, uncached=True, estimate=estimate)
        except Exception as exc:  # noqa: BLE001 — surface both failures
            raise RuntimeError(
                f"primary {self.primary_model} failed ({describe_error(primary_error) if primary_error else 'empty reply'}); "
                f"backup {self.backup_model} failed ({describe_error(exc)})"
            ) from exc
        for response in responses:
            yield response


# ---------------------------------------------------------------------------
# Factories used by agent.py / custom_agents.py
# ---------------------------------------------------------------------------

def _routed(hosted: str, effort: Optional[str]) -> BaseLlm:
    fallback = fallback_model_id()
    return ResilientLlm(
        local_model=local_model_id(),
        primary_model=hosted,
        backup_model=fallback if fallback != hosted else None,
        effort=effort,
    )


def high_quality_model(effort: str = "high") -> BaseLlm:
    """The root orchestrator's model: local, else the hosted model, else the fallback.

    Pair `effort` with `high_quality_config(effort)` for the Gemini fallback.
    """
    return _routed(high_quality_model_id(), effort)


def specialist_model(effort: str = "medium") -> BaseLlm:
    """A specialist's model: local, else the specialist hosted model, else the fallback."""
    return _routed(specialist_model_id(), effort)


def quick_model() -> BaseLlm:
    """Kept for template compatibility; specialists should use specialist_model()."""
    return specialist_model()


def search_model() -> BaseLlm:
    """A grounded (google_search / url_context) sub-agent's model.

    ADK gates its grounding tools on the model name, so the primary and the
    backup must both be Gemini ids, whatever the root runs. The wrapper
    reports the primary's real id, which is what that gate inspects.
    """
    primary = search_model_id()
    backup = os.environ.get("SEARCH_BACKUP_MODEL", DEFAULT_SEARCH_BACKUP_MODEL)
    for model_id in (primary, backup):
        if provider_for(model_id) != PROVIDER_GEMINI:
            raise UnsupportedModelError(f"search models must be Gemini ids, got {model_id!r}")
    return ResilientLlm(
        primary_model=primary,
        backup_model=backup if backup != primary else None,
        max_tokens=8192,
    )


# ---------------------------------------------------------------------------
# Direct vision calls (read_attachment, image review)
# ---------------------------------------------------------------------------

def generate_vision(parts: List[types.Part], model_id: Optional[str] = None) -> str:
    """One non-streaming Gemini call over image/PDF parts, with transient retry.

    Synchronous on purpose: the callers already run in worker threads. Returns
    the response text; raises on a non-transient error or an empty response
    after retries.
    """
    from google import genai

    model = model_id or vision_model_id()
    if provider_for(model) != PROVIDER_GEMINI:
        raise UnsupportedModelError(f"VISION_MODEL must be a Gemini id, got {model!r}")
    last_error: Optional[BaseException] = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=parts)],
            )
            text = (response.text or "").strip()
            if text:
                return text
            last_error = RuntimeError(f"{model} returned an empty response")
        except Exception as exc:  # noqa: BLE001 — retry only transient
            if not _is_transient(exc):
                raise
            last_error = exc
        if attempt < _RETRY_ATTEMPTS - 1:
            logger.warning("Vision call to %s retrying (%s)", model, describe_error(last_error))
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Vision call to {model} failed: {describe_error(last_error) if last_error else 'no attempt ran'}")
