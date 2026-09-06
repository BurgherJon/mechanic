"""Model construction for Maggie: Claude on the first-party Anthropic API,
Gemini for the direct vision helpers, and one resilience wrapper around
either.

Fleet standard (2026-09-06, see the Comites Model Roster):

    HIGH_QUALITY_AGENT_MODEL   claude-opus-5     root orchestrator
    SPECIALIST_AGENT_MODEL     claude-sonnet-5   sub-agents; also the root's backup
    VISION_MODEL               gemini-3.8-flash  direct genai extraction calls
    ANTHROPIC_SECRET_NAME      {BOT_ACCOUNT_ID}-anthropic-key in AGENT_PROJECT_ID

Rules this module encodes, each learned in production:

1. Never hand a bare "claude-*" string to Agent(model=...). ADK's registry
   maps those to its Vertex `Claude` class, and this estate has no Claude
   quota on Vertex. Claude is built explicitly as `AnthropicLlm` with a
   client that carries the first-party API key from Secret Manager.
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
5. Retry transient serving errors (429 / 500 / 503 / 529) before failing a
   turn, then fall back to the backup model on an exception or an empty
   final response. An empty final response is also how a Claude refusal
   surfaces through ADK, so the backup absorbs that case too.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr
from typing_extensions import override

from .secret_utilities import get_secret_from_secret_manager

logger = logging.getLogger(__name__)

DEFAULT_HIGH_QUALITY_MODEL = "claude-opus-5"
DEFAULT_SPECIALIST_MODEL = "claude-sonnet-5"
DEFAULT_VISION_MODEL = "gemini-3.8-flash"

# Output cap for every Claude agent. These calls are non-streaming, so the
# cap must leave the response inside the HTTP request timeout; 16384 covers
# the longest outputs in the estate (Sam's ~4000-word memory rewrite).
CLAUDE_MAX_OUTPUT_TOKENS = 16384

# Substrings that mark a transient serving failure worth retrying. Matched
# against str(exception) because Anthropic, google-genai and httpx each raise
# their own types. 500/INTERNAL were absent before 2026-09-06 and let a
# transient Vertex 500 kill a deploy verification.
_RETRYABLE_MARKERS = (
    "429", "RESOURCE_EXHAUSTED", "rate_limit",
    "500", "INTERNAL",
    "503", "UNAVAILABLE",
    "529", "overloaded", "Overloaded",
)
_RETRY_ATTEMPTS = 3
_INNER_CACHE_CAP = 8

_secret_cache: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Model ids and config
# ---------------------------------------------------------------------------

def high_quality_model_id() -> str:
    return os.environ.get("HIGH_QUALITY_AGENT_MODEL", DEFAULT_HIGH_QUALITY_MODEL)


def specialist_model_id() -> str:
    return os.environ.get("SPECIALIST_AGENT_MODEL", DEFAULT_SPECIALIST_MODEL)


def vision_model_id() -> str:
    return os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL)


def is_claude(model_id: str) -> bool:
    return model_id.startswith("claude-")


def _agent_config(model_id: str, effort: Optional[str]) -> types.GenerateContentConfig:
    """Per-agent generation config for the given model.

    Claude gets the effort level (adaptive thinking is on by default on
    Claude 5 models, so nothing else is needed) plus the output cap. Any
    other model gets a plain config, because the Anthropic subclass carries
    an `effort` field Gemini would reject.
    """
    if is_claude(model_id):
        from google.adk.models.anthropic_llm import AnthropicGenerateContentConfig

        return AnthropicGenerateContentConfig(
            effort=effort, max_output_tokens=CLAUDE_MAX_OUTPUT_TOKENS
        )
    return types.GenerateContentConfig()


def high_quality_config(effort: str = "high") -> types.GenerateContentConfig:
    return _agent_config(high_quality_model_id(), effort)


def specialist_config(effort: str = "medium") -> types.GenerateContentConfig:
    return _agent_config(specialist_model_id(), effort)


# ---------------------------------------------------------------------------
# Inner model construction
# ---------------------------------------------------------------------------

def _anthropic_api_key() -> str:
    if "anthropic" not in _secret_cache:
        project = os.environ.get("AGENT_PROJECT_ID") or os.environ["GOOGLE_CLOUD_PROJECT"]
        secret = os.environ.get("ANTHROPIC_SECRET_NAME") or (
            f"{os.environ['BOT_ACCOUNT_ID']}-anthropic-key"
        )
        _secret_cache["anthropic"] = get_secret_from_secret_manager(project, secret)
    return _secret_cache["anthropic"]


def _build_inner(model_id: str, max_tokens: int) -> BaseLlm:
    if is_claude(model_id):
        from anthropic import AsyncAnthropic
        from google.adk.models.anthropic_llm import AnthropicLlm

        return AnthropicLlm(
            model=model_id,
            max_tokens=max_tokens,
            client=AsyncAnthropic(api_key=_anthropic_api_key()),
        )
    from google.adk.models.google_llm import Gemini

    return Gemini(model=model_id)


def _is_transient(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_MARKERS)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

class ResilientLlm(BaseLlm):
    """Non-streaming, retrying, per-event-loop wrapper with a backup model.

    `model` (the BaseLlm field ADK copies into llm_request.model) is the
    primary's real id, so provider calls and tool-support checks see the
    true model name. The request is re-pointed at the backup only when the
    backup actually runs.
    """

    primary_model: str
    backup_model: Optional[str] = None
    max_tokens: int = CLAUDE_MAX_OUTPUT_TOKENS

    _inner: Dict[Tuple[Any, str], BaseLlm] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        *,
        primary_model: str,
        backup_model: Optional[str] = None,
        max_tokens: int = CLAUDE_MAX_OUTPUT_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=primary_model,
            primary_model=primary_model,
            backup_model=backup_model,
            max_tokens=max_tokens,
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
            inner = _build_inner(model_id, self.max_tokens)
            self._inner[key] = inner
        return inner

    async def _collect(self, model_id: str, llm_request: LlmRequest) -> List[LlmResponse]:
        """Run one complete non-streaming call with transient retry."""
        llm_request.model = model_id
        inner = self._inner_for(model_id)
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                responses: List[LlmResponse] = []
                async for response in inner.generate_content_async(llm_request, stream=False):
                    responses.append(response)
                return responses
            except Exception as exc:  # noqa: BLE001 — inspect, re-raise non-transient
                if not _is_transient(exc) or attempt == _RETRY_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Transient error from %s (attempt %d/%d): %s",
                    model_id, attempt + 1, _RETRY_ATTEMPTS, str(exc)[:200],
                )
                await asyncio.sleep(2 * (attempt + 1))
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

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ):
        primary_error: Optional[BaseException] = None
        responses: List[LlmResponse] = []
        try:
            responses = await self._collect(self.primary_model, llm_request)
            if not self._is_degenerate(responses):
                for response in responses:
                    yield response
                return
            finish = responses[-1].finish_reason if responses else None
            logger.warning(
                "Primary %s returned an empty final response (finish_reason=%s)",
                self.primary_model, finish,
            )
        except Exception as exc:  # noqa: BLE001 — fall through to backup
            primary_error = exc
            logger.warning(
                "Primary %s failed: %s: %s",
                self.primary_model, type(exc).__name__, str(exc)[:300],
            )

        if not self.backup_model:
            if primary_error is not None:
                raise primary_error
            for response in responses:
                yield response
            return

        logger.warning("Falling back to %s", self.backup_model)
        try:
            responses = await self._collect(self.backup_model, llm_request)
        except Exception as exc:  # noqa: BLE001 — surface both failures
            raise RuntimeError(
                f"primary {self.primary_model} failed ({primary_error}); "
                f"backup {self.backup_model} failed ({exc})"
            ) from exc
        for response in responses:
            yield response


# ---------------------------------------------------------------------------
# Factories used by agent.py / custom_agents.py
# ---------------------------------------------------------------------------

def high_quality_model() -> BaseLlm:
    """The root orchestrator's model, backed by the specialist model."""
    primary = high_quality_model_id()
    backup = specialist_model_id()
    return ResilientLlm(
        primary_model=primary,
        backup_model=backup if backup != primary else None,
    )


def specialist_model() -> BaseLlm:
    """A specialist's model, backed by the high-quality model."""
    primary = specialist_model_id()
    backup = high_quality_model_id()
    return ResilientLlm(
        primary_model=primary,
        backup_model=backup if backup != primary else None,
    )


def quick_model() -> BaseLlm:
    """Kept for template compatibility; specialists should use specialist_model()."""
    return specialist_model()


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
            logger.warning("Vision call to %s retrying (%s)", model, str(last_error)[:200])
            import time

            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Vision call to {model} failed: {last_error}")
