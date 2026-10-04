"""Model construction: Gemini on Vertex AI through ADK's native `Gemini`
class, behind one resilience wrapper.

Defaults (2026-09-28; override in .env):

    HIGH_QUALITY_AGENT_MODEL   gemini-3.1-pro-preview  root orchestrator
    SPECIALIST_AGENT_MODEL     gemini-3.8-flash        sub-agents; also the root's backup
    VISION_MODEL               gemini-3.8-flash        direct genai extraction calls
    SEARCH_AGENT_MODEL         gemini-3.8-flash        sub-agents using google_search/url_context

Gemini 3.1 Pro is served on Vertex only under its preview id and only on
the `global` endpoint, which agent.py forces before the Google imports.
No API key: the engine calls Vertex as its own service account.

Rules this module encodes, each learned in production:

1. Never hand a bare model string to Agent(model=...). Every model goes
   through a factory here so the wrapper below is always in the path.
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
   turn. An empty final response (no text, no tool call) gets one retry of
   the primary after a short pause, because a mid-turn change of model is
   worse than a fluke. Only then the backup, and the backup runs UNCACHED:
   ADK's context cache rewrites the request in place for the model that
   owns the cache, and Vertex refuses that cache for another model with
   400 "model in the inference request does not match the model in the
   cached content". So the request is snapshotted before the first call
   and restored before every further call, and the backup's copy carries
   no cache at all.
6. Heal orphaned tool calls before every model call, so a turn the
   platform cut off cannot poison the rest of the session.
7. Before every model call, drop thought parts left by the Claude era and
   stamp unsigned function calls with Gemini's dummy thought signature.
   Gemini 3 rejects a foreign signature ("Invalid thought signature", seen
   2026-09-28 on 3.8 Flash; 3.1 Pro rejects Claude's redacted thinking)
   and validates a signature on every function call in the current turn.
8. Log one `model usage` line per completed call with the prompt, cached,
   thought and output token counts.
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

logger = logging.getLogger(__name__)

DEFAULT_HIGH_QUALITY_MODEL = "gemini-3.1-pro-preview"
DEFAULT_SPECIALIST_MODEL = "gemini-3.8-flash"
DEFAULT_VISION_MODEL = "gemini-3.8-flash"
# Sub-agents that use ADK's google_search / url_context are gated to Gemini.
DEFAULT_SEARCH_MODEL = "gemini-3.8-flash"
DEFAULT_SEARCH_BACKUP_MODEL = "gemini-3.5-flash"

# Output cap for every agent. These calls are non-streaming, so the cap must
# leave the response inside the HTTP request timeout; 16384 covers the
# longest outputs in the estate (Sam's ~4000-word memory rewrite).
MAX_OUTPUT_TOKENS = 16384

# An agent's `effort` maps onto Gemini's thinking level. Gemini 3.1 Pro
# takes LOW / MEDIUM / HIGH; 3.8 Flash takes those and MINIMAL.
_THINKING_LEVELS = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}

# Substrings that mark a transient serving failure worth retrying. Matched
# against str(exception) because google-genai and httpx each raise their own
# types. 500/INTERNAL were absent before 2026-09-06 and let a transient
# Vertex 500 kill a deploy verification.
_RETRYABLE_MARKERS = (
    "429", "RESOURCE_EXHAUSTED", "rate_limit",
    "500", "INTERNAL",
    "503", "UNAVAILABLE",
    "529", "overloaded", "Overloaded",
)
_RETRY_ATTEMPTS = 3
# Pause before the one retry of a primary that returned an empty final response.
_DEGENERATE_RETRY_PAUSE_SECONDS = 2.0
_INNER_CACHE_CAP = 8


# ---------------------------------------------------------------------------
# Model ids and config
# ---------------------------------------------------------------------------

def high_quality_model_id() -> str:
    return os.environ.get("HIGH_QUALITY_AGENT_MODEL", DEFAULT_HIGH_QUALITY_MODEL)


def specialist_model_id() -> str:
    return os.environ.get("SPECIALIST_AGENT_MODEL", DEFAULT_SPECIALIST_MODEL)


def vision_model_id() -> str:
    return os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL)


def search_model_id() -> str:
    return os.environ.get("SEARCH_AGENT_MODEL", DEFAULT_SEARCH_MODEL)


def _require_gemini(model_id: str) -> str:
    """Fail at import, not on the first turn, when .env names another provider."""
    if not model_id.startswith("gemini-"):
        raise ValueError(
            f"model {model_id!r}: only Gemini on Vertex AI is wired up. Set "
            "HIGH_QUALITY_AGENT_MODEL / SPECIALIST_AGENT_MODEL / "
            "SEARCH_AGENT_MODEL in .env to a gemini-* id."
        )
    return model_id


def _agent_config(model_id: str, effort: Optional[str]) -> types.GenerateContentConfig:
    """Per-agent generation config: the output cap and the thinking level."""
    _require_gemini(model_id)
    thinking = None
    if effort:
        thinking = types.ThinkingConfig(thinking_level=_THINKING_LEVELS[effort])
    return types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS, thinking_config=thinking
    )


def high_quality_config(effort: str = "high") -> types.GenerateContentConfig:
    return _agent_config(high_quality_model_id(), effort)


def specialist_config(effort: str = "medium") -> types.GenerateContentConfig:
    return _agent_config(specialist_model_id(), effort)


# ---------------------------------------------------------------------------
# Inner model construction
# ---------------------------------------------------------------------------

def _build_inner(model_id: str) -> BaseLlm:
    from google.adk.models.google_llm import Gemini

    return Gemini(model=_require_gemini(model_id))


def _is_transient(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_MARKERS)


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


# Gemini 3 validates a thought signature on every function call in the
# current turn. A call made by another model (Claude, or an older Gemini)
# carries none, so the turn is rejected with 400 "Function call is missing a
# thought_signature". Google's documented escape for history that came from
# elsewhere is this dummy value, which tells the validator to skip the part.
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
    """The parts of an `LlmRequest` a Gemini call rewrites in place.

    ADK's context cache manager applies an active cache by stripping the
    system instruction, tools and tool config from `config`, setting
    `config.cached_content` and cutting the cached prefix off `contents`;
    the Gemini model also appends a user content when the history ends on
    the model's side. A second call on the same object, whether a retry or
    the backup, must start from the request as the flow built it, so it is
    captured here before the first call and put back before every later one.
    """

    def __init__(self, llm_request: LlmRequest):
        self.contents = list(llm_request.contents or [])
        self.config = llm_request.config.model_copy() if llm_request.config is not None else None
        self.cache_config = llm_request.cache_config
        self.cache_metadata = llm_request.cache_metadata
        self.cacheable_contents_token_count = llm_request.cacheable_contents_token_count

    def restore(self, llm_request: LlmRequest, *, uncached: bool = False) -> None:
        """Put the request back; `uncached` also drops every trace of the cache.

        The cache belongs to the model that built it. The backup gets the
        whole history and its own system instruction and tools, and no
        `cache_config`, so ADK neither reuses the primary's cache nor tries
        to build one for a call that runs once in a long while.
        """
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


def usage_line(model_id: str, responses: List[Any]) -> Optional[str]:
    """One line of token accounting for a completed call, or None.

    Cached tokens are the prompt prefix Vertex served from its cache; a
    non-zero count on the second turn of a session proves the prefix
    survived.
    """
    usage = None
    for response in reversed(list(responses)):
        if getattr(response, "usage_metadata", None) is not None:
            usage = response.usage_metadata
            break
    if usage is None:
        return None
    return (
        f"model usage {model_id}: prompt={usage.prompt_token_count or 0} "
        f"cached={usage.cached_content_token_count or 0} "
        f"thoughts={usage.thoughts_token_count or 0} "
        f"output={usage.candidates_token_count or 0}"
    )


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
    max_tokens: int = MAX_OUTPUT_TOKENS

    _inner: Dict[Tuple[Any, str], BaseLlm] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        *,
        primary_model: str,
        backup_model: Optional[str] = None,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        **kwargs: Any,
    ) -> None:
        _require_gemini(primary_model)
        if backup_model:
            _require_gemini(backup_model)
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
            inner = _build_inner(model_id)
            self._inner[key] = inner
        return inner

    async def _collect(
        self,
        model_id: str,
        llm_request: LlmRequest,
        snapshot: _RequestSnapshot,
        *,
        uncached: bool = False,
    ) -> List[LlmResponse]:
        """Run one complete non-streaming call with transient retry.

        Every attempt starts from the snapshot (see `_RequestSnapshot`): the
        previous attempt may have rewritten the request for its cache.
        """
        inner = self._inner_for(model_id)
        for attempt in range(_RETRY_ATTEMPTS):
            snapshot.restore(llm_request, uncached=uncached)
            llm_request.model = model_id
            if llm_request.config is not None and llm_request.config.max_output_tokens is None:
                llm_request.config.max_output_tokens = self.max_tokens
            try:
                responses: List[LlmResponse] = []
                async for response in inner.generate_content_async(llm_request, stream=False):
                    responses.append(response)
                line = usage_line(model_id, responses)
                if line:
                    logger.info(line)
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
        # Rules 6 and 7: repaired once here, so primary and backup both see
        # a history Gemini accepts.
        if llm_request.contents:
            heal_orphaned_tool_calls(llm_request.contents)
            if drop_thought_parts(llm_request.contents):
                logger.info("Dropped thought parts from a pre-Gemini history")
            stamp_thought_signatures(llm_request.contents)
        # Taken after the repair, before any model sees the request.
        snapshot = _RequestSnapshot(llm_request)
        try:
            for attempt in range(2):
                responses = await self._collect(self.primary_model, llm_request, snapshot)
                if not self._is_degenerate(responses):
                    for response in responses:
                        yield response
                    return
                finish = responses[-1].finish_reason if responses else None
                if attempt == 0:
                    # Rule 5: an empty STOP is usually a fluke; one more try on
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

        logger.warning("Falling back to %s (uncached)", self.backup_model)
        try:
            responses = await self._collect(self.backup_model, llm_request, snapshot, uncached=True)
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


def search_model() -> BaseLlm:
    """A grounded (google_search / url_context) sub-agent's model.

    ADK gates its grounding tools on the model name; the wrapper reports the
    primary's real id, which is what that gate inspects.
    """
    primary = search_model_id()
    backup = os.environ.get("SEARCH_BACKUP_MODEL", DEFAULT_SEARCH_BACKUP_MODEL)
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
    import time

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
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Vision call to {model} failed: {last_error}")
