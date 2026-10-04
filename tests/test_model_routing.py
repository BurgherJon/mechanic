"""Offline tests for the routing wrapper in model_utils.py: no network, no keys.

The file is identical in every agent repo that shares model_utils.py. Run from
the repo root: python -m pytest tests/
"""
import asyncio, importlib.util, os, sys
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

MU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_utils.py")

@pytest.fixture
def mu(monkeypatch):
    for k in ("LOCAL_AGENT_MODEL", "HIGH_QUALITY_AGENT_MODEL", "SPECIALIST_AGENT_MODEL", "FALLBACK_AGENT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("mu_under_test", MU_PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

class Fake(BaseLlm):
    """Replays steps; records the request contents and model each call saw."""
    steps: list = []
    seen: list = []
    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(dict(model=llm_request.model,
                              sigs=[p.thought_signature for c in llm_request.contents for p in (c.parts or []) if p.function_call],
                              max_out=llm_request.config.max_output_tokens if llm_request.config else None))
        step = self.steps.pop(0)
        if isinstance(step, BaseException): raise step
        yield step

def reply(text, thought=None, finish=types.FinishReason.STOP):
    parts = ([types.Part(text=thought, thought=True)] if thought else []) + ([types.Part(text=text)] if text else [])
    return LlmResponse(content=types.Content(role="model", parts=parts), finish_reason=finish)

def wire(mu, monkeypatch, local=(), hosted=(), gemini=(), up=True):
    fakes = {"local": Fake(model="l", steps=list(local), seen=[]),
             "openrouter": Fake(model="o", steps=list(hosted), seen=[]),
             "gemini": Fake(model="g", steps=list(gemini), seen=[])}
    monkeypatch.setattr(mu, "_build_inner", lambda model_id, effort: fakes[mu.provider_for(model_id)])
    async def avail(): return up
    monkeypatch.setattr(mu, "local_available", avail)
    return fakes

def req(user_text="hello", extra_chars=0, history=()):
    contents = list(history) + [types.Content(role="user", parts=[types.Part(text=user_text + "x" * extra_chars)])]
    return LlmRequest(contents=contents, config=types.GenerateContentConfig(max_output_tokens=16384))

def run(llm, request):
    async def go(): return [r async for r in llm.generate_content_async(request)]
    return asyncio.run(go())

def text_of(responses): return "".join(p.text or "" for r in responses for p in (r.content.parts or []))

def test_local_answers_when_up_and_small_and_thoughts_are_stripped(mu, monkeypatch):
    f = wire(mu, monkeypatch, local=[reply("hi from local", thought="secret reasoning")])
    out = run(mu.high_quality_model(), req())
    assert text_of(out) == "hi from local"
    assert not any(p.thought for r in out for p in r.content.parts)
    assert f["local"].seen[0]["model"] == "openai/agent-large"
    assert f["local"].seen[0]["max_out"] == 8192
    assert f["openrouter"].seen == []

def test_turn_that_started_hosted_stays_hosted_when_local_comes_back(mu, monkeypatch):
    llm = mu.high_quality_model()
    f = wire(mu, monkeypatch, hosted=[reply("call 1"), reply("call 2")], local=[reply("never")], up=False)
    first = req("same turn")
    assert text_of(run(llm, first)) == "call 1"
    # local comes back mid-turn; the tool loop's next call (same user message,
    # history grown by a tool call and its result) stays on the hosted model
    async def up(): return True
    monkeypatch.setattr(mu, "local_available", up)
    call = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id="c1", name="t", args={}))])
    resp = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(id="c1", name="t", response={"ok": 1}))])
    second = LlmRequest(contents=[first.contents[0], call, resp], config=types.GenerateContentConfig(max_output_tokens=16384))
    assert text_of(run(llm, second)) == "call 2"
    assert f["local"].seen == []

def test_tool_result_too_big_escalates_mid_turn(mu, monkeypatch):
    llm = mu.high_quality_model()
    f = wire(mu, monkeypatch, local=[reply("local call 1")], hosted=[reply("hosted call 2")])
    first = req("q")
    assert text_of(run(llm, first)) == "local call 1"
    call = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id="c1", name="t", args={}))])
    resp = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(id="c1", name="t", response={"dump": "z" * 200000}))])
    second = LlmRequest(contents=[first.contents[0], call, resp], config=types.GenerateContentConfig(max_output_tokens=16384))
    assert text_of(run(llm, second)) == "hosted call 2"

def test_new_turn_tries_local_again(mu, monkeypatch):
    llm = mu.high_quality_model()
    f = wire(mu, monkeypatch, hosted=[reply("hosted")], local=[reply("local again")])
    run(llm, req("first", extra_chars=200_000))
    assert text_of(run(llm, req("second message"))) == "local again"

def test_local_down_goes_hosted(mu, monkeypatch):
    f = wire(mu, monkeypatch, hosted=[reply("hosted")], up=False)
    assert text_of(run(mu.high_quality_model(), req())) == "hosted"

def test_local_timeout_escalates_once_and_marks_down(mu, monkeypatch):
    class Timeout(Exception): pass
    f = wire(mu, monkeypatch, local=[Timeout("Request timed out")], hosted=[reply("hosted")])
    marked = []
    monkeypatch.setattr(mu, "mark_local_down", lambda reason: marked.append(reason))
    assert text_of(run(mu.high_quality_model(), req())) == "hosted"
    assert len(f["local"].seen) == 1 and marked  # not retried after a timeout

def test_local_truncated_escalates(mu, monkeypatch):
    f = wire(mu, monkeypatch, local=[reply("half an ans", finish=types.FinishReason.MAX_TOKENS)], hosted=[reply("full")])
    assert text_of(run(mu.high_quality_model(), req())) == "full"

def test_local_empty_escalates(mu, monkeypatch):
    f = wire(mu, monkeypatch, local=[reply("", thought="only thinking")], hosted=[reply("hosted")])
    assert text_of(run(mu.high_quality_model(), req())) == "hosted"

def test_hosted_failure_falls_back_to_gemini_and_only_gemini_gets_stamps(mu, monkeypatch):
    call = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(id="c1", name="t", args={}))])
    resp = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(id="c1", name="t", response={"ok": 1}))])
    class Bad(Exception): pass
    f = wire(mu, monkeypatch, local=[Bad("boom")], hosted=[Bad("400 bad request")], gemini=[reply("flash")])
    out = run(mu.high_quality_model(), req("q", history=[types.Content(role="user", parts=[types.Part(text="earlier")]), call, resp]))
    assert text_of(out) == "flash"
    assert f["local"].seen[0]["sigs"] == [None]
    assert f["openrouter"].seen[0]["sigs"] == [None]
    assert f["gemini"].seen[0]["sigs"] == [mu.SKIP_THOUGHT_SIGNATURE]

def test_local_off_when_env_empty(mu, monkeypatch):
    monkeypatch.setenv("LOCAL_AGENT_MODEL", "")
    f = wire(mu, monkeypatch, hosted=[reply("hosted")], local=[reply("never")])
    assert text_of(run(mu.high_quality_model(), req())) == "hosted"

def test_budget_and_estimate(mu):
    assert mu.local_prompt_budget() == int(65536 * 0.8) - 8192
    r = req("hi", history=[types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name="t", response={"rows": "y" * 30000}))])])
    assert mu.estimate_request_tokens(r) >= 10000  # tool output counts

@pytest.mark.parametrize("bad", ["claude-opus-5", "gemini-3.1-pro-preview", "local/", "mimo"])
def test_retired_or_unknown_ids_fail(mu, bad):
    with pytest.raises(mu.UnsupportedModelError):
        mu.provider_for(bad)

def test_search_model_must_be_gemini(mu, monkeypatch):
    monkeypatch.setenv("SEARCH_AGENT_MODEL", "openrouter/xiaomi/mimo-v2.6-pro")
    with pytest.raises(mu.UnsupportedModelError):
        mu.search_model()

@pytest.mark.parametrize("value", ["off", "OFF", "none", ""])
def test_local_off_switch(mu, monkeypatch, value):
    monkeypatch.setenv("LOCAL_AGENT_MODEL", value)
    assert mu.local_model_id() is None
    assert mu.high_quality_model().local_model is None
