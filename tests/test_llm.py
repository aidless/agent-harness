"""LLM client: payload construction, response parsing, error mapping (mock httpx)."""

import threading

import httpx
import pytest

from agent_harness.config import Config
from agent_harness.llm import LLM, LLMError


class FakeTransport(httpx.BaseTransport):
    def __init__(self, status=200, body=None, raise_error=None):
        self.status = status
        self.body = body if body is not None else {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }
        self.raise_error = raise_error
        self.last_request = None

    def handle_request(self, request):
        self.last_request = request
        if self.raise_error:
            raise self.raise_error
        return httpx.Response(self.status, json=self.body)


def make_llm(transport, cfg=None):
    if cfg is None:
        # No retries by default so error-path tests stay fast; retry behaviour
        # is tested explicitly below with a custom config.
        cfg = Config(base_url="http://x/v1", model="m", max_retries=0)
    llm = LLM(cfg)
    # Keep the real Authorization header, swap only the transport.
    llm.client = httpx.Client(transport=transport, headers=dict(llm.client.headers))
    return llm


def test_url_joining_strips_trailing_slash():
    llm = LLM(Config(base_url="http://x/v1///", model="m"))
    assert llm.url == "http://x/v1/chat/completions"
    llm.close()


def test_payload_basics():
    tr = FakeTransport()
    llm = make_llm(tr)
    msg = llm.chat([{"role": "user", "content": "q"}])
    assert msg["content"] == "hi"
    import json
    payload = json.loads(tr.last_request.content)
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.0
    assert "tools" not in payload
    assert tr.last_request.headers["authorization"] == "Bearer local"


def test_payload_with_tools():
    tr = FakeTransport()
    llm = make_llm(tr)
    tools = [{"type": "function", "function": {"name": "bash"}}]
    llm.chat([{"role": "user", "content": "q"}], tools)
    import json
    payload = json.loads(tr.last_request.content)
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


def test_non_200_raises():
    tr = FakeTransport(status=500)
    llm = make_llm(tr)
    with pytest.raises(LLMError, match="HTTP 500"):
        llm.chat([])


def test_bad_body_raises():
    tr = FakeTransport(body={"no": "choices"})
    llm = make_llm(tr)
    with pytest.raises(LLMError, match="bad response"):
        llm.chat([])


def test_network_error_raises():
    tr = FakeTransport(raise_error=httpx.ConnectError("down"))
    llm = make_llm(tr)
    with pytest.raises(LLMError, match="request failed"):
        llm.chat([])


def test_context_manager_closes():
    tr = FakeTransport()
    with make_llm(tr) as llm:
        llm.chat([])
    assert llm.client.is_closed


def test_usage_callback_and_last_usage():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }
    tr = FakeTransport(body=body)
    seen = []
    llm = make_llm(tr, Config(base_url="http://x/v1", model="m"))
    llm.usage_callback = seen.append
    llm.chat([])
    assert llm.last_usage == {
        "model": "m", "input_tokens": 1200,
        "output_tokens": 300, "cache_read_tokens": 100,
    }
    assert seen == [llm.last_usage]


def test_usage_coerces_bad_counts_and_no_callback_when_zero():
    body = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": "oops", "completion_tokens": -5},
    }
    tr = FakeTransport(body=body)
    seen = []
    llm = make_llm(tr)
    llm.usage_callback = seen.append
    llm.chat([])
    assert llm.last_usage["input_tokens"] == 0
    assert llm.last_usage["output_tokens"] == 0
    assert seen == []  # no callback when all counts are zero


# ---- retry / backoff ----------------------------------------------------

class SequenceTransport(httpx.BaseTransport):
    """Return a scripted sequence of responses/errors, recording call count."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, body, headers = outcome
        return httpx.Response(status, json=body, headers=headers or {})


def ok_body():
    return {"choices": [{"message": {"content": "ok"}}]}


def fast_cfg(**kw):
    kw.setdefault("retry_base_delay", 0.001)
    kw.setdefault("retry_max_delay", 0.002)
    return Config(base_url="http://x/v1", model="m", **kw)


def test_retry_on_429_then_success():
    tr = SequenceTransport([(429, {"error": "slow"}, {"retry-after": "0"}),
                            (200, ok_body(), {})])
    llm = make_llm(tr, fast_cfg(max_retries=3))
    assert llm.chat([])["content"] == "ok"
    assert tr.calls == 2
    assert llm.retries_used == 1


def test_retry_exhausted_raises_last_status():
    tr = SequenceTransport([(503, {}, {}), (503, {}, {}), (503, {}, {})])
    llm = make_llm(tr, fast_cfg(max_retries=2))
    with pytest.raises(LLMError, match="HTTP 503"):
        llm.chat([])
    assert tr.calls == 3  # first attempt + 2 retries
    assert llm.retries_used == 2


def test_no_retry_on_permanent_400():
    tr = SequenceTransport([(400, {"error": "bad request"}, {})])
    llm = make_llm(tr, fast_cfg(max_retries=5))
    with pytest.raises(LLMError, match="HTTP 400"):
        llm.chat([])
    assert tr.calls == 1
    assert llm.retries_used == 0


def test_retry_on_network_error_then_success():
    tr = SequenceTransport([httpx.ConnectError("down"), (200, ok_body(), {})])
    llm = make_llm(tr, fast_cfg(max_retries=2))
    assert llm.chat([])["content"] == "ok"
    assert tr.calls == 2
    assert llm.retries_used == 1


def test_retry_counts_toward_last_usage_from_final_response():
    tr = SequenceTransport([(429, {}, {"retry-after": "0"}),
                            (200, {**ok_body(), "usage": {"prompt_tokens": 7,
                                                          "completion_tokens": 3}}, {})])
    llm = make_llm(tr, fast_cfg(max_retries=2))
    llm.chat([])
    assert llm.last_usage["input_tokens"] == 7
    assert llm.last_usage["output_tokens"] == 3


def test_abort_stops_retry_wait():
    tr = SequenceTransport([(429, {}, {}), (200, ok_body(), {})])
    cfg = fast_cfg(max_retries=3, retry_base_delay=5.0, retry_max_delay=5.0)
    llm = make_llm(tr, cfg)
    # Abort from a timer thread while the first backoff wait is in progress.
    threading.Timer(0.02, llm.abort).start()
    with pytest.raises(LLMError, match="aborted"):
        llm.chat([])
    assert tr.calls == 1  # never got to the second attempt
