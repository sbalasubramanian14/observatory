import httpx
import pytest
from feed.providers.base import (
    PaymentRequiredError,
    ProviderError,
    RateLimitError,
    TransientProviderError,
)
from feed.providers.openai_compatible import OpenAICompatibleProvider


def _make(monkeypatch, **kwargs):
    monkeypatch.setenv("FAKE_KEY", "k")
    defaults = dict(
        name="groq", model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1", env_var="FAKE_KEY",
    )
    defaults.update(kwargs)
    return OpenAICompatibleProvider(**defaults)


def _resp(status_code: int, body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    return httpx.Response(status_code, json=body or {}, request=request)


def test_complete_returns_message_content(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json_body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json_body"] = json_body
        return {"choices": [{"message": {"content": "hello world"}}]}

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch)

    result = provider.complete("say hi")

    assert result == "hello world"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer k"}
    assert captured["json_body"] == {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "say hi"}],
    }


def test_missing_env_var_raises_without_a_network_call(monkeypatch):
    def fake_post(*a, **k):
        raise AssertionError("must not be called when the key is missing")

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
    provider = OpenAICompatibleProvider(
        name="x", model="m", base_url="https://x.test/v1", env_var="MISSING_KEY_VAR",
    )

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_429_raises_rate_limit_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.HTTPStatusError("429", request=None, response=_resp(429))

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch)

    with pytest.raises(RateLimitError):
        provider.complete("hi")


def test_402_raises_payment_required_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.HTTPStatusError("402", request=None, response=_resp(402))

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch)

    with pytest.raises(PaymentRequiredError):
        provider.complete("hi")


def test_503_raises_transient_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.HTTPStatusError("503", request=None, response=_resp(503))

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch, max_retries=0)

    with pytest.raises(TransientProviderError):
        provider.complete("hi")


def test_timeout_raises_transient_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch, max_retries=0)

    with pytest.raises(TransientProviderError):
        provider.complete("hi")


def test_connection_error_raises_transient_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch, max_retries=0)

    with pytest.raises(TransientProviderError):
        provider.complete("hi")


def test_transient_error_is_retried_and_can_succeed(monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.HTTPStatusError("503", request=None, response=_resp(503))
        return {"choices": [{"message": {"content": "ok on 3rd try"}}]}

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch, max_retries=2)

    result = provider.complete("hi")

    assert result == "ok on 3rd try"
    assert calls["n"] == 3


def test_401_is_a_plain_provider_error_not_retried_or_rate_limited(monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        raise httpx.HTTPStatusError("401", request=None, response=_resp(401))

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch, max_retries=2)

    with pytest.raises(ProviderError) as exc_info:
        provider.complete("hi")

    assert not isinstance(exc_info.value, (RateLimitError, PaymentRequiredError,
                                          TransientProviderError))
    assert calls["n"] == 1  # not retried -- a bad key won't fix itself


def test_strips_think_block_from_a_realistic_reasoning_payload(monkeypatch):
    """Verified live: qwen/qwen3.6-27b emits visible <think>...</think> in
    its output. This is a realistic full-response payload shape (reasoning
    preamble, then the actual answer) that must come back clean."""
    raw = (
        "<think>\nLet me consider the headline. The user wants a JSON "
        "object with headline/summary/category. I should not include "
        "markdown fences.\n</think>\n"
        '{"headline": "Model ships", "summary": "A thing happened.", '
        '"category": "product"}'
    )

    def fake_post(*a, **k):
        return {"choices": [{"message": {"content": raw}}]}

    monkeypatch.setattr("feed.providers.openai_compatible._post", fake_post)
    provider = _make(monkeypatch)

    result = provider.complete("summarize")

    assert "<think>" not in result
    assert "</think>" not in result
    assert "consider the headline" not in result
    assert result.startswith("{")
    import json
    parsed = json.loads(result)
    assert parsed["headline"] == "Model ships"


def test_health_reflects_api_key_presence(monkeypatch):
    monkeypatch.delenv("MISSING_KEY_VAR2", raising=False)
    provider = OpenAICompatibleProvider(
        name="x", model="m", base_url="https://x.test/v1", env_var="MISSING_KEY_VAR2",
    )
    assert provider.health().healthy is False

    provider2 = _make(monkeypatch)
    assert provider2.health().healthy is True
