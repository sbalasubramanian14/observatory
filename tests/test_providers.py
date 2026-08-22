import subprocess
import pytest
from feed.providers.base import ProviderError, ProviderHealth, Tier
from feed.providers.claude_code import ClaudeCodeProvider
from feed.providers.gemini import GeminiProvider
from feed.providers.router import RouteResult, Router


# --- GeminiProvider -----------------------------------------------------

def test_gemini_complete_returns_text(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json_body, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json_body"] = json_body
        return {"candidates": [{"content": {"parts": [{"text": "hello world"}]}}]}

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="test-key", model="gemini-flash-latest")

    result = provider.complete("say hi")

    assert result == "hello world"
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-flash-latest:generateContent"
    )
    assert captured["headers"] == {"x-goog-api-key": "test-key"}
    assert captured["json_body"] == {"contents": [{"parts": [{"text": "say hi"}]}]}


def test_gemini_missing_api_key_raises_without_a_network_call(monkeypatch, tmp_path):
    def fake_post(*a, **k):
        raise AssertionError("must not be called when the key is missing")

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key=None)

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_gemini_wraps_transport_failures_as_provider_error(monkeypatch):
    def fake_post(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="k")

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_gemini_health_reflects_api_key_presence(monkeypatch, tmp_path):
    assert GeminiProvider(api_key="k").health() == ProviderHealth(healthy=True)

    # No .env and no real env var in scope, so the "missing key" branch is
    # actually exercised regardless of what this dev machine's own .env
    # (outside the test tree) happens to contain.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    h = GeminiProvider(api_key=None).health()
    assert h.healthy is False


def test_gemini_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    provider = GeminiProvider()
    assert provider.api_key == "env-key"


def test_gemini_dotenv_does_not_override_a_real_env_var(monkeypatch, tmp_path):
    """load_dotenv must never clobber an already-set env var -- an operator
    passing GEMINI_API_KEY=... on the command line must win over .env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from-real-env")

    provider = GeminiProvider()

    assert provider.api_key == "from-real-env"


def test_gemini_dotenv_is_used_when_no_real_env_var(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('GEMINI_API_KEY="from-dotenv"\n', encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = GeminiProvider()

    assert provider.api_key == "from-dotenv"


# --- ClaudeCodeProvider ---------------------------------------------------

def test_claude_code_complete_returns_stripped_stdout(monkeypatch):
    def fake_run(args, *, timeout):
        assert args == ["claude", "-p", "why does this matter?"]
        return subprocess.CompletedProcess(args, 0, stdout="  analysis text  \n", stderr="")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    assert provider.complete("why does this matter?") == "analysis text"


def test_claude_code_nonzero_exit_is_a_provider_error_not_a_crash(monkeypatch):
    def fake_run(args, *, timeout):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="rate limited")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_claude_code_timeout_is_a_provider_error_not_a_crash(monkeypatch):
    def fake_run(args, *, timeout):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider(timeout=1.0)

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_claude_code_empty_output_is_a_provider_error(monkeypatch):
    def fake_run(args, *, timeout):
        return subprocess.CompletedProcess(args, 0, stdout="   ", stderr="")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    with pytest.raises(ProviderError):
        provider.complete("hi")


# --- Router ---------------------------------------------------------------

class _StubProvider:
    def __init__(self, name, model, tier, *, text="ok", healthy=True, fail=False):
        self.name = name
        self.model = model
        self.tier = tier
        self._text = text
        self._healthy = healthy
        self._fail = fail
        self.calls: list[str] = []

    def complete(self, prompt, *, schema=None):
        self.calls.append(prompt)
        if self._fail:
            raise ProviderError(f"{self.name}: boom")
        return self._text

    def health(self):
        return ProviderHealth(healthy=self._healthy)


def test_router_bulk_request_always_uses_bulk_provider():
    bulk = _StubProvider("gemini", "gemini-flash-latest", Tier.BULK, text="bulk answer")
    deep = _StubProvider("claude-code", "claude-code", Tier.DEEP, text="deep answer")
    router = Router(bulk=bulk, deep=deep)

    result = router.complete("prompt", tier=Tier.BULK)

    assert result == RouteResult(text="bulk answer", provider="gemini",
                                 model="gemini-flash-latest", tier=Tier.BULK,
                                 degraded=False)
    assert deep.calls == []


def test_router_deep_request_uses_deep_provider_when_healthy():
    bulk = _StubProvider("gemini", "gemini-flash-latest", Tier.BULK)
    deep = _StubProvider("claude-code", "claude-code", Tier.DEEP, text="deep answer")
    router = Router(bulk=bulk, deep=deep)

    result = router.complete("prompt", tier=Tier.DEEP)

    assert result.text == "deep answer"
    assert result.provider == "claude-code"
    assert result.tier is Tier.DEEP
    assert result.degraded is False
    assert bulk.calls == []


def test_router_degrades_to_bulk_when_deep_is_unhealthy():
    bulk = _StubProvider("gemini", "gemini-flash-latest", Tier.BULK, text="bulk fallback")
    deep = _StubProvider("claude-code", "claude-code", Tier.DEEP, healthy=False)
    router = Router(bulk=bulk, deep=deep)

    result = router.complete("prompt", tier=Tier.DEEP)

    assert result.provider == "gemini"
    assert result.tier is Tier.BULK
    assert result.degraded is True


def test_router_degrades_to_bulk_when_deep_raises():
    bulk = _StubProvider("gemini", "gemini-flash-latest", Tier.BULK, text="bulk fallback")
    deep = _StubProvider("claude-code", "claude-code", Tier.DEEP, fail=True)
    router = Router(bulk=bulk, deep=deep)

    result = router.complete("full prompt", tier=Tier.DEEP, deep_prompt="simpler prompt")

    assert result.degraded is True
    assert result.provider == "gemini"
    assert bulk.calls == ["simpler prompt"]  # simpler fallback prompt, not the full one


def test_router_degrades_to_bulk_when_no_deep_provider_configured():
    bulk = _StubProvider("gemini", "gemini-flash-latest", Tier.BULK, text="bulk only")
    router = Router(bulk=bulk, deep=None)

    result = router.complete("prompt", tier=Tier.DEEP)

    assert result.provider == "gemini"
    assert result.degraded is True


def test_router_never_upgrades_a_bulk_request_even_if_deep_is_healthy():
    bulk = _StubProvider("gemini", "gemini-flash-latest", Tier.BULK, text="bulk answer")
    deep = _StubProvider("claude-code", "claude-code", Tier.DEEP, text="deep answer")
    router = Router(bulk=bulk, deep=deep)

    result = router.complete("prompt", tier=Tier.BULK)

    assert result.provider == "gemini"
    assert deep.calls == []
