import subprocess
import httpx
import pytest
from feed.providers.base import (
    PaymentRequiredError,
    ProviderError,
    ProviderHealth,
    RateLimitError,
    Tier,
    TransientProviderError,
)
from feed.providers.claude_code import ClaudeCodeProvider
from feed.providers.gemini import GeminiProvider
from feed.providers.router import RouteResult, Router


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(status_code, json={}, request=request)
    return httpx.HTTPStatusError(str(status_code), request=request, response=response)


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


def test_gemini_429_raises_rate_limit_error(monkeypatch):
    def fake_post(*a, **k):
        raise _http_error(429)

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="k")

    with pytest.raises(RateLimitError):
        provider.complete("hi")


def test_gemini_402_raises_payment_required_error(monkeypatch):
    def fake_post(*a, **k):
        raise _http_error(402)

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="k")

    with pytest.raises(PaymentRequiredError):
        provider.complete("hi")


def test_gemini_503_is_transient_and_retried_before_failing(monkeypatch):
    """Live-measured behaviour this task exists to fix: Gemini returned 503
    on 3 of 4 attempts. A single 503 must not immediately burn the story's
    attempt -- it gets a bounded retry first."""
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(503)
        return {"candidates": [{"content": {"parts": [{"text": "recovered"}]}}]}

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="k", max_retries=2)

    assert provider.complete("hi") == "recovered"
    assert calls["n"] == 2


def test_gemini_503_eventually_raises_transient_error_when_retries_exhausted(monkeypatch):
    def fake_post(*a, **k):
        raise _http_error(503)

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="k", max_retries=1)

    with pytest.raises(TransientProviderError):
        provider.complete("hi")


def test_gemini_strips_reasoning_before_returning(monkeypatch):
    def fake_post(*a, **k):
        return {"candidates": [{"content": {"parts": [
            {"text": "<think>internal chatter</think>the real answer"}
        ]}}]}

    monkeypatch.setattr("feed.providers.gemini._post", fake_post)
    provider = GeminiProvider(api_key="k")

    assert provider.complete("hi") == "the real answer"


# Bound at collection, before conftest's autouse guard replaces the module
# attribute -- test_claude_code_cli_encodes_stdin_as_utf8 tests the seam
# itself, so it needs the genuine function rather than the guard.
from feed.providers.claude_code import _run_cli as _REAL_RUN_CLI

# --- ClaudeCodeProvider ---------------------------------------------------

def test_claude_code_complete_returns_stripped_stdout(monkeypatch):
    def fake_run(args, *, timeout, input=None):
        assert args == ["claude", "-p"]      # prompt goes on stdin, not argv
        assert input == "why does this matter?"
        return subprocess.CompletedProcess(args, 0, stdout="  analysis text  \n", stderr="")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    assert provider.complete("why does this matter?") == "analysis text"


def test_claude_code_nonzero_exit_is_a_provider_error_not_a_crash(monkeypatch):
    def fake_run(args, *, timeout, input=None):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="rate limited")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_claude_code_timeout_is_a_provider_error_not_a_crash(monkeypatch):
    def fake_run(args, *, timeout, input=None):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider(timeout=1.0)

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_claude_code_empty_output_is_a_provider_error(monkeypatch):
    def fake_run(args, *, timeout, input=None):
        return subprocess.CompletedProcess(args, 0, stdout="   ", stderr="")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    with pytest.raises(ProviderError):
        provider.complete("hi")


def test_claude_code_strips_reasoning_before_returning(monkeypatch):
    def fake_run(args, *, timeout, input=None):
        return subprocess.CompletedProcess(
            args, 0, stdout="<think>mulling it over</think>the actual analysis", stderr="",
        )

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    provider = ClaudeCodeProvider()

    assert provider.complete("hi") == "the actual analysis"


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


def test_claude_code_sends_the_prompt_on_stdin_not_as_an_argument(monkeypatch):
    """Windows caps a whole command line at ~32,767 characters, so passing
    the prompt as argv raises `[WinError 206] The filename or extension is
    too long` once it grows. This is not hypothetical: the Top 50 ranking
    prompt carries ~100 headlines with summaries, and it failed in
    production the first run where enough stories had been summarized --
    silently degrading to the BULK provider, so the "ranked by Claude Code"
    feature quietly stopped being ranked by Claude Code.

    stdin has no such limit.
    """
    seen = {}

    def fake_run(args, *, timeout, input=None):
        seen["args"] = args
        seen["input"] = input
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)
    huge = "x" * 40_000

    assert ClaudeCodeProvider().complete(huge) == "ok"
    assert seen["input"] == huge
    assert huge not in " ".join(seen["args"])
    # A whole command line this size is what WinError 206 rejects.
    assert len(" ".join(seen["args"])) < 1000


def test_claude_code_cli_encodes_stdin_as_utf8(monkeypatch):
    """subprocess with `text=True` encodes stdin using the LOCALE codec,
    which on Windows is cp1252. AI summaries are full of typographic
    characters -- non-breaking hyphens, curly apostrophes, em dashes -- and
    cp1252 cannot represent them, so the stdin write raises
    UnicodeEncodeError, the CLI receives nothing, waits three seconds and
    exits 1.

    The router then degrades to BULK, so this surfaces as a feature quietly
    doing something else rather than as an error: the Top 50 kept getting
    written, just by Mistral instead of Claude Code. Pure-ASCII test
    prompts sail straight past it, which is exactly why it reached
    production.

    Asserts on the seam function itself -- this is the one place where the
    subprocess kwargs ARE the behaviour under test.
    """
    from feed.providers import claude_code

    seen = {}

    def fake_subprocess_run(args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    # No real process is spawned -- subprocess.run itself is replaced. The
    # module-level _REAL_RUN_CLI is needed because conftest's autouse guard
    # swaps the module attribute out, and the seam is what we are testing.
    monkeypatch.setattr(claude_code.subprocess, "run", fake_subprocess_run)

    _REAL_RUN_CLI(["claude", "-p"], timeout=10, input="non-breaking‑hyphen")

    assert seen.get("encoding") == "utf-8"


def test_claude_code_round_trips_typographic_characters(monkeypatch):
    """The end-to-end shape of the same bug: a prompt carrying the exact
    character that broke production must reach the CLI intact."""
    seen = {}

    def fake_run(args, *, timeout, input=None):
        seen["input"] = input
        return subprocess.CompletedProcess(args, 0, stdout="fine", stderr="")

    monkeypatch.setattr("feed.providers.claude_code._run_cli", fake_run)

    prompt = "Rank these:\n- GPT‑5.6 price‑performance — OpenAI’s move"
    assert ClaudeCodeProvider().complete(prompt) == "fine"
    assert seen["input"] == prompt
