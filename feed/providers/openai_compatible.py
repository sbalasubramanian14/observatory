from __future__ import annotations
import os
import httpx
from feed.providers._dotenv import load_dotenv
from feed.providers._retry import call_with_retry
from feed.providers.base import (
    ProviderError,
    ProviderHealth,
    Tier,
    TransientProviderError,
    raise_for_status_code,
)
from feed.providers.reasoning import strip_reasoning


def _post(url: str, *, headers: dict, json_body: dict, timeout: float) -> dict:
    """The real-network seam. Tests must monkeypatch this, never let it run
    for real -- see tests/conftest.py's autouse guard. Mirrors
    feed.providers.gemini._post: raises on non-2xx (via
    resp.raise_for_status()) rather than returning an error body, so
    complete() below can classify the raised httpx exception.
    """
    resp = httpx.post(url, headers=headers, json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class OpenAICompatibleProvider:
    """One class for every provider that speaks the OpenAI chat-completions
    wire format (spec: "ONE OpenAICompatibleProvider class parameterised by
    (base_url, model, env_var) covers four of the five" live-tested
    providers -- Groq, Mistral, OpenRouter, and Cerebras). Gemini keeps its
    own bespoke implementation (feed.providers.gemini) since its request/
    response shape is not OpenAI-compatible.

    Model name, base URL, and env var all come from feed.toml
    (feed.config.BulkProviderConfig) via the caller -- never hardcoded here,
    per the task brief: stale model names have broken this project three
    times in one day.
    """

    tier = Tier.BULK

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        env_var: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        api_key: str | None = None,
    ):
        load_dotenv()
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.env_var = env_var
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.api_key = api_key if api_key is not None else os.environ.get(env_var)

    def complete(self, prompt: str, *, schema: type | None = None) -> str:
        if not self.api_key:
            raise ProviderError(f"{self.name}: {self.env_var} is not set")
        url = f"{self.base_url}/chat/completions"
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}

        def _once() -> str:
            try:
                data = _post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json_body=body,
                    timeout=self.timeout,
                )
            except httpx.HTTPStatusError as exc:
                raise_for_status_code(
                    self.name, exc.response.status_code, str(exc)
                )
                raise  # pragma: no cover -- raise_for_status_code always raises
            except httpx.TimeoutException as exc:
                raise TransientProviderError(f"{self.name}: timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise TransientProviderError(
                    f"{self.name}: connection error: {exc}"
                ) from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(
                    f"{self.name}: {type(exc).__name__}: {exc}"
                ) from exc
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(
                    f"{self.name}: unexpected response shape: {exc}"
                ) from exc

        text = call_with_retry(
            _once, max_retries=self.max_retries, backoff_base=self.backoff_base
        )
        # Spec requirement 3: strip reasoning blocks UNCONDITIONALLY, before
        # this text is ever parsed as JSON or stored -- any model behind
        # this class may emit them (verified live: qwen/qwen3.6-27b does).
        return strip_reasoning(text)

    def health(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(healthy=False, detail=f"{self.env_var} is not set")
        return ProviderHealth(healthy=True)
