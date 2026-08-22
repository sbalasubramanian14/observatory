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

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _post(url: str, *, headers: dict, json_body: dict, timeout: float) -> dict:
    """The real-network seam. Tests must monkeypatch this, never let it run
    for real -- see tests/conftest.py's autouse guard, which raises if this
    is ever reached without being replaced (mirrors the existing pattern
    for feed.stages.normalize._fetch_remote_text).
    """
    resp = httpx.post(url, headers=headers, json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class GeminiProvider:
    """Tier 1 (BULK) provider: Gemini's API, called once per story.

    Model is `gemini-flash-latest` by default rather than a pinned version
    like `gemini-2.0-flash` (confirmed dead) -- the `-latest` alias avoids
    that staleness recurring, at the cost of the model occasionally
    changing under us.
    """
    name = "gemini"
    tier = Tier.BULK

    def __init__(self, *, model: str = "gemini-flash-latest",
                 api_key: str | None = None, timeout: float = 30.0,
                 max_retries: int = 2, backoff_base: float = 0.5):
        load_dotenv()
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def complete(self, prompt: str, *, schema: type | None = None) -> str:
        if not self.api_key:
            raise ProviderError("gemini: GEMINI_API_KEY is not set")
        url = ENDPOINT.format(model=self.model)
        body = {"contents": [{"parts": [{"text": prompt}]}]}

        def _once() -> str:
            try:
                data = _post(url, headers={"x-goog-api-key": self.api_key},
                            json_body=body, timeout=self.timeout)
            except httpx.HTTPStatusError as exc:
                raise_for_status_code("gemini", exc.response.status_code, str(exc))
                raise  # pragma: no cover -- raise_for_status_code always raises
            except httpx.TimeoutException as exc:
                raise TransientProviderError(f"gemini: timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise TransientProviderError(f"gemini: connection error: {exc}") from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"gemini: {type(exc).__name__}: {exc}") from exc
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(f"gemini: unexpected response shape: {exc}") from exc

        text = call_with_retry(_once, max_retries=self.max_retries,
                               backoff_base=self.backoff_base)
        # Spec requirement 3: strip reasoning blocks unconditionally --
        # "any model may do this", including whatever Gemini serves behind
        # -latest. See feed.providers.reasoning.strip_reasoning.
        return strip_reasoning(text)

    def health(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(healthy=False, detail="GEMINI_API_KEY is not set")
        return ProviderHealth(healthy=True)
