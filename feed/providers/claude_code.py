from __future__ import annotations
import subprocess
from feed.providers.base import ProviderError, ProviderHealth, Tier
from feed.providers.reasoning import strip_reasoning


def _run_cli(args: list[str], *, timeout: float,
             input: str | None = None) -> subprocess.CompletedProcess:
    """The real-process seam. Tests must monkeypatch this, never let it run
    for real -- see tests/conftest.py's autouse guard, which raises if this
    is ever reached without being replaced.
    """
    # encoding="utf-8" is load-bearing, not tidiness. With text=True alone,
    # subprocess encodes stdin and decodes stdout using the LOCALE codec --
    # cp1252 on this machine. AI summaries are full of characters cp1252
    # cannot represent (non-breaking hyphens, curly apostrophes, em
    # dashes), so the stdin write raised UnicodeEncodeError, the CLI got
    # nothing, waited three seconds and exited 1. The router then degraded
    # to BULK, which is why this showed up as the Top 50 being ranked by
    # Mistral rather than as any error. errors="replace" on the way back:
    # a mangled character in a summary is worth far less than a lost call.
    return subprocess.run(args, capture_output=True, timeout=timeout, input=input,
                          encoding="utf-8", errors="replace")


class ClaudeCodeProvider:
    """Tier 2 (DEEP) provider: shells out to the locally installed `claude`
    CLI, subscription-billed rather than metered per token -- hence the
    router budgets it by call count (spec 3.5), not tokens.
    """
    name = "claude-code"
    tier = Tier.DEEP

    def __init__(self, *, model: str = "claude-code", timeout: float = 120.0,
                 binary: str = "claude"):
        self.model = model
        self.timeout = timeout
        self.binary = binary

    def complete(self, prompt: str, *, schema: type | None = None) -> str:
        # The prompt goes on STDIN, never in argv. Windows caps an entire
        # command line at ~32,767 characters, and argv passing failed in
        # production with `[WinError 206] The filename or extension is too
        # long` the first time the Top 50 ranking prompt (about 100
        # headlines with summaries) grew past it. The router dutifully
        # degraded to the BULK provider, so the failure surfaced not as an
        # error but as a feature quietly no longer doing what it claimed --
        # "ranked by Claude Code" ranked by Mistral instead. `claude -p`
        # with no prompt argument reads the prompt from stdin, which has no
        # length limit.
        try:
            proc = _run_cli([self.binary, "-p"], timeout=self.timeout, input=prompt)
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"claude-code: timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise ProviderError(f"claude-code: {type(exc).__name__}: {exc}") from exc

        if proc.returncode != 0:
            raise ProviderError(
                f"claude-code: exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
        text = (proc.stdout or "").strip()
        if not text:
            raise ProviderError("claude-code: empty output")
        # Spec requirement 3: strip reasoning blocks unconditionally --
        # "any model may do this", not just the OpenAI-compatible chain.
        text = strip_reasoning(text)
        if not text:
            raise ProviderError("claude-code: output was only reasoning content")
        return text

    def health(self) -> ProviderHealth:
        # No cheap local probe for "is the CLI actually reachable and
        # authenticated" that doesn't itself cost a call -- health() is
        # deliberately optimistic (assume available) and lets complete()'s
        # failure be what the router degrades on. This matches spec 3.5's
        # "if the DEEP provider is rate-limited" framing: the failure is
        # discovered by trying, not predicted in advance.
        return ProviderHealth(healthy=True)
