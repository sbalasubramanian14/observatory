from __future__ import annotations
import subprocess
from feed.providers.base import ProviderError, ProviderHealth, Tier


def _run_cli(args: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    """The real-process seam. Tests must monkeypatch this, never let it run
    for real -- see tests/conftest.py's autouse guard, which raises if this
    is ever reached without being replaced.
    """
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


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
        try:
            proc = _run_cli([self.binary, "-p", prompt], timeout=self.timeout)
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
        return text

    def health(self) -> ProviderHealth:
        # No cheap local probe for "is the CLI actually reachable and
        # authenticated" that doesn't itself cost a call -- health() is
        # deliberately optimistic (assume available) and lets complete()'s
        # failure be what the router degrades on. This matches spec 3.5's
        # "if the DEEP provider is rate-limited" framing: the failure is
        # discovered by trying, not predicted in advance.
        return ProviderHealth(healthy=True)
