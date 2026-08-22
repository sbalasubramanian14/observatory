from __future__ import annotations
import os
from pathlib import Path

# A tiny, dependency-free ".env" reader. python-dotenv is not in
# pyproject.toml's dependencies and one file's worth of KEY=VALUE parsing
# does not justify adding it. Deliberately minimal: no interpolation, no
# multiline values, no export keyword -- just what GEMINI_API_KEY=... needs.


def load_dotenv(path: Path | str = ".env") -> None:
    """Populate os.environ from a simple KEY=VALUE .env file.

    Never overrides a variable already set in the real environment (matches
    the conventional python-dotenv default), so an operator's explicit
    `GEMINI_API_KEY=... feed run` still wins over whatever is in .env.
    Silently does nothing if the file does not exist -- .env is optional,
    e.g. in CI where the key comes from a real secret instead.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
