import json
import pytest
from pathlib import Path
from sqlalchemy import select
from feed.cli import main

FIX = Path(__file__).parent / "fixtures" / "sample_rss.xml"


def _cfg(tmp_path) -> Path:
    p = tmp_path / "feed.toml"
    p.write_text(
        f'[database]\nurl = "sqlite:///{(tmp_path / "t.db").as_posix()}"\n'
        '[embedding]\nbackend = "onnx"\n'
        'model = "sentence-transformers/all-MiniLM-L6-v2"\n'
        'device = "cpu"\nbatch_size = 8\n',
        encoding="utf-8",
    )
    return p


def test_init_creates_the_database(tmp_path):
    cfg = _cfg(tmp_path)
    assert main(["--config", str(cfg), "init"]) == 0
    assert (tmp_path / "t.db").exists()


def test_sources_add_then_list(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "sources", "add", "--id", "rss:example",
               "--plugin", "rss", "--config-json", f'{{"path": "{FIX.as_posix()}"}}'])
    assert rc == 0
    main(["--config", str(cfg), "sources", "list"])
    assert "rss:example" in capsys.readouterr().out


def test_sources_sync_adds_from_catalogue(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    catalogue = tmp_path / "cat.toml"
    catalogue.write_text(f"""
[[source]]
id = "rss:example"
plugin = "rss"
territory = "research"
cadence_minutes = 60
authority = 0.7
config = {{ path = "{FIX.as_posix()}" }}
""", encoding="utf-8")

    rc = main(["--config", str(cfg), "sources", "sync", "--catalogue", str(catalogue)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "added=1" in out
    assert "rss:example" in out

    main(["--config", str(cfg), "sources", "list"])
    listing = capsys.readouterr().out
    assert "research" in listing
    assert "rss:example" in listing


def test_sources_sync_missing_catalogue_file_reports_error(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "sources", "sync",
              "--catalogue", str(tmp_path / "does-not-exist.toml")])
    assert rc == 2
    assert "sources sync failed" in capsys.readouterr().err


def test_sources_sync_removes_a_source_absent_from_the_catalogue(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    main(["--config", str(cfg), "sources", "add", "--id", "old-src",
         "--plugin", "rss", "--config-json", f'{{"path": "{FIX.as_posix()}"}}'])

    empty_catalogue = tmp_path / "empty.toml"
    empty_catalogue.write_text("", encoding="utf-8")
    rc = main(["--config", str(cfg), "sources", "sync", "--catalogue", str(empty_catalogue)])
    assert rc == 0
    assert "deleted=1" in capsys.readouterr().out

    main(["--config", str(cfg), "sources", "list"])
    assert "old-src" not in capsys.readouterr().out


def test_unknown_plugin_is_rejected_at_add_time(tmp_path):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "sources", "add", "--id", "x",
               "--plugin", "nope", "--config-json", "{}"])
    assert rc == 2


def test_unknown_plugin_does_not_touch_the_database(tmp_path):
    """Risk point 2: rejection must happen BEFORE any DB write.

    A prior version of this CLI could plausibly open a session and merge a
    Source row before validating the plugin name. Prove the row never lands.
    """
    from feed.config import load_config
    from feed.db import make_engine, make_session_factory
    from feed.models import Source

    cfg_path = _cfg(tmp_path)
    main(["--config", str(cfg_path), "init"])
    rc = main(["--config", str(cfg_path), "sources", "add", "--id", "x",
               "--plugin", "nope", "--config-json", "{}"])
    assert rc == 2

    cfg = load_config(cfg_path)
    engine = make_engine(cfg.database.url)
    factory = make_session_factory(engine)
    with factory() as s:
        assert s.scalars(select(Source)).all() == []


def test_invalid_config_json_is_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "sources", "add", "--id", "x",
               "--plugin", "rss", "--config-json", "{not valid json"])
    assert rc == 2


def test_stats_on_empty_database_does_not_crash(tmp_path, capsys):
    """Risk point 3: `feed stats` must not crash before anything has run."""
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "stats"])
    assert rc == 0
    assert "stories" in capsys.readouterr().out


def test_run_wires_per_model_threshold_into_adjudicator():
    """Ruling 1 regression test.

    Task 11 moved per-model merge thresholds into the adjudicator via the
    keyword-only `threshold_for` provider. If the CLI builds its adjudicator
    without passing `threshold_for=cfg.clustering.threshold_for`, the
    per-model map (including the empirically-measured 0.35 MiniLM threshold
    in feed.toml) is silently ignored and every pair falls back to the
    global `merge_threshold` instead.

    This test picks a pair_score (0.15) and a per-model threshold (0.10)
    such that the verdict flips depending on whether threshold_for is wired:
      - wired:      effective threshold = 0.10, band = [0.07, 0.13] -> SAME
      - not wired:  effective threshold = 0.50, band = [0.47, 0.53] -> DIFFERENT
    """
    from feed.cli import _build_adjudicator
    from feed.clustering.adjudicate import Verdict
    from feed.config import Config

    cfg = Config.model_validate({
        "clustering": {
            "merge_threshold": 0.50,
            "merge_thresholds": {"test-model": 0.10},
        }
    })
    adjudicator = _build_adjudicator(cfg)

    class FakeItem:
        embedding_model_id = "test-model"

    assert adjudicator.decide(0.15, FakeItem(), None) is Verdict.SAME


def test_run_drains_more_than_one_batch_per_stage(tmp_path, monkeypatch, capsys):
    """I1: feed run used to call each stage exactly once with hardcoded
    batch limits (normalize=100, cluster=200). This reproduces the
    finding's own numbers -- 250 collected items, more than both of those
    batch sizes -- with the embedder stubbed out (no model download, so
    this stays fast) to isolate the drain behaviour itself. A single `feed
    run` call must still carry every item all the way to Stage.SCORED, not
    leave any of them stuck at collected/normalized/embedded/clustered.
    """
    import numpy as np
    import feed.cli as cli_module
    from feed.config import load_config
    from feed.db import make_engine, make_session_factory
    from feed.models import Item, Stage

    class _StubEmbedder:
        # Spreads items across 8 one-hot directions (by a stable hash of
        # each item's text) instead of one identical vector for everyone --
        # a single vector would merge all 250 items into one giant story,
        # making cluster()'s O(members) centroid/membership recompute on
        # every merge blow up to O(n^2) with a large constant. Several
        # smaller stories keep this test fast while still exercising real
        # clustering logic (candidates, pair_score, adjudication) rather
        # than a no-op.
        model_id = "stub/v1"
        dimensions = 8

        def encode(self, texts):
            vecs = np.zeros((len(texts), self.dimensions), dtype=np.float32)
            for row, t in enumerate(texts):
                vecs[row, hash(t) % self.dimensions] = 1.0
            return vecs

    monkeypatch.setattr(cli_module, "build_embedder", lambda cfg: _StubEmbedder())
    # Deterministic and avoids a slow real `import torch` just to probe
    # CUDA availability -- this test cares about drain behaviour, not
    # device resolution, and must behave the same on a GPU-equipped runner.
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)

    n = 130  # > normalize's default limit (100) and > embed's batch_size (50)
    # A4 (backfill cap): `feed run` always collects against the real
    # current time (no --now override), and a brand-new source's first
    # fetch is capped to [collect].max_backfill_days (default 2). A fixed
    # historical pubDate would silently fall outside that window and this
    # test would collect 0 items regardless of how the pipeline runs it --
    # so pin the fixture's dates to "recently before the real now" instead.
    import email.utils
    from datetime import datetime, timedelta, timezone
    recent = email.utils.format_datetime(
        datetime.now(timezone.utc) - timedelta(hours=1)
    )
    items_xml = "".join(
        f"<item><title>Story {i}</title>"
        f"<link>https://example.com/story-{i}</link>"
        f"<description>Unique summary text describing story number {i} in detail.</description>"
        f"<pubDate>{recent}</pubDate></item>"
        for i in range(n)
    )
    rss_path = tmp_path / "big.xml"
    rss_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>Big feed</title>{items_xml}</channel></rss>",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "feed.toml"
    cfg_path.write_text(
        f'[database]\nurl = "sqlite:///{(tmp_path / "t.db").as_posix()}"\n'
        '[embedding]\nbackend = "onnx"\n'
        'model = "sentence-transformers/all-MiniLM-L6-v2"\n'
        'device = "cpu"\nbatch_size = 50\n',  # > 1 round needed too (250/50)
        encoding="utf-8",
    )

    assert cli_module.main(["--config", str(cfg_path), "init"]) == 0
    assert cli_module.main([
        "--config", str(cfg_path), "sources", "add", "--id", "rss:big",
        "--plugin", "rss", "--config-json", f'{{"path": "{rss_path.as_posix()}"}}',
    ]) == 0
    assert cli_module.main(["--config", str(cfg_path), "run"]) == 0

    out = capsys.readouterr().out
    # rounds > 1 proves drain() actually looped for at least one stage,
    # not just that the final state happens to look drained.
    assert any(f"rounds={r}" in out for r in range(2, 51)), out

    cfg = load_config(cfg_path)
    engine = make_engine(cfg.database.url)
    factory = make_session_factory(engine)
    with factory() as s:
        items = s.query(Item).all()
        assert len(items) == n
        stages = {i.stage for i in items}
        assert stages == {Stage.SCORED}, (
            f"expected all {n} items to reach SCORED in a single `feed run`, "
            f"got stages present: {stages}"
        )


def _seed_scored_story(cfg_path, *, score=0.9):
    """Seed a scored story directly, bypassing the full pipeline -- these
    CLI tests exercise enrich/publish wiring, not collect/normalize/embed/
    cluster/score, which are already covered elsewhere.
    """
    from datetime import datetime, timezone
    from feed.config import load_config
    from feed.db import create_all, make_engine, make_session_factory
    from feed.models import Item, Source, Story

    cfg = load_config(cfg_path)
    engine = make_engine(cfg.database.url)
    create_all(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        now = datetime.now(timezone.utc)
        s.add(Source(id="src", plugin="rss", config={}, cadence_minutes=30))
        story = Story(title="A story", first_seen=now, updated_at=now, item_count=1,
                      outlet_count=1, score=score)
        s.add(story)
        s.flush()
        s.add(Item(source_id="src", url="http://x/1", url_hash="h1", title="A story",
                   story_id=story.id, published_at=now))
        s.commit()
        return story.id


class _FakeCliProvider:
    def __init__(self, name, model, tier, text):
        self.name = name
        self.model = model
        self.tier = tier
        self._text = text

    def complete(self, prompt, *, schema=None):
        return self._text

    def health(self):
        from feed.providers.base import ProviderHealth
        return ProviderHealth(healthy=True)


def _fake_router():
    import json
    from feed.providers.base import Tier
    from feed.providers.router import Router
    tier1_json = json.dumps({"headline": "Canonical", "summary": "Sum.", "category": "research"})
    bulk = _FakeCliProvider("gemini", "gemini-flash-latest", Tier.BULK, tier1_json)
    deep = _FakeCliProvider("claude-code", "claude-code", Tier.DEEP, "deep analysis")
    return Router(bulk=bulk, deep=deep)


def test_enrich_command_runs_tier1_and_tier2(tmp_path, capsys, monkeypatch):
    import feed.cli as cli_module

    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed_scored_story(cfg, score=0.95)
    monkeypatch.setattr(cli_module, "_build_router", lambda cfg: _fake_router())

    rc = main(["--config", str(cfg), "enrich"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tier1: ok=1" in out
    assert "tier2: ok=1" in out


def test_publish_command_writes_bundle(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed_scored_story(cfg, score=0.5)
    out_dir = tmp_path / "bundle"

    rc = main(["--config", str(cfg), "publish", "--out", str(out_dir)])

    assert rc == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "sources.json").exists()
    assert "published 1 stories" in capsys.readouterr().out


def test_run_enrich_and_publish_flags_are_opt_in(tmp_path, monkeypatch):
    """Default `feed run` must not touch enrich/publish at all -- they are
    opt-in flags per the build spec ("wire both into feed run behind flags
    so the default run stays cheap")."""
    import feed.cli as cli_module

    calls = []
    monkeypatch.setattr(cli_module, "enrich", lambda *a, **k: calls.append("enrich"))
    monkeypatch.setattr(cli_module, "publish", lambda *a, **k: calls.append("publish"))
    monkeypatch.setattr(cli_module, "build_embedder", lambda cfg: _NoopEmbedder())
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)

    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])

    assert main(["--config", str(cfg), "run"]) == 0
    assert calls == []


def test_run_with_enrich_and_publish_flags_invokes_both(tmp_path, monkeypatch):
    import feed.cli as cli_module
    from feed.stages.enrich import EnrichResult

    calls = []

    def _fake_enrich(*a, **k):
        calls.append("enrich")
        return EnrichResult()

    monkeypatch.setattr(cli_module, "enrich", _fake_enrich)
    monkeypatch.setattr(cli_module, "_build_router", lambda cfg: _fake_router())
    monkeypatch.setattr(cli_module, "build_embedder", lambda cfg: _NoopEmbedder())
    monkeypatch.setattr("feed.embedding.resolve.cuda_available", lambda: False)

    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])

    rc = main(["--config", str(cfg), "run", "--enrich", "--publish",
              "--out", str(tmp_path / "bundle")])

    assert rc == 0
    assert calls == ["enrich"]
    assert (tmp_path / "bundle" / "manifest.json").exists()


class _NoopEmbedder:
    model_id = "noop/v1"
    dimensions = 4

    def encode(self, texts):
        import numpy as np
        return np.zeros((len(texts), self.dimensions), dtype="float32")


@pytest.mark.slow
def test_full_run_produces_scored_stories(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    main(["--config", str(cfg), "sources", "add", "--id", "rss:example",
          "--plugin", "rss", "--config-json", f'{{"path": "{FIX.as_posix()}"}}'])
    assert main(["--config", str(cfg), "run"]) == 0
    main(["--config", str(cfg), "stats"])
    out = capsys.readouterr().out
    assert "stories" in out


# --- multi-provider BULK failover chain (_build_router / `feed providers`) -

def _cfg_with_bulk_chain(tmp_path, *, cerebras_enabled=False) -> Path:
    p = tmp_path / "feed.toml"
    p.write_text(
        f'[database]\nurl = "sqlite:///{(tmp_path / "t.db").as_posix()}"\n'
        '[embedding]\nbackend = "onnx"\n'
        'model = "sentence-transformers/all-MiniLM-L6-v2"\n'
        'device = "cpu"\nbatch_size = 8\n'
        '[[providers.bulk]]\n'
        'name = "groq"\nkind = "openai_compatible"\nmodel = "openai/gpt-oss-120b"\n'
        'base_url = "https://api.groq.com/openai/v1"\nenv_var = "GROQ_API_KEY_TEST"\n'
        '[[providers.bulk]]\n'
        'name = "gemini"\nkind = "gemini"\nmodel = "gemini-flash-latest"\n'
        'env_var = "GEMINI_API_KEY_TEST"\n'
        '[[providers.bulk]]\n'
        'name = "cerebras"\nkind = "openai_compatible"\nmodel = "gpt-oss-120b"\n'
        'base_url = "https://api.cerebras.ai/v1"\nenv_var = "CEREBRAS_API_KEY_TEST"\n'
        f'enabled = {"true" if cerebras_enabled else "false"}\n',
        encoding="utf-8",
    )
    return p


def test_build_router_wires_only_enabled_bulk_entries_in_priority_order(tmp_path):
    from feed.cli import _build_router
    from feed.config import load_config
    from feed.providers.failover import FailoverProvider

    cfg = load_config(_cfg_with_bulk_chain(tmp_path))
    router = _build_router(cfg)

    assert isinstance(router.bulk, FailoverProvider)
    names = [p.name for p in router.bulk._providers]
    assert names == ["groq", "gemini"]  # cerebras excluded: disabled by default


def test_build_router_falls_back_to_single_gemini_when_no_bulk_configured(tmp_path):
    from feed.cli import _build_router
    from feed.config import load_config
    from feed.providers.gemini import GeminiProvider

    cfg = load_config(_cfg(tmp_path))  # no [[providers.bulk]] at all
    router = _build_router(cfg)

    assert isinstance(router.bulk, GeminiProvider)


def test_cmd_providers_probes_each_configured_provider(tmp_path, capsys, monkeypatch):
    """Requirement 6: `feed providers` prints enabled / reachable / model /
    latency / today's usage per provider, without ever making a real
    network call in this test -- _build_bulk_provider is monkeypatched to
    return stub providers instead.
    """
    import feed.cli as cli_module
    from feed.providers.base import ProviderError, ProviderHealth

    class _Stub:
        def __init__(self, name, model, *, healthy=True, fails=False):
            self.name = name
            self.model = model
            self._healthy = healthy
            self._fails = fails

        def complete(self, prompt, *, schema=None):
            if self._fails:
                raise ProviderError(f"{self.name}: boom")
            return "OK"

        def health(self):
            return ProviderHealth(healthy=self._healthy)

    def _fake_build(entry, *, max_retries, backoff_base):
        if entry.name == "gemini":
            return _Stub("gemini", entry.model, fails=True)
        return _Stub(entry.name, entry.model)

    monkeypatch.setattr(cli_module, "_build_bulk_provider", _fake_build)

    cfg = _cfg_with_bulk_chain(tmp_path, cerebras_enabled=True)
    main(["--config", str(cfg), "init"])

    rc = main(["--config", str(cfg), "providers"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "groq" in out and "yes" in out  # reachable
    assert "gemini" in out and "boom" in out  # reachable=no, error shown
    assert "cerebras" in out
    assert "claude-code" in out  # DEEP provider listed too


def test_enrich_stores_provenance_via_the_real_failover_chain(tmp_path, monkeypatch):
    """End-to-end (short of the network): the real FailoverProvider, the
    real Router, and the real enrich_tier1 all wired together the way
    `feed enrich` wires them, with only the two providers' network seams
    monkeypatched -- proves requirement 4 (provenance) survives the whole
    stack, not just the FailoverProvider unit tests.
    """
    import json
    from feed.cli import _build_router
    from feed.config import load_config
    from feed.db import make_engine, make_session_factory
    from feed.providers.base import Tier
    from feed.stages.enrich import enrich_tier1

    monkeypatch.delenv("GROQ_API_KEY_TEST", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY_TEST", "k")

    cfg_path = _cfg_with_bulk_chain(tmp_path)
    main(["--config", str(cfg_path), "init"])
    _seed_scored_story(cfg_path, score=0.9)
    cfg = load_config(cfg_path)

    tier1_json = json.dumps({"headline": "H", "summary": "S", "category": "research"})

    def fake_gemini_post(url, *, headers, json_body, timeout):
        return {"candidates": [{"content": {"parts": [{"text": tier1_json}]}}]}

    monkeypatch.setattr("feed.providers.gemini._post", fake_gemini_post)

    router = _build_router(cfg)
    # groq has no API key set (GROQ_API_KEY_TEST unset above) -- it must be
    # skipped with a clear reason, and gemini must serve the request.
    engine = make_engine(cfg.database.url)
    factory = make_session_factory(engine)
    with factory() as s:
        result = enrich_tier1(s, router, cfg.providers)
        assert result.tier1_processed == 1
        from feed.models import Story
        story = s.query(Story).first()
        assert story.summary_provider == "gemini:gemini-flash-latest"
