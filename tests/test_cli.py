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
    items_xml = "".join(
        f"<item><title>Story {i}</title>"
        f"<link>https://example.com/story-{i}</link>"
        f"<description>Unique summary text describing story number {i} in detail.</description>"
        f"<pubDate>Tue, 18 Aug 2026 09:00:00 GMT</pubDate></item>"
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
