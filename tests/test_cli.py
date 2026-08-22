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
