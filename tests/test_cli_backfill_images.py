"""`feed backfill-images`: the one-off (but resumable) sweep of the
existing corpus for items that never got a chance at the og:image
fallback (Phase D-images). Exercises the real CLI entry point end to end,
faking only the network seam (feed.imaging._get).
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

from feed import imaging
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


def _og_html(url: str) -> str:
    return f'<html><head><meta property="og:image" content="{url}"/></head></html>'


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "", url: str = ""):
        self.status_code = status_code
        self.text = text
        self.url = url


def _seed(cfg_path, items):
    """items: list of dicts with keys url_hash, url, source_id, plus
    optional image_url/image_checked_at/stage."""
    from feed.config import load_config
    from feed.db import create_all, make_engine, make_session_factory
    from feed.models import Item, Source, Stage

    cfg = load_config(cfg_path)
    engine = make_engine(cfg.database.url)
    create_all(engine)
    factory = make_session_factory(engine)
    with factory() as s:
        for sid in {it.get("source_id", "src") for it in items}:
            s.add(Source(id=sid, plugin="rss", config={}, cadence_minutes=30))
        s.flush()
        for it in items:
            s.add(Item(
                source_id=it.get("source_id", "src"),
                url=it["url"], url_hash=it["url_hash"], title="T",
                image_url=it.get("image_url"),
                image_checked_at=it.get("image_checked_at"),
                stage=it.get("stage", Stage.SCORED),
            ))
        s.commit()


def test_backfill_images_reports_zero_on_an_empty_database(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    rc = main(["--config", str(cfg), "backfill-images"])
    assert rc == 0
    assert "0 item(s) eligible" in capsys.readouterr().out


def test_backfill_images_gains_images_and_reports_per_source(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed(cfg, [
        dict(source_id="techcrunch", url_hash="a", url="https://example.com/a"),
        dict(source_id="techcrunch", url_hash="b", url="https://example.com/b"),
        dict(source_id="theverge", url_hash="c", url="https://example.com/blocked"),
    ])

    def _fake_get(url, *, timeout):
        if "blocked" in url:
            return _FakeResponse(403, "", url)
        return _FakeResponse(200, _og_html("https://img.example.com/found.jpg"), url)

    monkeypatch.setattr(imaging, "_get", _fake_get)
    rc = main(["--config", str(cfg), "backfill-images", "--host-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 item(s) eligible" in out
    assert "attempted=3 gained=2" in out
    assert "techcrunch" in out and "gained=2" in out
    assert "theverge" in out and "blocked_403=1" in out


def test_backfill_images_skips_items_that_already_have_an_image(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed(cfg, [
        dict(url_hash="has-image", url="https://example.com/has-image",
             image_url="https://img.example.com/already.jpg"),
    ])
    calls = []
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: calls.append(url) or _FakeResponse(200, "", url))
    rc = main(["--config", str(cfg), "backfill-images"])
    assert rc == 0
    assert calls == []
    assert "0 item(s) eligible" in capsys.readouterr().out


def test_backfill_images_skips_already_checked_items(tmp_path, capsys, monkeypatch):
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed(cfg, [
        dict(url_hash="checked", url="https://example.com/checked",
             image_checked_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ])
    calls = []
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: calls.append(url) or _FakeResponse(200, "", url))
    main(["--config", str(cfg), "backfill-images"])
    assert calls == []


def test_backfill_images_excludes_failed_items(tmp_path, capsys, monkeypatch):
    from feed.models import Stage
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed(cfg, [
        dict(url_hash="broken", url="https://example.com/broken", stage=Stage.FAILED),
    ])
    calls = []
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: calls.append(url) or _FakeResponse(200, "", url))
    main(["--config", str(cfg), "backfill-images"])
    assert calls == []


def test_backfill_images_limit_and_resumability(tmp_path, capsys, monkeypatch):
    """--limit N attempts only N items; a second, unlimited run picks up
    exactly the remainder -- proving the whole command is resumable
    without any extra bookkeeping beyond image_checked_at.
    """
    cfg = _cfg(tmp_path)
    main(["--config", str(cfg), "init"])
    _seed(cfg, [
        dict(url_hash=f"r{i}", url=f"https://example.com/r{i}") for i in range(5)
    ])
    monkeypatch.setattr(imaging, "_get",
                        lambda url, *, timeout: _FakeResponse(
                            200, _og_html("https://img.example.com/x.jpg"), url))

    rc = main(["--config", str(cfg), "backfill-images", "--limit", "2", "--host-delay", "0"])
    assert rc == 0
    out1 = capsys.readouterr().out
    assert "2 item(s) eligible" in out1
    assert "attempted=2 gained=2" in out1

    rc = main(["--config", str(cfg), "backfill-images", "--host-delay", "0"])
    assert rc == 0
    out2 = capsys.readouterr().out
    assert "3 item(s) eligible" in out2
    assert "attempted=3 gained=3" in out2
