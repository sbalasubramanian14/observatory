import numpy as np
from feed.embedding.base import pack as real_pack, unpack
from feed.models import Item, Source, Stage
from feed.stages.embed import embed


class FakeEmbedder:
    model_id = "fake/model-v1"
    dimensions = 4

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.arange(4, dtype=np.float32), (len(texts), 1))


class ExplodingEmbedder(FakeEmbedder):
    def encode(self, texts):
        raise RuntimeError("gpu on fire")


class OneBadItemEmbedder(FakeEmbedder):
    """Batch encode() fails; per-item fallback fails only for the item
    whose text contains the configured trigger substring."""

    def __init__(self, trigger):
        super().__init__()
        self.trigger = trigger
        self.batch_calls = 0
        self.single_calls = 0

    def encode(self, texts):
        if len(texts) > 1:
            self.batch_calls += 1
            raise RuntimeError("batch backend hiccup")
        self.single_calls += 1
        if self.trigger in texts[0]:
            raise RuntimeError("bad row: unencodable text")
        return np.tile(np.arange(4, dtype=np.float32), (len(texts), 1))


def _seed(session, n=3):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    for i in range(n):
        session.add(Item(source_id="s", url=f"http://x/{i}", url_hash=f"h{i}",
                         title=f"T{i}", text=f"body {i}", stage=Stage.NORMALIZED))
    session.commit()


def test_embeds_and_advances(session):
    _seed(session)
    emb = FakeEmbedder()
    res = embed(session, emb)
    assert res.processed == 3
    items = session.query(Item).all()
    assert all(i.stage is Stage.EMBEDDED for i in items)
    assert all(i.embedding_model_id == "fake/model-v1" for i in items)
    assert np.array_equal(unpack(items[0].embedding), np.arange(4, dtype=np.float32))


def test_encodes_in_one_batched_call(session):
    _seed(session, n=5)
    emb = FakeEmbedder()
    embed(session, emb, limit=5)
    assert len(emb.calls) == 1 and len(emb.calls[0]) == 5


def test_embeds_title_plus_text(session):
    _seed(session, n=1)
    emb = FakeEmbedder()
    embed(session, emb)
    assert emb.calls[0][0].startswith("T0")
    assert "body 0" in emb.calls[0][0]


def test_only_claims_normalized_items(session):
    _seed(session, n=3)
    collected = session.query(Item).filter_by(url_hash="h0").one()
    collected.stage = Stage.COLLECTED
    session.commit()
    emb = FakeEmbedder()
    res = embed(session, emb)
    assert res.processed == 2
    assert session.query(Item).filter_by(url_hash="h0").one().stage is Stage.COLLECTED


def test_claim_order_is_deterministic_and_limited(session):
    _seed(session, n=10)
    emb = FakeEmbedder()
    res = embed(session, emb, limit=4)
    assert res.processed == 4
    embedded = {i.url_hash for i in session.query(Item).all() if i.stage is Stage.EMBEDDED}
    # deterministic id-order claim => the first 4 seeded items (h0..h3)
    assert embedded == {"h0", "h1", "h2", "h3"}


def test_pack_unpack_roundtrips_through_the_database(session):
    _seed(session, n=1)
    emb = FakeEmbedder()
    embed(session, emb)
    session.expire_all()  # force a real reload from the DB, not the ORM identity cache
    item = session.query(Item).one()
    assert np.array_equal(unpack(item.embedding), np.arange(4, dtype=np.float32))


def test_backend_failure_marks_the_batch_failed_not_the_process(session):
    _seed(session)
    res = embed(session, ExplodingEmbedder())
    assert res.failed == 3 and res.processed == 0
    assert all(i.stage is Stage.FAILED for i in session.query(Item).all())
    assert "gpu on fire" in session.query(Item).first().error


def test_batch_failure_falls_back_to_per_item_and_isolates_the_bad_row(session):
    _seed(session, n=3)
    emb = OneBadItemEmbedder(trigger="body 1")
    res = embed(session, emb)

    assert res.processed == 2
    assert res.failed == 1
    by_hash = {i.url_hash: i for i in session.query(Item).all()}
    assert by_hash["h0"].stage is Stage.EMBEDDED
    assert by_hash["h2"].stage is Stage.EMBEDDED
    assert by_hash["h1"].stage is Stage.FAILED
    assert "bad row" in by_hash["h1"].error
    assert by_hash["h0"].embedding_model_id == "fake/model-v1"
    assert np.array_equal(unpack(by_hash["h0"].embedding), np.arange(4, dtype=np.float32))
    # the fast path was tried exactly once, then exactly one single-item
    # call per item in the batch (no doubling, no infinite loop)
    assert emb.batch_calls == 1
    assert emb.single_calls == 3


def test_batch_failure_with_every_item_bad_marks_all_failed_without_looping(session):
    _seed(session, n=3)
    emb = OneBadItemEmbedder(trigger="body")  # matches every item's text
    res = embed(session, emb)

    assert res.processed == 0
    assert res.failed == 3
    items = session.query(Item).all()
    assert all(i.stage is Stage.FAILED for i in items)
    assert all("bad row" in i.error for i in items)
    assert emb.batch_calls == 1
    assert emb.single_calls == 3


def test_pack_failure_in_fallback_isolates_the_row_and_others_still_embed(session, monkeypatch):
    """Reviewer finding: _embed_one's try only wrapped encode(); pack()/commit()
    failures escaped _embed_one, the batch loop, and embed() itself. This
    proves the whole per-item operation is now covered: encode succeeds for
    every item (trigger never matches), but pack() is made to blow up on the
    second item only."""
    _seed(session, n=3)
    emb = OneBadItemEmbedder(trigger="__never_matches__")

    calls = {"n": 0}

    def flaky_pack(vec):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("corrupt vector shape")
        return real_pack(vec)

    monkeypatch.setattr("feed.stages.embed.pack", flaky_pack)

    res = embed(session, emb)

    assert res.processed == 2
    assert res.failed == 1
    items = session.query(Item).order_by(Item.id).all()
    assert items[0].stage is Stage.EMBEDDED
    assert items[1].stage is Stage.FAILED
    assert "corrupt vector shape" in items[1].error
    assert items[2].stage is Stage.EMBEDDED


def test_pack_failure_on_the_batch_success_path_does_not_escape_embed(session, monkeypatch):
    """I2: encode() succeeds (the fast/normal path, not the fallback), but
    pack() then raises while assigning the batch's vectors. Before the fix
    that loop-and-commit block sat outside any try in embed(), so this
    exception would propagate straight out of embed() itself -- aborting
    cmd_run before cluster or score ever ran, for every item in the batch,
    not just the malformed one. This proves embed() now catches it, rolls
    back, and recovers via the same per-item isolation the encode()-failure
    path already gets -- so a single bad row no longer takes down the run.
    """
    _seed(session, n=3)
    emb = FakeEmbedder()  # encode() succeeds normally; batch path is taken

    calls = {"n": 0}
    def flaky_pack(vec):
        calls["n"] += 1
        if calls["n"] == 1:  # fails during the FIRST pass (the batch loop)
            raise ValueError("corrupt vector shape")
        return real_pack(vec)
    monkeypatch.setattr("feed.stages.embed.pack", flaky_pack)

    res = embed(session, emb)  # must not raise

    assert res.processed == 3
    assert res.failed == 0
    items = session.query(Item).order_by(Item.id).all()
    assert all(i.stage is Stage.EMBEDDED for i in items)


def test_commit_failure_on_the_batch_success_path_does_not_escape_embed(session, monkeypatch):
    """Same finding, but for the batch's own session.commit() call. Before
    the fix, a RuntimeError there propagated straight out of embed(),
    leaving every item in the batch stuck at NORMALIZED with no StageResult
    returned and the rest of cmd_run's pipeline never reached. It must now
    be caught and recovered via the per-item fallback instead.
    """
    _seed(session, n=3)
    emb = FakeEmbedder()

    calls = {"n": 0}
    real_commit = session.commit
    def flaky_commit():
        calls["n"] += 1
        if calls["n"] == 1:  # the batch's own commit, not a fallback commit
            raise RuntimeError("disk full during batch commit")
        return real_commit()
    monkeypatch.setattr(session, "commit", flaky_commit)

    res = embed(session, emb)  # must not raise

    assert res.processed == 3
    assert res.failed == 0
    items = session.query(Item).order_by(Item.id).all()
    assert all(i.stage is Stage.EMBEDDED for i in items)


def test_commit_failure_in_fallback_isolates_the_row_and_others_still_embed(session, monkeypatch):
    """Same finding, but for the session.commit() half of the uncovered
    success path. The second item's success-path commit raises; that must
    be caught, rolled back, and turned into a FAILED row -- not escape
    embed() entirely and leave the third item unattempted."""
    _seed(session, n=3)
    emb = OneBadItemEmbedder(trigger="__never_matches__")

    calls = {"n": 0}
    real_commit = session.commit

    def flaky_commit():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real_commit()

    monkeypatch.setattr(session, "commit", flaky_commit)

    res = embed(session, emb)

    assert res.processed == 2
    assert res.failed == 1
    items = session.query(Item).order_by(Item.id).all()
    assert items[0].stage is Stage.EMBEDDED
    assert items[1].stage is Stage.FAILED
    assert "disk full" in items[1].error
    assert items[2].stage is Stage.EMBEDDED
