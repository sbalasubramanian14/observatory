from feed.models import Item, Source, Stage
from feed.stages.base import run_stage


def _seed(session, n=3):
    session.add(Source(id="s", plugin="rss", config={}, cadence_minutes=30))
    for i in range(n):
        session.add(Item(source_id="s", url=f"http://x/{i}", url_hash=f"h{i}", title=f"T{i}"))
    session.commit()


def test_advances_rows_to_next_stage(session):
    _seed(session)
    res = run_stage(session, name="noop", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=lambda s, it: None)
    assert res.processed == 3 and res.failed == 0
    assert all(i.stage is Stage.NORMALIZED for i in session.query(Item).all())


def test_one_bad_row_does_not_stop_the_batch(session):
    _seed(session)
    def handler(s, item):
        if item.url_hash == "h1":
            raise RuntimeError("boom")
    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=handler)
    assert res.processed == 2 and res.failed == 1
    by_hash = {i.url_hash: i for i in session.query(Item).all()}
    assert by_hash["h0"].stage is Stage.NORMALIZED
    assert by_hash["h2"].stage is Stage.NORMALIZED
    assert by_hash["h1"].stage is Stage.FAILED
    assert "boom" in by_hash["h1"].error


def test_only_claims_matching_stage(session):
    _seed(session)
    session.query(Item).filter_by(url_hash="h0").one().stage = Stage.EMBEDDED
    session.commit()
    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=lambda s, it: None)
    assert res.processed == 2


def test_limit_is_respected(session):
    _seed(session, n=10)
    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=lambda s, it: None, limit=4)
    assert res.processed == 4


def test_handler_mutation_is_discarded_on_failure_and_later_rows_still_usable(session):
    """Reproduces the post-rollback staleness trap directly.

    The handler mutates the item's title (dirtying the ORM-tracked object)
    before raising, so a naive re-fetch that reuses `session.get()` returning
    the *same* still-attached-but-expired instance must not resurrect that
    uncommitted mutation. And critically, the item processed *after* the
    failing row must still be writable/committable using an object that was
    loaded into the session before the rollback happened.
    """
    _seed(session)

    def handler(s, item):
        if item.url_hash == "h1":
            item.title = "MUTATED-SHOULD-NOT-PERSIST"
            raise RuntimeError("boom")

    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=handler)
    assert res.processed == 2 and res.failed == 1

    by_hash = {i.url_hash: i for i in session.query(Item).all()}
    assert by_hash["h0"].stage is Stage.NORMALIZED
    assert by_hash["h2"].stage is Stage.NORMALIZED
    assert by_hash["h1"].stage is Stage.FAILED
    # the mutation made just before the raise must have been rolled back
    assert by_hash["h1"].title == "T1"


def test_consecutive_failures_do_not_break_the_run(session):
    """Two rollbacks back-to-back must not corrupt the loop or the surviving row."""
    _seed(session)

    def handler(s, item):
        if item.url_hash in ("h0", "h1"):
            raise RuntimeError(f"boom-{item.url_hash}")

    res = run_stage(session, name="x", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=handler)
    assert res.processed == 1 and res.failed == 2
    assert len(res.errors) == 2
    by_hash = {i.url_hash: i for i in session.query(Item).all()}
    assert by_hash["h0"].stage is Stage.FAILED
    assert by_hash["h1"].stage is Stage.FAILED
    assert by_hash["h2"].stage is Stage.NORMALIZED
    assert "boom-h0" in by_hash["h0"].error
    assert "boom-h1" in by_hash["h1"].error


def test_stage_result_field_names_and_error_tuples(session):
    """Locks the exact StageResult surface the brief specifies."""
    _seed(session)

    def handler(s, item):
        if item.url_hash == "h1":
            raise ValueError("bad row")

    res = run_stage(session, name="checked", claim_stage=Stage.COLLECTED,
                    next_stage=Stage.NORMALIZED, handler=handler)
    assert res.name == "checked"
    assert res.processed == 2
    assert res.failed == 1
    assert len(res.errors) == 1
    item_id, message = res.errors[0]
    h1 = session.query(Item).filter_by(url_hash="h1").one()
    assert item_id == h1.id
    assert "bad row" in message
