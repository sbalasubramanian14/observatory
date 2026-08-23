from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from sqlalchemy import func, select
import feed.sources  # noqa: F401  (registers plugins)
from feed.catalogue import DEFAULT_CATALOGUE_PATH, load_catalogue
from feed.clustering.adjudicate import NullAdjudicator, ThresholdAdjudicator
from feed.config import BulkProviderConfig, Config, load_config
from feed.db import create_all, make_engine, make_session_factory
from feed.embedding import build_embedder
from feed.embedding.resolve import resolve
from feed.imaging import DEFAULT_HOST_DELAY, DEFAULT_MAX_WORKERS, resolve_images
from feed.models import Item, Source, Stage, Story
from feed.providers.base import Provider, ProviderError
from feed.providers.claude_code import ClaudeCodeProvider
from feed.providers.failover import FailoverProvider
from feed.providers.gemini import GeminiProvider
from feed.providers.health import ProviderHealthTracker
from feed.providers.openai_compatible import OpenAICompatibleProvider
from feed.providers.router import Router
from feed.sources.registry import known_plugins
from feed.stages.base import DEFAULT_MAX_ROUNDS, drain
from feed.stages.cluster import cluster
from feed.stages.collect import collect
from feed.stages.embed import embed
from feed.stages.enrich import enrich
from feed.stages.normalize import normalize
from feed.stages.publish import publish
from feed.stages.relevance import gate_relevance, sweep_existing_corpus
from feed.stages.score import score_stories
from feed.stages.sync import sync_sources
from feed.doctor import run_doctor
from feed.pipeline import DEFAULT_ALMANAC_DIR, DEFAULT_ALMANAC_REPO, DEFAULT_KEEP_LOGS, \
    DEFAULT_STAGE_TIMEOUT, run_pipeline

log = logging.getLogger(__name__)


def _session(cfg: Config):
    engine = make_engine(cfg.database.url)
    return engine, make_session_factory(engine)


def _build_adjudicator(cfg: Config) -> NullAdjudicator:
    """Build the Phase 1 adjudicator, wired to per-model thresholds.

    Ruling 1 (task-14 brief): Task 11 moved per-model merge thresholds into
    the adjudicator via the keyword-only `threshold_for` provider. Without
    passing `threshold_for=cfg.clustering.threshold_for` here, the real
    pipeline would silently ignore the per-model threshold map -- including
    the 0.35 MiniLM threshold measured in Task 12 and stored in feed.toml --
    and always fall back to the single global `merge_threshold` instead.
    See tests/test_cli.py::test_run_wires_per_model_threshold_into_adjudicator.
    """
    return NullAdjudicator(
        ThresholdAdjudicator(
            merge_threshold=cfg.clustering.merge_threshold,
            threshold_for=cfg.clustering.threshold_for,
        )
    )


def _build_bulk_provider(entry: BulkProviderConfig, *, max_retries: int,
                         backoff_base: float) -> Provider:
    """One [[providers.bulk]] entry -> one live Provider instance. Model
    name, base URL, and env var all come from feed.toml (never hardcoded
    here) -- see BulkProviderConfig's docstring for why that matters.
    """
    if entry.kind == "gemini":
        return GeminiProvider(model=entry.model, timeout=entry.timeout,
                              max_retries=max_retries, backoff_base=backoff_base)
    return OpenAICompatibleProvider(
        name=entry.name, model=entry.model, base_url=entry.base_url,
        env_var=entry.env_var, timeout=entry.timeout,
        max_retries=max_retries, backoff_base=backoff_base,
    )


def _build_router(cfg: Config) -> Router:
    """Wire the spec 3.5 provider router: a priority-ordered, failing-over
    BULK chain (requirement 1) as BULK, Claude Code as DEEP. Building this
    never itself fails on a missing API key -- that surfaces per-story as a
    ProviderError, isolated by enrich stage's own failure handling, not as
    a crash here. Kept out of `_session()`'s call path entirely so a plain
    `feed run` (no --enrich) never even imports/constructs a provider.

    Only ENABLED entries (cfg.providers.bulk[*].enabled) join the chain --
    Cerebras ships disabled by default (measured 402 "Payment required").
    A bare feed.toml with no [[providers.bulk]] entries at all falls back
    to a single Gemini provider, preserving this function's behaviour
    before this task existed.
    """
    entries = [e for e in cfg.providers.bulk if e.enabled]
    if entries:
        providers = [
            _build_bulk_provider(e, max_retries=cfg.providers.max_retries,
                                 backoff_base=cfg.providers.backoff_base)
            for e in entries
        ]
        engine = make_engine(cfg.database.url)
        create_all(engine)  # provider_status is a new table; must exist
        bulk: Provider = FailoverProvider(
            providers, session_factory=make_session_factory(engine),
            rate_limit_disable_threshold=cfg.providers.rate_limit_disable_threshold,
        )
    else:
        bulk = GeminiProvider(model=cfg.providers.gemini_model, timeout=cfg.providers.gemini_timeout)
    deep = ClaudeCodeProvider(timeout=cfg.providers.claude_code_timeout)
    return Router(bulk=bulk, deep=deep)


def cmd_init(args, cfg: Config) -> int:
    engine, _ = _session(cfg)
    create_all(engine)
    print(f"initialised {cfg.database.url}")
    return 0


def cmd_sources_add(args, cfg: Config) -> int:
    if args.plugin not in known_plugins():
        print(f"unknown plugin {args.plugin!r}; known: {known_plugins()}", file=sys.stderr)
        return 2
    try:
        conf = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        print(f"--config-json is not valid JSON: {exc}", file=sys.stderr)
        return 2
    _, factory = _session(cfg)
    with factory() as s:
        s.merge(Source(id=args.id, plugin=args.plugin, config=conf,
                       cadence_minutes=args.cadence, authority=args.authority,
                       max_backfill_days=args.max_backfill_days))
        s.commit()
    print(f"added source {args.id}")
    return 0


def cmd_sources_list(args, cfg: Config) -> int:
    _, factory = _session(cfg)
    with factory() as s:
        for src in s.scalars(select(Source).order_by(Source.id)):
            if not src.enabled:
                state = "disabled"
            elif src.consecutive_failures == 0:
                state = "ok"
            else:
                state = f"FAILING x{src.consecutive_failures}"
            if src.coverage_warning:
                state += "  COVERAGE-WARN"
            territory = src.territory or "-"
            print(f"{src.id:<28} {src.plugin:<18} {territory:<15} "
                 f"every {src.cadence_minutes:>4}m  {state}")
    return 0


def cmd_sources_sync(args, cfg: Config) -> int:
    """Reconciles the `source` table against sources.catalogue.toml (spec:
    "adding a source is editing a list", replacing ad-hoc `feed sources
    add` calls). See feed.stages.sync.sync_sources for the reconciliation
    rules (add / update / leave unchanged / disable / delete).
    """
    try:
        entries = load_catalogue(args.catalogue)
    except (FileNotFoundError, ValueError) as exc:
        print(f"sources sync failed: {exc}", file=sys.stderr)
        return 2
    _, factory = _session(cfg)
    with factory() as s:
        res = sync_sources(s, entries)
    print(f"sources sync: added={len(res.added)} updated={len(res.updated)} "
          f"unchanged={len(res.unchanged)} disabled={len(res.disabled)} "
          f"deleted={len(res.deleted)}")
    for label, ids in (("added", res.added), ("updated", res.updated),
                      ("disabled", res.disabled), ("deleted", res.deleted)):
        if ids:
            print(f"  {label:<10}{', '.join(ids)}")
    return 0


def cmd_run(args, cfg: Config) -> int:
    """Run one full pipeline pass: collect, then drain normalize/embed/
    cluster/score to completion.

    I1 fix: each of those four stages claims a fixed-size batch per call
    (normalize=100, embed=cfg.embedding.batch_size, cluster=200 by
    default). Calling each stage exactly once -- the previous behaviour --
    left any remainder beyond one batch stranded at the prior stage with no
    warning, and the backlog would only grow on a source producing more
    than a batch's worth of items between runs. `drain()` loops each stage
    until it reports zero progress, so a single `feed run` invocation
    genuinely empties the queue (bounded by drain()'s own safety cap, so a
    stage that can never make progress still can't hang this forever).
    """
    _, factory = _session(cfg)
    backend, model, device = resolve(cfg.embedding)
    print(f"embedding: backend={backend} model={model} device={device}")
    embedder = build_embedder(cfg.embedding)
    adjudicator = _build_adjudicator(cfg)
    with factory() as s:
        c = collect(s, cfg=cfg.collect)
        print(f"collect:   new={c.new_items} dupes={c.skipped_duplicates} "
              f"source_errors={len(c.source_errors)}")
        for name, stage_fn in [
            ("normalize", lambda: normalize(s)),
            ("embed", lambda: embed(s, embedder, limit=cfg.embedding.batch_size)),
        ]:
            res = drain(stage_fn)
            print(f"{name+':':<11}ok={res.processed} failed={res.failed} "
                  f"rounds={res.rounds}")
            if res.rounds >= DEFAULT_MAX_ROUNDS:
                log.warning(
                    "%s: hit the %d-round drain safety cap; the queue may "
                    "not be fully drained -- check for a stuck row",
                    name, DEFAULT_MAX_ROUNDS,
                )

        # Issue 3: the off-topic gate. Runs after embed (reuses the
        # embedding Tier 0 just computed -- no extra encode call) and
        # before cluster (so a rejected item never joins a story in the
        # first place). NOT drain()'d -- see gate_relevance's own
        # docstring for why a single pass over every currently-EMBEDDED
        # item is the right amount of work, unlike the batched stages
        # above.
        rel_res = gate_relevance(s, cfg.relevance, embedder)
        print(f"relevance: ok={rel_res.processed} failed={rel_res.failed} "
              f"rejected={rel_res.rejected}")

        for name, stage_fn in [
            ("cluster", lambda: cluster(s, cfg.clustering, adjudicator)),
            ("score", lambda: score_stories(s, cfg.scoring)),
        ]:
            res = drain(stage_fn)
            print(f"{name+':':<11}ok={res.processed} failed={res.failed} "
                  f"rounds={res.rounds}")
            if res.rounds >= DEFAULT_MAX_ROUNDS:
                log.warning(
                    "%s: hit the %d-round drain safety cap; the queue may "
                    "not be fully drained -- check for a stuck row",
                    name, DEFAULT_MAX_ROUNDS,
                )

        # Both stages are opt-in (spec build order: enrich is "the only
        # paid stage", publish pushes to a public repo) so the default
        # `feed run` stays free and side-effect-free outside the local db.
        if getattr(args, "enrich", False):
            router = _build_router(cfg)
            er = enrich(s, router, cfg.providers)
            print(f"enrich:    tier1 ok={er.tier1_processed} failed={er.tier1_failed}  "
                  f"tier2 ok={er.tier2_processed} failed={er.tier2_failed} "
                  f"degraded={er.tier2_degraded}")

        if getattr(args, "publish", False):
            out_dir = args.out or cfg.publish.out_dir
            pr = publish(s, cfg.publish, out_dir)
            if pr.published:
                print(f"publish:   stories={pr.story_count} pages={pr.page_count} "
                      f"pruned={pr.pruned} out={pr.out_dir}")
            else:
                print(f"publish:   FAILED: {pr.error}", file=sys.stderr)
    return 0


def cmd_enrich(args, cfg: Config) -> int:
    _, factory = _session(cfg)
    router = _build_router(cfg)
    with factory() as s:
        er = enrich(s, router, cfg.providers)
    print(f"tier1: ok={er.tier1_processed} failed={er.tier1_failed}")
    print(f"tier2: ok={er.tier2_processed} failed={er.tier2_failed} "
          f"degraded={er.tier2_degraded}")
    for story_id, msg in er.errors:
        print(f"  story={story_id}: {msg}", file=sys.stderr)
    return 0


def cmd_providers(args, cfg: Config) -> int:
    """Requirement 6: probe each configured provider and print enabled /
    reachable / model / latency / today's usage -- "how the user diagnoses
    'why is nothing being summarized'." Makes one real, cheap completion
    call per enabled provider (this command exists specifically to answer
    "is the network path to this provider actually working right now",
    which a stubbed health() check cannot answer), so it is never exercised
    against real providers in the default test suite -- see
    tests/test_cli_providers.py, which monkeypatches _build_bulk_provider.
    """
    engine = make_engine(cfg.database.url)
    create_all(engine)
    session_factory = make_session_factory(engine)

    header = f"{'provider':<12}{'kind':<18}{'enabled':<9}{'reachable':<22}{'model':<42}{'latency':<10}today"
    print(header)
    for entry in cfg.providers.bulk:
        provider = _build_bulk_provider(
            entry, max_retries=cfg.providers.max_retries,
            backoff_base=cfg.providers.backoff_base,
        )
        with session_factory() as s:
            tracker = ProviderHealthTracker(
                s, rate_limit_disable_threshold=cfg.providers.rate_limit_disable_threshold,
            )
            status = tracker.status_today(entry.name)
            usage = f"{status.successes}ok/{status.failures}fail"
            if status.disabled:
                usage += f" DISABLED ({status.disabled_reason})"

        reachable, latency = "n/a (disabled)", "n/a"
        if entry.enabled:
            h = provider.health()
            if not h.healthy:
                reachable, latency = f"no ({h.detail})", "n/a"
            else:
                start = time.monotonic()
                try:
                    provider.complete("Reply with only the single word: OK.")
                    latency = f"{(time.monotonic() - start) * 1000:.0f}ms"
                    reachable = "yes"
                except ProviderError as exc:
                    reachable = f"no ({exc})"
        print(f"{entry.name:<12}{entry.kind:<18}{str(entry.enabled):<9}"
              f"{reachable:<22}{entry.model:<42}{latency:<10}{usage}")

    # Claude Code (DEEP, spec 3.5) is local, not part of the BULK failover
    # chain, and its health() is deliberately optimistic (no cheap
    # network/subprocess probe exists that doesn't itself cost a call) --
    # see feed/providers/claude_code.py. Listed for completeness, not probed.
    deep = ClaudeCodeProvider(timeout=cfg.providers.claude_code_timeout)
    print(f"{'claude-code':<12}{'local-cli':<18}{'True':<9}"
          f"{'n/a (local CLI)':<22}{deep.model:<42}{'n/a':<10}n/a")
    return 0


def cmd_publish(args, cfg: Config) -> int:
    _, factory = _session(cfg)
    out_dir = args.out or cfg.publish.out_dir
    with factory() as s:
        pr = publish(s, cfg.publish, out_dir)
    if not pr.published:
        print(f"publish failed: {pr.error}", file=sys.stderr)
        return 1
    print(f"published {pr.story_count} stories across {pr.page_count} page(s) "
          f"to {pr.out_dir} (pruned {pr.pruned} stale file(s))")
    return 0


def cmd_pipeline(args, cfg: Config) -> int:
    """The one-click entry point (spec Phase F): sources sync -> run ->
    enrich -> publish -> push the bundle to observatory-almanac. See
    feed.pipeline.run_pipeline for the failure policy and PIPELINE-CLI.md
    for the exit code contract. This is what scripts/run-pipeline.ps1 (and
    therefore observatory.bat / the scheduled task) actually invokes.
    """
    result = run_pipeline(
        config=args.config,
        cwd=Path.cwd(),
        catalogue=args.catalogue,
        out_dir=args.out,
        logs_dir=args.logs_dir,
        keep_logs=args.keep_logs,
        almanac_dir=args.almanac_dir,
        almanac_repo=args.almanac_repo,
        skip_almanac_push=args.skip_almanac_push,
        stage_timeout=args.stage_timeout,
    )
    return result.exit_code


def cmd_relevance_sweep(args, cfg: Config) -> int:
    """Issue 3's corpus cleanup: retroactively apply the off-topic gate to
    stories already ingested before it existed (e.g. the Verge film
    review that reached publish off the old whole-site feed). Dry-run by
    default -- prints every offending item and how many there are WITHOUT
    touching the database; pass --apply to actually detach/reject them.
    `feed publish` (and the one-click pipeline) must be re-run afterward
    for the live site to reflect the cleanup -- this command only touches
    feed.db.
    """
    _, factory = _session(cfg)
    embedder = build_embedder(cfg.embedding)
    with factory() as s:
        res = sweep_existing_corpus(s, cfg.relevance, embedder, apply=args.apply,
                                    source_ids=args.source or None)
    mode = "APPLIED" if args.apply else "dry-run (pass --apply to make this permanent)"
    print(f"relevance sweep [{mode}]: scanned={res.scanned} "
          f"off_topic={len(res.findings)} stories_deleted={res.stories_deleted}")
    for f in res.findings:
        print(f"  item={f.item_id:<7} story={f.story_id!s:<7} "
             f"[{f.story_category or '-'} {f.story_score if f.story_score is not None else '-'}]  "
             f"{f.title[:70]!r}  source={f.source_id}")
        print(f"      {f.reason}")
    return 0


def cmd_doctor(args, cfg: Config) -> int:
    """Preflight diagnosis (spec Phase F): what turns "it didn't work" into
    a diagnosis. See feed.doctor.run_doctor for the individual checks.
    """
    report = run_doctor(cfg, probe_providers=not args.no_probe)
    report.print(file=sys.stdout)
    return 0 if report.ok else 1


def cmd_backfill_images(args, cfg: Config) -> int:
    """One-off (but safely re-runnable) sweep of the existing corpus for
    items that never got a chance at the og:image fallback -- specifically
    every item normalized before that fallback existed (see
    feed.stages.normalize's D0 history: the fallback only runs going
    forward, on items normalize() itself just processed; it never revisits
    an item that already advanced past Stage.NORMALIZED). Uses the exact
    same feed.imaging.resolve_images concurrent/rate-limited pass as the
    normalize stage's own post-step -- one implementation, two call sites.

    Resumable: candidates are selected by `image_checked_at IS NULL`, and
    resolve_images() commits each item's result (image_url and
    image_checked_at together) as soon as that item's fetch completes, not
    in one batch at the end. Interrupting this command (Ctrl-C, a crash,
    hitting --limit) and re-running it later picks up exactly where it
    left off -- already-checked items are excluded from the next run's
    candidate query without a separate "was this run interrupted" flag.
    """
    _, factory = _session(cfg)
    with factory() as s:
        stmt = (
            select(Item)
            .where(
                (Item.image_url.is_(None)) | (Item.image_url == ""),
                Item.image_checked_at.is_(None),
                Item.stage != Stage.FAILED,
            )
            .order_by(Item.id)
        )
        if args.limit:
            stmt = stmt.limit(args.limit)
        items = list(s.scalars(stmt))
        print(f"backfill-images: {len(items)} item(s) eligible "
              f"(max_workers={args.max_workers} host_delay={args.host_delay}s)")
        if not items:
            return 0

        def _progress(done: int, total: int) -> None:
            if done == total or done % 20 == 0:
                print(f"  ...{done}/{total} attempted", flush=True)

        result = resolve_images(
            s, items, max_workers=args.max_workers, host_delay=args.host_delay,
            timeout=args.timeout, on_progress=_progress,
        )
        attempted = sum(sum(counts.values()) for counts in result.by_source.values())
        print(f"backfill-images: attempted={attempted} gained={result.gained} "
              f"sources={result.total_sources}")
        for source_id in sorted(result.by_source):
            counts = result.by_source[source_id]
            gained = counts.pop("gained", 0)
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            suffix = f"  ({reasons})" if reasons else ""
            print(f"  {source_id:<28} gained={gained}{suffix}")
    return 0


def cmd_stats(args, cfg: Config) -> int:
    _, factory = _session(cfg)
    with factory() as s:
        print("items by stage:")
        for stage, n in s.execute(
            select(Item.stage, func.count()).group_by(Item.stage)
        ):
            print(f"  {stage.value:<12}{n}")
        # Issue 3: rejections must be visible, never a silent drop -- see
        # feed.stages.relevance. The stage-breakdown loop above already
        # shows "rejected  N" (Item.stage is Stage.REJECTED is just
        # another stage value), but a dedicated line plus a by-source
        # breakdown is what actually answers "which source is drifting
        # off-topic" at a glance, without grepping the db by hand.
        rejected_total = s.scalar(
            select(func.count()).where(Item.stage == Stage.REJECTED)
        ) or 0
        print(f"rejected as off-topic: {rejected_total}")
        if rejected_total:
            for source_id, n in s.execute(
                select(Item.source_id, func.count())
                .where(Item.stage == Stage.REJECTED)
                .group_by(Item.source_id)
                .order_by(func.count().desc())
            ):
                print(f"    {source_id:<28}{n}")

        total = s.scalar(select(func.count()).select_from(Story)) or 0
        print(f"stories: {total}")
        print("top stories by importance:")
        for st in s.scalars(
            select(Story).where(Story.score.is_not(None))
            .order_by(Story.score.desc()).limit(10)
        ):
            print(f"  {st.score:.3f}  [{st.outlet_count} outlets]  {st.title[:70]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="feed")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    run = sub.add_parser("run")
    run.add_argument("--enrich", action="store_true",
                     help="also run Tier 1/Tier 2 LLM enrichment (spends API/CLI calls)")
    run.add_argument("--publish", action="store_true",
                     help="also publish the static bundle after enriching")
    run.add_argument("--out", type=Path, default=None,
                     help="bundle output directory (default: [publish].out_dir)")
    run.set_defaults(func=cmd_run)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    enrich_p = sub.add_parser("enrich")
    enrich_p.set_defaults(func=cmd_enrich)

    sub.add_parser("providers").set_defaults(func=cmd_providers)

    publish_p = sub.add_parser("publish")
    publish_p.add_argument("--out", type=Path, default=None,
                           help="bundle output directory (default: [publish].out_dir)")
    publish_p.set_defaults(func=cmd_publish)

    pipeline_p = sub.add_parser(
        "pipeline",
        help="one-click run: sources sync -> run -> enrich -> publish -> push to observatory-almanac",
    )
    pipeline_p.add_argument("--catalogue", type=Path, default=None,
                            help="path to the source catalogue TOML (default: sources.catalogue.toml)")
    pipeline_p.add_argument("--out", type=Path, default=None,
                            help="bundle output directory (default: [publish].out_dir)")
    pipeline_p.add_argument("--logs-dir", type=Path, default=None,
                            help="directory for timestamped run logs (default: ./logs)")
    pipeline_p.add_argument("--keep-logs", type=int, default=DEFAULT_KEEP_LOGS,
                            help=f"number of past run logs to retain (default: {DEFAULT_KEEP_LOGS})")
    pipeline_p.add_argument("--almanac-dir", type=Path, default=None,
                            help="local clone of the almanac repo (default: ./.cache/observatory-almanac)")
    pipeline_p.add_argument("--almanac-repo", default=DEFAULT_ALMANAC_REPO,
                            help=f"GitHub repo the bundle is pushed to (default: {DEFAULT_ALMANAC_REPO})")
    pipeline_p.add_argument("--skip-almanac-push", action="store_true",
                            help="run every stage but do not push to the almanac repo")
    pipeline_p.add_argument("--stage-timeout", type=float, default=DEFAULT_STAGE_TIMEOUT,
                            help=f"per-stage subprocess timeout in seconds (default: {DEFAULT_STAGE_TIMEOUT:.0f})")
    pipeline_p.set_defaults(func=cmd_pipeline)

    doctor_p = sub.add_parser("doctor", help="preflight: diagnose why the pipeline isn't working")
    doctor_p.add_argument("--no-probe", action="store_true",
                          help="skip live network probes of each LLM provider (key-presence check only)")
    doctor_p.set_defaults(func=cmd_doctor)

    backfill_p = sub.add_parser(
        "backfill-images",
        help="fetch og:image for existing items that never got a chance at it",
    )
    backfill_p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                            help="bounded concurrency for the fetch pool "
                                 f"(default: {DEFAULT_MAX_WORKERS})")
    backfill_p.add_argument("--host-delay", type=float, default=DEFAULT_HOST_DELAY,
                            help="minimum seconds between two requests to the "
                                 f"same host (default: {DEFAULT_HOST_DELAY})")
    backfill_p.add_argument("--timeout", type=float, default=15.0,
                            help="per-request timeout in seconds (default: 15.0)")
    backfill_p.add_argument("--limit", type=int, default=None,
                            help="attempt at most N items (default: all eligible)")
    backfill_p.set_defaults(func=cmd_backfill_images)

    srcs = sub.add_parser("sources").add_subparsers(dest="sub", required=True)
    add = srcs.add_parser("add")
    add.add_argument("--id", required=True)
    add.add_argument("--plugin", required=True)
    add.add_argument("--config-json", default="{}")
    add.add_argument("--cadence", type=int, default=30)
    add.add_argument("--authority", type=float, default=0.5)
    add.add_argument("--max-backfill-days", type=int, default=None,
                     help="per-source override of [collect].max_backfill_days "
                          "(default: use the global value)")
    add.set_defaults(func=cmd_sources_add)
    srcs.add_parser("list").set_defaults(func=cmd_sources_list)
    sync = srcs.add_parser("sync")
    sync.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH,
                      help="path to the source catalogue TOML (default: sources.catalogue.toml)")
    sync.set_defaults(func=cmd_sources_sync)

    rel = sub.add_parser("relevance").add_subparsers(dest="sub", required=True)
    sweep = rel.add_parser("sweep")
    sweep.add_argument("--apply", action="store_true",
                       help="actually detach/reject off-topic items (default: dry-run, report only)")
    sweep.add_argument("--source", action="append", default=[],
                       help="scope the sweep to this source id (repeatable); "
                            "default: every source")
    sweep.set_defaults(func=cmd_relevance_sweep)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args, load_config(args.config))
