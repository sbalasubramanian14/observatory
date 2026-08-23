"use client";

import { useEffect, useMemo, useState } from "react";
import { getManifest, getAllStories, getEmbeddingsIndex, getSources } from "@/lib/bundle";
import {
  getCentroids,
  getDismissedIds,
  getReadIds,
  rerankFeed,
  usePersonalization,
  type RankedStory,
} from "@/lib/personalization";
import { sortNonImportance, useSortMode } from "@/lib/sort";
import { storyMatchesFilter, useSourceFilter } from "@/lib/sourceFilter";
import { useViewMode } from "@/lib/viewMode";
import type { FeedStoryRow, SourcesFile } from "@/lib/types";
import { StoryCard } from "@/components/StoryCard";
import { StoryDeck } from "@/components/StoryDeck";
import { SortControl } from "@/components/SortControl";
import { SourceFilter } from "@/components/SourceFilter";
import { ViewModeToggle } from "@/components/ViewModeToggle";
import styles from "./page.module.css";

const PAGE_SIZE = 20;

export default function FeedPage() {
  const { enabled } = usePersonalization();
  const [sortMode] = useSortMode();
  const [viewMode] = useViewMode();
  const [selectedSourceIds, setSelectedSourceIds] = useSourceFilter();
  const [stories, setStories] = useState<FeedStoryRow[] | null>(null);
  const [embeddings, setEmbeddings] = useState<Map<number, Float32Array>>(new Map());
  const [embeddingModelId, setEmbeddingModelId] = useState("");
  const [sources, setSources] = useState<SourcesFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [dismissedThisSession, setDismissedThisSession] = useState<Set<number>>(new Set());
  // Bumped whenever a signal is recorded so the ranking recomputes from the
  // latest centroids without re-fetching the bundle.
  const [rankVersion, setRankVersion] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const manifest = await getManifest();
        // Story rows + their lead images are what the reader is actually
        // waiting on, so they're requested (and rendered) first, standalone.
        // embeddings.bin is the largest single file in the bundle (600KB+)
        // and is only needed for personalization's fit-based re-rank — it
        // deliberately does NOT block first paint by sharing a Promise.all
        // with the story fetch, so it isn't competing with story images
        // for bandwidth on a slow connection. sources.json is tiny but
        // joins it here for the same "not needed for first paint" reason.
        const all = await getAllStories(manifest);
        if (cancelled) return;
        setStories(all);
        setEmbeddingModelId(manifest.embedding_model_id);

        const [embedIndex, sourcesFile] = await Promise.all([
          getEmbeddingsIndex(manifest),
          getSources(manifest).catch(() => null),
        ]);
        if (cancelled) return;
        setEmbeddings(embedIndex);
        setSources(sourcesFile);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Source/territory filter is applied FIRST, before sort/personalization
  // ever see the list — every other mode (recency, outlets, unread, and
  // importance's fit-based re-rank + serendipity reservation) then
  // operates on the already-narrowed set, so e.g. the 15% reserved-slot
  // budget is 15% of the filtered feed, not the whole published one, and
  // "Showing X of Y" below counts against the filtered total. Filtering
  // after ranking instead would have let a filtered-out story still eat a
  // reserved slot or a page-size budget for nothing.
  const filteredStories = useMemo(
    () => (stories ? stories.filter((s) => storyMatchesFilter(s, selectedSourceIds)) : null),
    [stories, selectedSourceIds],
  );

  const ranked: RankedStory[] = useMemo(() => {
    if (!filteredStories) return [];
    // "Importance" is the only sort mode personalization's fit-based
    // re-rank and serendipity reservation apply to — see lib/sort.ts's
    // sortNonImportance docstring for why the other three modes render
    // literally instead.
    if (sortMode === "importance") {
      const centroids = getCentroids();
      return rerankFeed(filteredStories, embeddings, centroids, enabled);
    }
    const ordered = sortNonImportance(filteredStories, sortMode, getReadIds(), getDismissedIds());
    return ordered.map((story) => ({ story, fit: 0, combined: story.score, reserved: false }));
    // rankVersion forces a recompute after a save/dismiss signal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredStories, embeddings, enabled, sortMode, rankVersion]);

  const visible = ranked
    .filter((r) => !dismissedThisSession.has(r.story.id))
    .slice(0, visibleCount);

  const strugglingSources = sources?.sources.filter((s) => s.enabled && s.consecutive_failures > 0) ?? [];
  const filterActive = selectedSourceIds.size > 0;
  // The mobile card deck's whole point is a genuinely full-viewport, one-
  // story-at-a-time read — the title/subtitle/status-row block above it
  // would otherwise eat a third of the screen before a single card
  // appears. Collapsing to just the controls row keeps sort/filter/view
  // reachable without that cost; StoryDeck itself measures and fills
  // exactly what's left (see its height-measurement effect).
  const isCards = viewMode === "cards";

  function handleDismiss(id: number) {
    setDismissedThisSession((prev) => new Set(prev).add(id));
    setRankVersion((v) => v + 1);
  }

  const subtitle =
    sortMode !== "importance"
      ? {
          recency: "Newest first, exactly as published.",
          outlets: "Stories with the most independent coverage first.",
          unread: "Unread stories first, then already-read ones — both sorted by recency.",
        }[sortMode]
      : enabled
        ? "Ranked by a blend of published importance and your on-device fit, with room reserved for what your profile might miss."
        : "The published importance order, exactly as the pipeline ranked it.";

  return (
    <div className={`${styles.container} ${isCards ? styles.containerCards : ""}`}>
      <div className={`${styles.hero} ${isCards ? styles.heroCompact : ""}`}>
        <div className={styles.heroTop}>
          <h1 className={styles.title}>The Feed</h1>
          <div className={styles.heroControls}>
            <SourceFilter sources={sources?.sources ?? []} stories={stories ?? []} />
            <SortControl />
            <ViewModeToggle />
          </div>
        </div>
        {!isCards && <p className={styles.subtitle}>{subtitle}</p>}
      </div>

      {!isCards && strugglingSources.length > 0 && (
        <div className={styles.banner}>
          <span>
            {strugglingSources.length} source{strugglingSources.length === 1 ? " is" : "s are"}{" "}
            failing to update — coverage may be incomplete.
          </span>
          <a className={styles.bannerLink} href="/sources/">
            View sources health →
          </a>
        </div>
      )}

      {error && <p className={`${styles.state} ${styles.errorState}`}>Couldn&apos;t load the feed: {error}</p>}

      {!error && !stories && (
        <div className={styles.list} aria-busy="true" aria-label="Loading feed">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className={styles.skeleton} />
          ))}
        </div>
      )}

      {stories && (
        <>
          {!isCards && (
            <div className={styles.statusRow}>
              <span>
                Showing <span className={styles.count}>{visible.length}</span> of{" "}
                <span className={styles.count}>{ranked.length}</span>{" "}
                {sortMode === "unread" ? "stories" : "unread stories"}
              </span>
            </div>
          )}

          {visible.length === 0 && (
            <div className={styles.state}>
              <p>
                {filterActive && ranked.length === 0
                  ? "No stories match the selected sources or territories."
                  : sortMode === "unread"
                    ? "No stories in the current feed window."
                    : "Nothing left unread right now — check back later, or turn off personalization to see the full published feed."}
              </p>
              {filterActive && ranked.length === 0 && (
                <button
                  type="button"
                  className={styles.clearFilterLink}
                  onClick={() => setSelectedSourceIds(new Set())}
                >
                  Clear source filter
                </button>
              )}
            </div>
          )}

          {viewMode === "cards" ? (
            <div className={styles.deckWrap}>
              <StoryDeck
                stories={visible}
                embeddings={embeddings}
                embeddingModelId={embeddingModelId}
                onDismiss={handleDismiss}
                onSignal={() => setRankVersion((v) => v + 1)}
                onNearEnd={() => setVisibleCount((c) => c + PAGE_SIZE)}
              />
            </div>
          ) : (
            <>
              <div className={styles.list}>
                {visible.map((r, i) => (
                  <StoryCard
                    key={r.story.id}
                    ranked={r}
                    embedding={embeddings.get(r.story.id)}
                    embeddingModelId={embeddingModelId}
                    onDismiss={handleDismiss}
                    onSignal={() => setRankVersion((v) => v + 1)}
                    // First couple of cards are above the fold regardless
                    // of viewport/column count — see LeadImage's docstring.
                    priority={i < 2}
                  />
                ))}
              </div>

              {visibleCount < ranked.length - dismissedThisSession.size && (
                <div className={styles.loadMoreRow}>
                  <button
                    type="button"
                    className={styles.loadMore}
                    onClick={() => setVisibleCount((c) => c + PAGE_SIZE)}
                  >
                    Load more
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
