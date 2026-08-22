"use client";

import { useEffect, useMemo, useState } from "react";
import { getManifest, getAllStories, getEmbeddingsIndex, getSources } from "@/lib/bundle";
import { getCentroids, rerankFeed, usePersonalization, type RankedStory } from "@/lib/personalization";
import type { FeedStoryRow, SourcesFile } from "@/lib/types";
import { StoryCard } from "@/components/StoryCard";
import styles from "./page.module.css";

const PAGE_SIZE = 20;

export default function FeedPage() {
  const { enabled } = usePersonalization();
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
        const [all, embedIndex, sourcesFile] = await Promise.all([
          getAllStories(manifest),
          getEmbeddingsIndex(manifest),
          getSources(manifest).catch(() => null),
        ]);
        if (cancelled) return;
        setStories(all);
        setEmbeddings(embedIndex);
        setEmbeddingModelId(manifest.embedding_model_id);
        setSources(sourcesFile);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const ranked: RankedStory[] = useMemo(() => {
    if (!stories) return [];
    const centroids = getCentroids();
    return rerankFeed(stories, embeddings, centroids, enabled);
    // rankVersion forces a recompute after a save/dismiss signal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stories, embeddings, enabled, rankVersion]);

  const visible = ranked
    .filter((r) => !dismissedThisSession.has(r.story.id))
    .slice(0, visibleCount);

  const strugglingSources = sources?.sources.filter((s) => s.enabled && s.consecutive_failures > 0) ?? [];

  function handleDismiss(id: number) {
    setDismissedThisSession((prev) => new Set(prev).add(id));
    setRankVersion((v) => v + 1);
  }

  return (
    <div className={styles.container}>
      <div className={styles.hero}>
        <h1 className={styles.title}>The Feed</h1>
        <p className={styles.subtitle}>
          {enabled
            ? "Ranked by a blend of published importance and your on-device fit, with room reserved for what your profile might miss."
            : "The published importance order, exactly as the pipeline ranked it."}
        </p>
      </div>

      {strugglingSources.length > 0 && (
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
          <div className={styles.statusRow}>
            <span>
              Showing <span className={styles.count}>{visible.length}</span> of{" "}
              <span className={styles.count}>{ranked.length}</span> unread stories
            </span>
          </div>

          {visible.length === 0 && (
            <p className={styles.state}>
              Nothing left unread right now — check back later, or turn off personalization to
              see the full published feed.
            </p>
          )}

          <div className={styles.list}>
            {visible.map((r) => (
              <StoryCard
                key={r.story.id}
                ranked={r}
                embedding={embeddings.get(r.story.id)}
                embeddingModelId={embeddingModelId}
                onDismiss={handleDismiss}
                onSignal={() => setRankVersion((v) => v + 1)}
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
    </div>
  );
}
