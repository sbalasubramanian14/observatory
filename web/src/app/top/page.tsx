"use client";

// Top 50 — importance as judged, not as computed.
//
// The feed's `score` is arithmetic (authority + velocity + novelty) and is
// reader-independent by design. That makes it honest about *reach* and
// blind to *meaning*: a heavily syndicated funding round and a frontier
// capability result look identical to it. This page renders the DEEP
// provider's judgement instead — see feed/stages/rank.py — grouped into
// bands, with the one-line reason each story was placed where it was.
//
// No extra network request: the ranking rides on the feed rows the client
// already fetches, so this is a filter and a sort over data in hand.

import { useEffect, useMemo, useState } from "react";
import { getManifest, getAllStories } from "@/lib/bundle";
import {
  BAND_BLURBS,
  BAND_LABELS,
  IMPORTANCE_BANDS,
  type FeedStoryRow,
  type ImportanceBand,
} from "@/lib/types";
import { StoryListRow } from "@/components/StoryListRow";
import { markDismissed } from "@/lib/personalization";
import styles from "./page.module.css";

export default function TopPage() {
  const [stories, setStories] = useState<FeedStoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(() => new Set());

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const manifest = await getManifest();
        const all = await getAllStories(manifest);
        if (!cancelled) setStories(all);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const ranked = useMemo(
    () =>
      (stories ?? [])
        .filter((s) => s.importance_rank !== null && s.importance_band !== null)
        .sort((a, b) => (a.importance_rank ?? 0) - (b.importance_rank ?? 0)),
    [stories],
  );

  const grouped = useMemo(() => {
    const out = new Map<ImportanceBand, FeedStoryRow[]>();
    for (const band of IMPORTANCE_BANDS) out.set(band, []);
    for (const s of ranked) {
      // The band was validated server-side against rank.BANDS before it was
      // ever written, so an unknown value here would mean a bundle from a
      // newer pipeline than this client. Skip rather than crash.
      const bucket = out.get(s.importance_band as ImportanceBand);
      if (bucket) bucket.push(s);
    }
    return out;
  }, [ranked]);

  function handleDismiss(id: number) {
    markDismissed(id);
    setHidden((prev) => new Set(prev).add(id));
  }

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Top {ranked.length || 50}</h1>
      <p className={styles.subtitle}>
        The most important stories in the current window, read and ranked by Claude Code
        rather than scored by formula. The feed&apos;s own importance number counts how
        many outlets picked something up; this asks what actually follows from it.
      </p>

      {error && (
        <p className={`${styles.state} ${styles.errorState}`}>
          Couldn&apos;t load stories: {error}
        </p>
      )}
      {!error && !stories && <p className={styles.state}>Loading…</p>}

      {stories && ranked.length === 0 && (
        <div className={styles.empty}>
          <p className={styles.emptyTitle}>No ranking yet</p>
          <p className={styles.emptyBody}>
            The Top 50 is written by the ranking stage of the pipeline. Run{" "}
            <code className={styles.code}>observatory.bat</code> — or{" "}
            <code className={styles.code}>feed rank</code> on its own — and it will appear
            here after the next publish.
          </p>
        </div>
      )}

      {stories &&
        IMPORTANCE_BANDS.map((band) => {
          const rows = (grouped.get(band) ?? []).filter((s) => !hidden.has(s.id));
          if (rows.length === 0) return null;
          return (
            <section key={band} className={styles.band} aria-labelledby={`band-${band}`}>
              <div className={styles.bandHeader}>
                <h2 id={`band-${band}`} className={styles.bandTitle}>
                  <span className={`${styles.bandDot} ${styles[band]}`} aria-hidden="true" />
                  {BAND_LABELS[band]}
                  <span className={styles.bandCount}>{rows.length}</span>
                </h2>
                <p className={styles.bandBlurb}>{BAND_BLURBS[band]}</p>
              </div>

              <ol className={styles.list}>
                {rows.map((story) => (
                  <li key={story.id} className={styles.item}>
                    <span className={styles.rank} aria-label={`Rank ${story.importance_rank}`}>
                      {story.importance_rank}
                    </span>
                    <div className={styles.rowWrap}>
                      <StoryListRow
                        story={story}
                        actionLabel="Dismiss"
                        onAction={() => handleDismiss(story.id)}
                      />
                      {story.importance_reason && (
                        <p className={styles.reason}>{story.importance_reason}</p>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          );
        })}
    </div>
  );
}
