"use client";

// Shared page shell for /saved (D2) and /dismissed (D3) — same layout,
// different id source and action, per the D3 instruction to reuse D2's
// layout rather than build a second bespoke page.
import { useEffect, useState } from "react";
import Link from "next/link";
import { getManifest, getAllStories } from "@/lib/bundle";
import type { FeedStoryRow } from "@/lib/types";
import { StoryListRow, StoryListMissingRow } from "./StoryListRow";
import styles from "./StoryListPage.module.css";

export function StoryListPage({
  title,
  subtitle,
  getIdsOrdered,
  removeId,
  actionLabel,
  emptyTitle,
  emptyBody,
  secondaryLink,
  getFallback,
  onStoriesLoaded,
}: {
  title: string;
  subtitle: string;
  /** Returns ids in the order they were added (oldest first) — this
   * component reverses it for "most recently [saved|dismissed] first". */
  getIdsOrdered: () => number[];
  removeId: (id: number) => void;
  actionLabel: string;
  emptyTitle: string;
  emptyBody: string;
  secondaryLink?: { href: string; label: string };
  /** Where to look when the current bundle no longer carries a story.
   * /saved supplies the on-device snapshot taken at save time; /dismissed
   * supplies nothing, because a dismissed story that has aged out needs
   * no rendering — letting it go is the outcome the reader asked for. */
  getFallback?: (id: number) => FeedStoryRow | null;
  /** Called once with everything the bundle currently carries, so /saved
   * can refresh its snapshots against the live data. */
  onStoriesLoaded?: (rows: FeedStoryRow[]) => void;
}) {
  const [stories, setStories] = useState<FeedStoryRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Lazy initializer (not an effect) reads localStorage once, at first
  // render — same pattern StoryCard uses for its own saved-state read.
  // getIdsOrdered() safely returns [] during the static-export prerender
  // (no `window`), then resolves to the real order once this hydrates in
  // the browser. Local mutations after that go through handleRemove below,
  // which updates `ids` directly instead of re-reading storage.
  const [ids, setIds] = useState<number[]>(() => [...getIdsOrdered()].reverse());

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

  // Deliberately a separate effect keyed on `stories` rather than a call
  // inside the fetch above: that one owns an empty dependency array (fetch
  // once per mount) and reaching a callback prop from inside it would mean
  // either a stale closure or a ref written during render. `stories` goes
  // null -> array exactly once, so this fires once too — and
  // refreshSavedSnapshots is idempotent, so an extra call from a caller
  // passing an unstable callback costs nothing.
  useEffect(() => {
    if (stories) onStoriesLoaded?.(stories);
  }, [stories, onStoriesLoaded]);

  function handleRemove(id: number) {
    removeId(id);
    setIds((prev) => prev.filter((x) => x !== id));
  }

  const byId = new Map((stories ?? []).map((s) => [s.id, s]));

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>{title}</h1>
      <p className={styles.subtitle}>{subtitle}</p>

      {secondaryLink && (
        <Link href={secondaryLink.href} className={styles.secondaryLink}>
          {secondaryLink.label} →
        </Link>
      )}

      {error && (
        <p className={`${styles.state} ${styles.errorState}`}>Couldn&apos;t load stories: {error}</p>
      )}
      {!error && !stories && <p className={styles.state}>Loading…</p>}

      {stories && ids.length === 0 && (
        <div className={styles.empty}>
          <p className={styles.emptyTitle}>{emptyTitle}</p>
          <p className={styles.emptyBody}>{emptyBody}</p>
        </div>
      )}

      {stories && ids.length > 0 && (
        <div className={styles.list}>
          {ids.map((id) => {
            const story = byId.get(id) ?? getFallback?.(id) ?? null;
            return story ? (
              <StoryListRow
                key={id}
                story={story}
                actionLabel={actionLabel}
                onAction={() => handleRemove(id)}
              />
            ) : (
              <StoryListMissingRow
                key={id}
                id={id}
                actionLabel={actionLabel}
                onAction={() => handleRemove(id)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
