"use client";

import { useState } from "react";
import Link from "next/link";
import type { StoryDetail } from "@/lib/types";
import type { RankedStory } from "@/lib/personalization";
import { markSaved, markDismissed, recordSignal, getSavedIds } from "@/lib/personalization";
import { getStoryDetail } from "@/lib/bundle";
import { formatAge, formatSourceName, hostnameOf } from "@/lib/format";
import { CategoryTag } from "./CategoryTag";
import { ImportanceMeter } from "./ImportanceMeter";
import { LeadImage } from "./LeadImage";
import { SourceAvatar } from "./SourceAvatar";
import styles from "./StoryCard.module.css";

export function StoryCard({
  ranked,
  embedding,
  embeddingModelId,
  onDismiss,
  onSignal,
  priority = false,
}: {
  ranked: RankedStory;
  embedding: Float32Array | undefined;
  embeddingModelId: string;
  onDismiss: (id: number) => void;
  onSignal?: () => void;
  /** Above-the-fold cards — see LeadImage's `priority` prop. */
  priority?: boolean;
}) {
  const { story, reserved } = ranked;
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState<StoryDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(() => getSavedIds().has(story.id));
  // Issue 1: the summary is clamped to 3 lines below (styles.summary) with
  // no way to read the rest — expand-in-place, not a navigation, so the
  // reader doesn't lose their place in the feed. A plain <button> (not a
  // clickable <p>/div with a synthetic role) so Tab/Enter/Space and touch
  // all work for free via native button semantics, and it lives entirely
  // inside StoryCard — the mobile card deck (StoryDeckCard) already shows
  // the full summary with its own internal scroll, so there's no swipe-
  // gesture surface to fight here.
  const [summaryExpanded, setSummaryExpanded] = useState(false);

  async function toggleExpand() {
    const next = !expanded;
    setExpanded(next);
    if (next && !detail && !loading) {
      setLoading(true);
      try {
        const d = await getStoryDetail(story.detail_path);
        setDetail(d);
      } finally {
        setLoading(false);
      }
    }
  }

  function handleSave() {
    // Pass the row itself: /saved must still be able to render this card
    // after the story ages out of the bundle's rolling window.
    markSaved(story.id, story);
    setSaved(true);
    if (embedding) recordSignal("save", embedding, embeddingModelId);
    onSignal?.();
  }

  function handleDismiss() {
    markDismissed(story.id);
    if (embedding) recordSignal("dismiss", embedding, embeddingModelId);
    onDismiss(story.id);
  }

  const primarySource = story.source_ids[0] ?? "unknown";

  return (
    <article className={`${styles.card} ${reserved ? styles.reserved : ""}`}>
      <div className={styles.identity}>
        <SourceAvatar sourceId={primarySource} />
        <div className={styles.identityText}>
          <span className={styles.sourceName}>
            {formatSourceName(primarySource)}
            {story.outlet_count > 1 ? ` +${story.outlet_count - 1}` : ""}
          </span>
          <span className={styles.timestamp}>{formatAge(story.updated_at)}</span>
        </div>
        <div className={styles.actionRow}>
          <button
            type="button"
            className={`${styles.iconButton} ${saved ? styles.iconButtonActive : ""}`}
            onClick={handleSave}
            title="Save — nudges your positive interest profile"
            aria-label="Save story"
            aria-pressed={saved}
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill={saved ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2">
              <path d="M6 4h12v16l-6-4-6 4V4z" />
            </svg>
          </button>
          <button
            type="button"
            className={styles.iconButton}
            onClick={handleDismiss}
            title="Dismiss — nudges your negative interest profile and hides this story"
            aria-label="Dismiss story"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>
      </div>

      {reserved && (
        <span className={styles.reservedBadge}>◆ Outside your usual — high importance</span>
      )}

      <div className={styles.main}>
        <div className={styles.metaTop}>
          <CategoryTag category={story.category} />
          <ImportanceMeter score={story.score} />
        </div>
        {/* The "open" signal is recorded once, on the story detail page's
            mount effect — not here — so it fires exactly once per open
            regardless of whether the reader arrived via this card, a
            direct link, or browser back/forward. */}
        <Link href={`/story/?id=${story.id}`} className={styles.headline}>
          {story.title}
        </Link>
        <LeadImage src={story.lead_image_url} alt="" variant="card" priority={priority} />
        <p
          className={`${styles.summary} ${!story.summary ? styles.summaryEmpty : ""} ${
            summaryExpanded ? styles.summaryExpanded : ""
          }`}
        >
          {story.summary ?? "No summary yet — this story hasn't been through Tier 1 processing."}
        </p>
        {story.summary && (
          <button
            type="button"
            className={styles.summaryToggle}
            onClick={() => setSummaryExpanded((v) => !v)}
            aria-expanded={summaryExpanded}
          >
            {summaryExpanded ? "Show less" : "Read more"}
          </button>
        )}
        <div className={styles.metaBottom}>
          <span className={styles.metaItem}>
            <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" rx="1" />
              <rect x="14" y="3" width="7" height="7" rx="1" />
              <rect x="3" y="14" width="7" height="7" rx="1" />
              <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
            {story.outlet_count} outlet{story.outlet_count === 1 ? "" : "s"}
          </span>
          <span className={styles.metaItem}>
            <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
            {story.item_count} item{story.item_count === 1 ? "" : "s"}
          </span>
          <button type="button" className={styles.expandButton} onClick={toggleExpand}>
            {expanded ? "Hide evidence" : "Show evidence"}
            <svg
              className={`${styles.chevron} ${expanded ? styles.chevronOpen : ""}`}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
        </div>
      </div>
      {expanded && (
        <div className={styles.detail}>
          {loading && <p className={styles.detailLoading}>Loading evidence…</p>}
          {detail && (
            <>
              {detail.analysis && (
                <div className={styles.analysis}>
                  <span className={styles.analysisLabel}>
                    Tier 2 analysis{detail.analysis_provider ? ` — ${detail.analysis_provider}` : ""}
                  </span>
                  <p className={styles.analysisText}>{detail.analysis}</p>
                </div>
              )}
              <p className={styles.evidenceLabel}>
                Evidence ({detail.evidence.length} article{detail.evidence.length === 1 ? "" : "s"})
              </p>
              <ul className={styles.evidenceList}>
                {detail.evidence.map((ev) => (
                  <li key={ev.id}>
                    <a
                      className={styles.evidenceLink}
                      href={ev.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span>{ev.title}</span>
                      <span className={styles.evidenceSource}>{hostnameOf(ev.url)}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </article>
  );
}
