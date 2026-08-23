"use client";

import { useState } from "react";
import Link from "next/link";
import type { StoryDetail } from "@/lib/types";
import type { RankedStory } from "@/lib/personalization";
import { markSaved, markDismissed, recordSignal, getSavedIds } from "@/lib/personalization";
import { getStoryDetail } from "@/lib/bundle";
import { formatAge, hostnameOf } from "@/lib/format";
import { CategoryTag } from "./CategoryTag";
import { ImportanceMeter } from "./ImportanceMeter";
import { LeadImage } from "./LeadImage";
import styles from "./StoryCard.module.css";

export function StoryCard({
  ranked,
  embedding,
  embeddingModelId,
  onDismiss,
  onSignal,
}: {
  ranked: RankedStory;
  embedding: Float32Array | undefined;
  embeddingModelId: string;
  onDismiss: (id: number) => void;
  onSignal?: () => void;
}) {
  const { story, reserved } = ranked;
  const [expanded, setExpanded] = useState(false);
  const [detail, setDetail] = useState<StoryDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(() => getSavedIds().has(story.id));

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
    markSaved(story.id);
    setSaved(true);
    if (embedding) recordSignal("save", embedding, embeddingModelId);
    onSignal?.();
  }

  function handleDismiss() {
    markDismissed(story.id);
    if (embedding) recordSignal("dismiss", embedding, embeddingModelId);
    onDismiss(story.id);
  }

  return (
    <article className={`${styles.card} ${reserved ? styles.reserved : ""}`}>
      <div className={styles.main}>
        <LeadImage src={story.lead_image_url} alt="" variant="thumb" />
        <div className={styles.body}>
          {reserved && (
            <span className={styles.reservedBadge}>◆ Outside your usual — high importance</span>
          )}
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
          <p className={`${styles.summary} ${!story.summary ? styles.summaryEmpty : ""}`}>
            {story.summary ?? "No summary yet — this story hasn't been through Tier 1 processing."}
          </p>
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
            <span className={styles.metaItem}>
              <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 7v5l3 3" />
              </svg>
              {formatAge(story.updated_at)}
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
        <div className={styles.actions}>
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
