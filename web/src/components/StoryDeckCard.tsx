"use client";

// One full-viewport slide in the mobile "cards" deck (StoryDeck.tsx).
// Vertical paging between slides is native CSS scroll-snap (see the .deck
// container); this component only handles the *horizontal* swipe-to-save
// (right) / swipe-to-dismiss (left) gesture, via pointer events with
// `touch-action: pan-y` on the card so the browser keeps handling vertical
// scroll natively while JS only ever sees horizontal intent.
import { useRef, useState } from "react";
import Link from "next/link";
import type { RankedStory } from "@/lib/personalization";
import { markSaved, markDismissed, recordSignal, getSavedIds } from "@/lib/personalization";
import { formatAge, formatSourceName } from "@/lib/format";
import { CategoryTag } from "./CategoryTag";
import { ImportanceMeter } from "./ImportanceMeter";
import { LeadImage } from "./LeadImage";
import { SourceAvatar } from "./SourceAvatar";
import styles from "./StoryDeck.module.css";

const COMMIT_THRESHOLD = 88;

export function StoryDeckCard({
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
  const { story } = ranked;
  const [saved, setSaved] = useState(() => getSavedIds().has(story.id));
  const [drag, setDrag] = useState(0);
  const [dragging, setDragging] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; y: number; active: boolean; dx: number } | null>(null);

  function handleSave() {
    if (saved) return;
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

  function onPointerDown(e: React.PointerEvent) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    start.current = { x: e.clientX, y: e.clientY, active: false, dx: 0 };
  }

  function onPointerMove(e: React.PointerEvent) {
    if (!start.current) return;
    const dx = e.clientX - start.current.x;
    const dy = e.clientY - start.current.y;
    if (!start.current.active) {
      if (Math.abs(dx) < 12 && Math.abs(dy) < 12) return;
      // Vertical intent — let the ancestor scroller (pan-y) handle it.
      if (Math.abs(dy) > Math.abs(dx)) {
        start.current = null;
        return;
      }
      start.current.active = true;
      setDragging(true);
    }
    // Mirrored into the ref (not just React state) so onPointerUp can read
    // the *current* drag distance synchronously — a burst of pointermove
    // events followed immediately by pointerup can outrun a React state
    // update/re-render, which would otherwise leave onPointerUp holding a
    // stale `drag` from before this move. The ref mutates immediately, no
    // render needed, so it can never be behind.
    start.current.dx = dx;
    setDrag(dx);
  }

  function onPointerUp() {
    if (!start.current?.active) {
      start.current = null;
      return;
    }
    const finalDx = start.current.dx;
    start.current = null;
    setDragging(false);
    if (finalDx > COMMIT_THRESHOLD) handleSave();
    else if (finalDx < -COMMIT_THRESHOLD) handleDismiss();
    setDrag(0);
  }

  const primarySource = story.source_ids[0] ?? "unknown";
  const pull = Math.max(-1, Math.min(1, drag / 160));

  return (
    <div style={{ position: "relative", flex: 1, display: "flex", minHeight: 0 }}>
      <div
        className={`${styles.affordance} ${styles.affordanceDismiss}`}
        style={{ opacity: pull < 0 ? Math.min(1, -pull * 1.4) : 0 }}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </div>
      <div
        className={`${styles.affordance} ${styles.affordanceSave}`}
        style={{ opacity: pull > 0 ? Math.min(1, pull * 1.4) : 0 }}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2">
          <path d="M6 4h12v16l-6-4-6 4V4z" />
        </svg>
      </div>

      <article
        ref={cardRef}
        className={styles.card}
        style={{
          transform: `translateX(${drag}px) rotate(${pull * 4}deg)`,
          transition: dragging ? "none" : "transform 0.25s ease",
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div className={styles.identity}>
          <SourceAvatar sourceId={primarySource} />
          <div className={styles.identityText}>
            <span className={styles.sourceName}>
              {formatSourceName(primarySource)}
              {story.outlet_count > 1 ? ` +${story.outlet_count - 1}` : ""}
            </span>
            <span className={styles.timestamp}>{formatAge(story.updated_at)}</span>
          </div>
        </div>

        {story.lead_image_url && (
          <div className={styles.imageWrap}>
            <LeadImage src={story.lead_image_url} alt="" variant="card" />
          </div>
        )}

        <div className={styles.body}>
          <div className={styles.metaTop}>
            <CategoryTag category={story.category} />
            <ImportanceMeter score={story.score} />
          </div>
          <Link href={`/story/?id=${story.id}`} className={styles.headline}>
            {story.title}
          </Link>
          <p className={`${styles.summary} ${!story.summary ? styles.summaryEmpty : ""}`}>
            {story.summary ?? "No summary yet — this story hasn't been through Tier 1 processing."}
          </p>
          {!story.lead_image_url && <div className={styles.textOnlyPad} aria-hidden="true" />}
        </div>

        <div className={styles.footer}>
          <div className={styles.metaBottom}>
            <span className={styles.metaItem}>{story.outlet_count} outlets</span>
            <span className={styles.metaItem}>{story.item_count} items</span>
          </div>
          <div className={styles.actionRow}>
            <button
              type="button"
              className={`${styles.iconButton} ${saved ? styles.iconButtonActive : ""}`}
              onClick={handleSave}
              aria-label="Save story"
              aria-pressed={saved}
            >
              <svg viewBox="0 0 24 24" width="16" height="16" fill={saved ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2">
                <path d="M6 4h12v16l-6-4-6 4V4z" />
              </svg>
            </button>
            <button type="button" className={styles.iconButton} onClick={handleDismiss} aria-label="Dismiss story">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          </div>
        </div>
      </article>
    </div>
  );
}
