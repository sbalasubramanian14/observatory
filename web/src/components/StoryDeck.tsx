"use client";

// Full-viewport, one-story-at-a-time deck for the mobile "cards" view mode
// (lib/viewMode.ts). Vertical paging is CSS scroll-snap — native momentum
// and snapping, so there is no JS scroll physics to jank. The deck's own
// height is measured to exactly fill the space below the sticky Header,
// which is what keeps the header pinned without the outer page ever
// scrolling (see StoryDeck.module.css's top comment).
import { useEffect, useRef, useState } from "react";
import type { RankedStory } from "@/lib/personalization";
import { StoryDeckCard } from "./StoryDeckCard";
import styles from "./StoryDeck.module.css";

export function StoryDeck({
  stories,
  embeddings,
  embeddingModelId,
  onDismiss,
  onSignal,
  onNearEnd,
}: {
  stories: RankedStory[];
  embeddings: Map<number, Float32Array>;
  embeddingModelId: string;
  onDismiss: (id: number) => void;
  onSignal?: () => void;
  /** Called (at most once per growth of `stories`) once the reader is
   * within a few cards of the end, so the caller can lift its page-size
   * cap — a manual "Load more" tap would break the swipe flow this view
   * exists for. */
  onNearEnd?: () => void;
}) {
  const deckRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | null>(null);
  const firedFor = useRef(0);

  useEffect(() => {
    const el = deckRef.current;
    if (!el) return;
    function measure() {
      const top = el!.getBoundingClientRect().top;
      setHeight(window.innerHeight - top);
    }
    measure();
    window.addEventListener("resize", measure);
    window.addEventListener("orientationchange", measure);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("orientationchange", measure);
    };
  }, []);

  useEffect(() => {
    const el = deckRef.current;
    if (!el || !onNearEnd) return;
    function onScroll() {
      if (!el || el.clientHeight === 0) return;
      const index = Math.round(el.scrollTop / el.clientHeight);
      if (index >= stories.length - 3 && firedFor.current !== stories.length) {
        firedFor.current = stories.length;
        onNearEnd?.();
      }
    }
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [stories.length, onNearEnd]);

  return (
    <div
      ref={deckRef}
      className={styles.deck}
      style={{ height: height ? `${height}px` : "80vh" }}
    >
      {stories.map((r, i) => (
        <section key={r.story.id} className={styles.slide} aria-label={`Story ${i + 1} of ${stories.length}`}>
          <StoryDeckCard
            ranked={r}
            embedding={embeddings.get(r.story.id)}
            embeddingModelId={embeddingModelId}
            onDismiss={onDismiss}
            onSignal={onSignal}
          />
          <p className={styles.swipeHint} aria-hidden="true">
            {i === 0 ? "Swipe up for next · right to save · left to dismiss" : `${i + 1} of ${stories.length}`}
          </p>
        </section>
      ))}
    </div>
  );
}
