"use client";

import Link from "next/link";
import type { FeedStoryRow } from "@/lib/types";
import { formatAge } from "@/lib/format";
import { CategoryTag } from "./CategoryTag";
import { ImportanceMeter } from "./ImportanceMeter";
import { LeadImage } from "./LeadImage";
import styles from "./StoryListRow.module.css";

/** One row in the /saved or /dismissed list (D2, D3). Deliberately not
 * StoryCard: that component is wired to the feed's personalization signals
 * (save/dismiss buttons nudge the on-device centroid), which don't apply
 * here — this is a plain list with one reversible action per row. */
export function StoryListRow({
  story,
  actionLabel,
  onAction,
}: {
  story: FeedStoryRow;
  actionLabel: string;
  onAction: () => void;
}) {
  return (
    <article className={styles.row}>
      <LeadImage src={story.lead_image_url} alt="" variant="thumb" />
      <div className={styles.body}>
        <div className={styles.metaTop}>
          <CategoryTag category={story.category} />
          <ImportanceMeter score={story.score} />
        </div>
        <Link href={`/story/?id=${story.id}`} className={styles.headline}>
          {story.title}
        </Link>
        <span className={styles.meta}>
          {story.outlet_count} outlet{story.outlet_count === 1 ? "" : "s"} · {formatAge(story.updated_at)}
        </span>
      </div>
      <button type="button" className={styles.actionButton} onClick={onAction}>
        {actionLabel}
      </button>
    </article>
  );
}

/** A saved/dismissed id that no longer resolves to a story in the current
 * bundle window (pruned by retention, spec §4.4). Shown rather than
 * silently hidden — see StoryListPage's docstring for the rationale — with
 * just enough context to explain why and a way to clear it out. */
export function StoryListMissingRow({
  id,
  actionLabel,
  onAction,
}: {
  id: number;
  actionLabel: string;
  onAction: () => void;
}) {
  return (
    <article className={`${styles.row} ${styles.missing}`}>
      <div className={styles.body}>
        <span className={styles.missingText}>
          Story #{id} is no longer available — it has aged out of the published feed&apos;s
          retention window.
        </span>
      </div>
      <button type="button" className={styles.actionButton} onClick={onAction}>
        {actionLabel}
      </button>
    </article>
  );
}
