"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { getManifest, getAllStories, getStoryDetail, getEmbeddingsIndex } from "@/lib/bundle";
import { markRead, recordSignal } from "@/lib/personalization";
import type { StoryDetail } from "@/lib/types";
import { CategoryTag } from "@/components/CategoryTag";
import { ImportanceMeter } from "@/components/ImportanceMeter";
import { formatAge, formatDateTime, hostnameOf, formatScore } from "@/lib/format";
import styles from "./page.module.css";

const BREAKDOWN_LABELS: Record<string, string> = {
  authority: "Source authority",
  entity: "Entity weight",
  novelty: "Novelty",
  velocity: "Cross-source velocity",
};

function StoryDetailInner() {
  const params = useSearchParams();
  const idParam = params.get("id");
  const [detail, setDetail] = useState<StoryDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!idParam) return;
    const id = Number(idParam);
    let cancelled = false;

    (async () => {
      try {
        const manifest = await getManifest();
        const [allStories, embeddings] = await Promise.all([
          getAllStories(manifest),
          getEmbeddingsIndex(manifest),
        ]);
        const row = allStories.find((s) => s.id === id);
        if (!row) {
          if (!cancelled) setNotFound(true);
          return;
        }
        const d = await getStoryDetail(row.detail_path);
        if (cancelled) return;
        setDetail(d);
        markRead(id);
        const embedding = embeddings.get(id);
        if (embedding) recordSignal("open", embedding, manifest.embedding_model_id);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [idParam]);

  return (
    <div className={styles.container}>
      <Link href="/" className={styles.back}>
        ← Back to feed
      </Link>

      {!idParam && <p className={`${styles.state} ${styles.errorState}`}>No story id given.</p>}
      {notFound && <p className={`${styles.state} ${styles.errorState}`}>Story not found in the current bundle.</p>}
      {error && <p className={`${styles.state} ${styles.errorState}`}>Couldn&apos;t load this story: {error}</p>}
      {!detail && !notFound && !error && idParam && <p className={styles.state}>Loading…</p>}

      {detail && (
        <>
          <div className={styles.metaTop}>
            <CategoryTag category={detail.category} />
            <ImportanceMeter score={detail.score} />
          </div>

          <h1 className={styles.title}>{detail.title}</h1>

          <p className={`${styles.summary} ${!detail.summary ? styles.summaryEmpty : ""}`}>
            {detail.summary ?? "No summary yet — this story hasn't been through Tier 1 processing."}
          </p>

          <div className={styles.metaBar}>
            <div className={styles.metaStat}>
              <span className={styles.metaStatLabel}>Outlets</span>
              <span className={styles.metaStatValue}>{detail.outlet_count}</span>
            </div>
            <div className={styles.metaStat}>
              <span className={styles.metaStatLabel}>Items</span>
              <span className={styles.metaStatValue}>{detail.item_count}</span>
            </div>
            <div className={styles.metaStat}>
              <span className={styles.metaStatLabel}>First seen</span>
              <span className={styles.metaStatValue}>{formatAge(detail.first_seen)}</span>
            </div>
            <div className={styles.metaStat}>
              <span className={styles.metaStatLabel}>Updated</span>
              <span className={styles.metaStatValue}>{formatDateTime(detail.updated_at)}</span>
            </div>
          </div>

          {detail.analysis && (
            <div className={styles.section}>
              <p className={styles.sectionTitle}>
                Tier 2 analysis{detail.analysis_provider ? ` · ${detail.analysis_provider}` : ""}
              </p>
              <div className={styles.analysis}>{detail.analysis}</div>
            </div>
          )}

          <div className={styles.section}>
            <p className={styles.sectionTitle}>Score breakdown</p>
            <div className={styles.breakdown}>
              {Object.entries(detail.score_breakdown).map(([key, value]) => (
                <div className={styles.breakdownItem} key={key}>
                  <span className={styles.breakdownLabel}>{BREAKDOWN_LABELS[key] ?? key}</span>
                  <span className={styles.breakdownBar}>
                    <span
                      className={styles.breakdownFill}
                      style={{ width: `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%` }}
                    />
                  </span>
                  <span className={styles.breakdownValue}>{formatScore(value)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.section}>
            <p className={styles.sectionTitle}>
              Evidence — {detail.evidence.length} contributing article
              {detail.evidence.length === 1 ? "" : "s"}
            </p>
            <ul className={styles.evidenceList}>
              {detail.evidence.map((ev) => (
                <li key={ev.id}>
                  <a
                    className={styles.evidenceItem}
                    href={ev.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span className={styles.evidenceTitle}>{ev.title}</span>
                    <span className={styles.evidenceMeta}>
                      <span className={styles.evidenceSource}>{ev.source_id}</span>
                      <span>{hostnameOf(ev.url)}</span>
                      <span>{formatAge(ev.published_at)}</span>
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

export default function StoryDetailPage() {
  return (
    <Suspense fallback={<div className={styles.container}><p className={styles.state}>Loading…</p></div>}>
      <StoryDetailInner />
    </Suspense>
  );
}
