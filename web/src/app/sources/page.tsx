"use client";

import { useEffect, useState } from "react";
import { getManifest, getSources } from "@/lib/bundle";
import type { SourcesFile } from "@/lib/types";
import { SourceCard } from "@/components/SourceCard";
import { formatDateTime } from "@/lib/format";
import styles from "./page.module.css";

export default function SourcesPage() {
  const [data, setData] = useState<SourcesFile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const manifest = await getManifest();
        const sources = await getSources(manifest);
        if (!cancelled) setData(sources);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const healthy = data?.sources.filter((s) => s.enabled && s.consecutive_failures === 0).length ?? 0;
  const degraded =
    data?.sources.filter((s) => s.enabled && s.consecutive_failures > 0 && s.consecutive_failures < 3)
      .length ?? 0;
  const failing = data?.sources.filter((s) => s.enabled && s.consecutive_failures >= 3).length ?? 0;
  const disabled = data?.sources.filter((s) => !s.enabled).length ?? 0;

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Sources Health</h1>
      <p className={styles.subtitle}>
        Silent coverage loss is the failure mode that defeats this product — a broken connector
        stops stories from ever reaching the feed without any error a reader would see. This page
        makes that visible.
      </p>

      {error && <p className={`${styles.state} ${styles.errorState}`}>Couldn&apos;t load source health: {error}</p>}
      {!data && !error && <p className={styles.state}>Loading…</p>}

      {data && (
        <>
          <div className={styles.summary}>
            <div className={styles.summaryStat}>
              <span className={styles.summaryLabel}>Healthy</span>
              <span className={styles.summaryValue}>{healthy}</span>
            </div>
            <div className={styles.summaryStat}>
              <span className={styles.summaryLabel}>Degraded</span>
              <span className={styles.summaryValue}>{degraded}</span>
            </div>
            <div className={styles.summaryStat}>
              <span className={styles.summaryLabel}>Failing</span>
              <span className={styles.summaryValue}>{failing}</span>
            </div>
            <div className={styles.summaryStat}>
              <span className={styles.summaryLabel}>Disabled</span>
              <span className={styles.summaryValue}>{disabled}</span>
            </div>
          </div>
          <p className={styles.generated}>Report generated {formatDateTime(data.generated_at)}</p>
          <div className={styles.grid}>
            {data.sources.map((s) => (
              <SourceCard key={s.id} source={s} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
