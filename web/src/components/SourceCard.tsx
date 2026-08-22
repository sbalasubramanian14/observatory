import type { SourceHealth } from "@/lib/types";
import { formatDateTime } from "@/lib/format";
import styles from "./SourceCard.module.css";

function status(source: SourceHealth): "healthy" | "degraded" | "failing" | "disabled" {
  if (!source.enabled) return "disabled";
  if (source.consecutive_failures >= 3) return "failing";
  if (source.consecutive_failures > 0) return "degraded";
  return "healthy";
}

const STATUS_LABEL: Record<string, string> = {
  healthy: "Healthy",
  degraded: "Degraded",
  failing: "Failing",
  disabled: "Disabled",
};

export function SourceCard({ source }: { source: SourceHealth }) {
  const s = status(source);

  return (
    <div className={styles.card}>
      <div className={styles.top}>
        <div>
          <span className={styles.id}>{source.id}</span>
          <span className={styles.plugin}>{source.plugin} connector</span>
        </div>
        <span className={`${styles.statusPill} ${styles[s]}`}>
          <span className={styles.dot} />
          {STATUS_LABEL[s]}
        </span>
      </div>
      <div className={styles.stats}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Last run</span>
          <span className={styles.statValue}>{formatDateTime(source.last_run_at)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Cadence</span>
          <span className={styles.statValue}>{source.cadence_minutes}m</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Consecutive failures</span>
          <span className={styles.statValue}>{source.consecutive_failures}</span>
        </div>
      </div>
      {source.last_error && <div className={styles.error}>{source.last_error}</div>}
    </div>
  );
}
