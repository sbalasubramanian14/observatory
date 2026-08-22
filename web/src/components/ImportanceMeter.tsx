import { formatScore, importanceTier } from "@/lib/format";
import styles from "./ImportanceMeter.module.css";

export function ImportanceMeter({ score }: { score: number }) {
  const tier = importanceTier(score);
  const pct = Math.max(4, Math.min(100, Math.round(score * 100)));

  return (
    <span className={`${styles.wrap} ${styles[tier]}`} title={`Importance score: ${formatScore(score)}/100`}>
      <span className={styles.bar}>
        <span className={styles.fill} style={{ width: `${pct}%` }} />
      </span>
      <span className={styles.value}>{formatScore(score)}</span>
    </span>
  );
}
