import { formatScore, importanceTier } from "@/lib/format";
import styles from "./ImportanceMeter.module.css";

// Ten discrete ticks rather than a continuous fill. A progress bar reads as
// "loading"; a segmented gauge reads as a measurement, which is what this
// actually is -- and it's the one piece of the interface the whole ranking
// pipeline exists to produce, so it should not look like chrome.
//
// The score/tier contract is untouched: same formatScore, same
// importanceTier thresholds, same 0..1 input. Only the rendering changed.
const TICKS = 10;

export function ImportanceMeter({ score }: { score: number }) {
  const tier = importanceTier(score);
  // At least one lit tick, so a real-but-tiny score never reads as "no data".
  const lit = Math.max(1, Math.min(TICKS, Math.round(score * TICKS)));

  return (
    <span
      className={`${styles.wrap} ${styles[tier]}`}
      title={`Importance score: ${formatScore(score)}/100`}
    >
      <span className={styles.ticks} aria-hidden="true">
        {Array.from({ length: TICKS }, (_, i) => (
          <span key={i} className={`${styles.tick} ${i < lit ? styles.tickOn : ""}`} />
        ))}
      </span>
      <span className={styles.value}>{formatScore(score)}</span>
    </span>
  );
}
