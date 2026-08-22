"use client";

import { usePersonalization } from "@/lib/personalization";
import styles from "./PersonalizationToggle.module.css";

export function PersonalizationToggle() {
  const { enabled, setEnabled } = usePersonalization();

  return (
    <div className={styles.wrap}>
      <span className={styles.label}>For you</span>
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        aria-label="Personalized ranking"
        title={enabled ? "Personalized ranking is on" : "Personalized ranking is off — showing published order"}
        className={`${styles.switch} ${enabled ? styles.switchOn : ""}`}
        onClick={() => setEnabled(!enabled)}
      >
        <span className={styles.knob} />
      </button>
    </div>
  );
}
