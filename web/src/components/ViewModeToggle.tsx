"use client";

import { useViewMode } from "@/lib/viewMode";
import styles from "./ViewModeToggle.module.css";

export function ViewModeToggle() {
  const [mode, setMode] = useViewMode();

  return (
    <div className={styles.wrap} role="radiogroup" aria-label="Feed view">
      <button
        type="button"
        role="radio"
        aria-checked={mode === "list"}
        title="List view"
        aria-label="List view"
        className={`${styles.option} ${mode === "list" ? styles.optionActive : ""}`}
        onClick={() => setMode("list")}
      >
        <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M4 6h16M4 12h16M4 18h10" />
        </svg>
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={mode === "cards"}
        title="One story at a time"
        aria-label="Card view"
        className={`${styles.option} ${mode === "cards" ? styles.optionActive : ""}`}
        onClick={() => setMode("cards")}
      >
        <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="5" y="3" width="14" height="18" rx="3" />
        </svg>
      </button>
    </div>
  );
}
