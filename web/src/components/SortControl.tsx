"use client";

import { SORT_OPTIONS, useSortMode } from "@/lib/sort";
import styles from "./SortControl.module.css";

export function SortControl() {
  const [mode, setMode] = useSortMode();

  return (
    <div className={styles.wrap}>
      <label className={styles.label} htmlFor="sort-select">
        Sort
      </label>
      <select
        id="sort-select"
        className={styles.select}
        value={mode}
        onChange={(e) => setMode(e.target.value as typeof mode)}
      >
        {SORT_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
