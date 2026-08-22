"use client";

import { useTheme, type ThemePreference } from "@/lib/theme";
import styles from "./ThemeToggle.module.css";

const OPTIONS: { value: ThemePreference; label: string; icon: React.ReactNode }[] = [
  {
    value: "light",
    label: "Light theme",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
    ),
  },
  {
    value: "system",
    label: "System theme",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="4" width="18" height="13" rx="2" />
        <path d="M8 21h8M12 17v4" />
      </svg>
    ),
  },
  {
    value: "dark",
    label: "Dark theme",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M20.5 14.5A8.5 8.5 0 119.5 3.5a7 7 0 0011 11z" />
      </svg>
    ),
  },
];

export function ThemeToggle() {
  const { preference, setPreference } = useTheme();

  return (
    <div className={styles.group} role="radiogroup" aria-label="Theme">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          type="button"
          role="radio"
          aria-checked={preference === opt.value}
          title={opt.label}
          aria-label={opt.label}
          className={`${styles.option} ${preference === opt.value ? styles.optionActive : ""}`}
          onClick={() => setPreference(opt.value)}
        >
          <span className={styles.icon}>{opt.icon}</span>
        </button>
      ))}
    </div>
  );
}
