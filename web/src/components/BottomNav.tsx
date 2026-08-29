"use client";

// Floating bottom nav bar (Reference A's "clean, tactile, unmistakably a
// phone app" cue) — mobile only, see .nav's media query. Desktop keeps
// Header's top nav; this isn't a second copy of it, it's the mobile
// wayfinding surface, so it carries the "More" sheet for theme/
// personalization instead of duplicating Header's controls row.
//
// Top 50 is here because Header.module.css hides .nav below 768px, so
// this bar was the ONLY nav on mobile -- and it used to omit /top
// entirely. The route, its page, and its data were all fine; the
// installed PWA (portrait, standalone) simply had no way to reach the
// one view the ranking stage exists to produce.
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSavedCount } from "@/lib/personalization";
import { ThemeToggle } from "./ThemeToggle";
import { PersonalizationToggle } from "./PersonalizationToggle";
import styles from "./BottomNav.module.css";

function FeedIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4 6h16M4 12h16M4 18h10" />
    </svg>
  );
}
function TopIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M5 20v-7M12 20V6M19 20v-9" />
    </svg>
  );
}
function SourcesIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8" />
    </svg>
  );
}
function SavedIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M6 4h12v16l-6-4-6 4V4z" />
    </svg>
  );
}
function MoreIcon() {
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v4M12 15.5h.01" />
    </svg>
  );
}

export function BottomNav() {
  const pathname = usePathname();
  const savedCount = useSavedCount();
  const [sheetOpen, setSheetOpen] = useState(false);

  return (
    <>
      {sheetOpen && (
        <>
          <div className={styles.sheetScrim} onClick={() => setSheetOpen(false)} />
          <div className={styles.sheet} role="dialog" aria-label="Display settings">
            <div className={styles.sheetRow}>
              <span className={styles.sheetLabel}>Theme</span>
              <ThemeToggle />
            </div>
            <div className={styles.sheetRow}>
              <span className={styles.sheetLabel}>Personalized ranking</span>
              <PersonalizationToggle />
            </div>
          </div>
        </>
      )}
      <nav className={styles.nav} aria-label="Primary">
        <Link
          href="/"
          className={`${styles.item} ${pathname === "/" ? styles.itemActive : ""}`}
          aria-label="Feed"
          onClick={() => setSheetOpen(false)}
        >
          <FeedIcon />
          <span className={styles.itemLabel}>Feed</span>
        </Link>
        <Link
          href="/top/"
          className={`${styles.item} ${pathname?.startsWith("/top") ? styles.itemActive : ""}`}
          aria-label="Top 50"
          onClick={() => setSheetOpen(false)}
        >
          <TopIcon />
          <span className={styles.itemLabel}>Top 50</span>
        </Link>
        <Link
          href="/sources/"
          className={`${styles.item} ${pathname?.startsWith("/sources") ? styles.itemActive : ""}`}
          aria-label="Sources"
          onClick={() => setSheetOpen(false)}
        >
          <SourcesIcon />
          <span className={styles.itemLabel}>Sources</span>
        </Link>
        <Link
          href="/saved/"
          className={`${styles.item} ${pathname?.startsWith("/saved") ? styles.itemActive : ""}`}
          aria-label="Saved"
          onClick={() => setSheetOpen(false)}
        >
          {savedCount > 0 && <span className={styles.badge}>{savedCount}</span>}
          <SavedIcon />
          <span className={styles.itemLabel}>Saved</span>
        </Link>
        <button
          type="button"
          className={`${styles.item} ${sheetOpen ? styles.itemActive : ""}`}
          aria-label="Display settings"
          aria-expanded={sheetOpen}
          onClick={() => setSheetOpen((v) => !v)}
        >
          <MoreIcon />
          <span className={styles.itemLabel}>More</span>
        </button>
      </nav>
    </>
  );
}
