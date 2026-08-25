"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./Header.module.css";
import { ThemeToggle } from "./ThemeToggle";
import { PersonalizationToggle } from "./PersonalizationToggle";
import { useSavedCount } from "@/lib/personalization";

export function Header() {
  const pathname = usePathname();
  const savedCount = useSavedCount();

  return (
    <header className={styles.header}>
      <Link href="/" className={styles.brand}>
        <span className={styles.mark}>Observatory</span>
        <span className={styles.tagline}>AI news, importance-ranked</span>
      </Link>
      <nav className={styles.nav}>
        <Link
          href="/"
          className={`${styles.navLink} ${pathname === "/" ? styles.navLinkActive : ""}`}
        >
          Feed
        </Link>
        <Link
          href="/top/"
          className={`${styles.navLink} ${pathname?.startsWith("/top") ? styles.navLinkActive : ""}`}
        >
          Top 50
        </Link>
        <Link
          href="/sources/"
          className={`${styles.navLink} ${pathname?.startsWith("/sources") ? styles.navLinkActive : ""}`}
        >
          Sources
        </Link>
        <Link
          href="/saved/"
          className={`${styles.navLink} ${pathname?.startsWith("/saved") ? styles.navLinkActive : ""}`}
        >
          Saved
          {savedCount > 0 && <span className={styles.badge}>{savedCount}</span>}
        </Link>
        <div className={styles.controls}>
          <PersonalizationToggle />
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
