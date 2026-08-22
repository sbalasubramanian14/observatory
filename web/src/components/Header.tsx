"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./Header.module.css";
import { ThemeToggle } from "./ThemeToggle";
import { PersonalizationToggle } from "./PersonalizationToggle";

export function Header() {
  const pathname = usePathname();

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
          href="/sources/"
          className={`${styles.navLink} ${pathname?.startsWith("/sources") ? styles.navLinkActive : ""}`}
        >
          Sources
        </Link>
        <div className={styles.controls}>
          <PersonalizationToggle />
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
