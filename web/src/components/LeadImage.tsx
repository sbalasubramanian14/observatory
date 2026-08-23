"use client";

// Renders a story's lead image (spec D0) with an explicit aspect ratio (no
// layout jump while it loads), native lazy loading, and a graceful
// text-only fallback: a null src renders nothing, and an image that fails
// to load at runtime (hotlink protection, a 404, a since-removed asset)
// hides itself the same way rather than showing a broken-image icon. The
// text-only look must be the same whether we never had a URL or the URL
// just didn't pan out.
import { useState } from "react";
import Image from "next/image";
import styles from "./LeadImage.module.css";

export function LeadImage({
  src,
  alt,
  variant = "thumb",
}: {
  src: string | null;
  alt: string;
  variant?: "thumb" | "hero";
}) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) return null;

  return (
    <div className={`${styles.wrap} ${variant === "hero" ? styles.hero : styles.thumb}`}>
      <Image
        src={src}
        alt={alt}
        fill
        unoptimized
        // The hero variant is always the top-of-page image on the story
        // detail page — reliably the LCP element there — so it loads
        // eagerly. Card thumbnails are lazy: most never enter the
        // viewport on a long feed.
        loading={variant === "hero" ? "eager" : "lazy"}
        sizes={variant === "hero" ? "(max-width: 780px) 100vw, 780px" : "104px"}
        style={{ objectFit: "cover" }}
        onError={() => setFailed(true)}
      />
    </div>
  );
}
