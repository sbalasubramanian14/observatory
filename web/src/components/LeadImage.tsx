"use client";

// Renders a story's lead image (spec D0) with an explicit aspect ratio (no
// layout jump while it loads), native lazy loading, and a graceful
// text-only fallback: a null src renders nothing, and an image that fails
// to load at runtime (hotlink protection, a 404, a since-removed asset)
// hides itself the same way rather than showing a broken-image icon. The
// text-only look must be the same whether we never had a URL or the URL
// just didn't pan out.
//
// Sizing: `next.config.ts` sets `images.unoptimized: true` (required for
// `output: export` — there is no server to run Next's image API route
// against), so without help the browser downloads whatever full-resolution
// file the publisher happens to host — measured at 600KB-1MB+ for several
// real stories in the bundle, which alone pushed mobile LCP past 13s under
// throttled network conditions. `resizedSrc` below routes the request
// through wsrv.nl (a free, widely-used public image-resizing proxy — see
// https://wsrv.nl) asking for exactly the pixel size this variant actually
// displays, which cuts most of those images to 20-30KB. It fails open: if
// the proxy is ever unreachable, `onError` below falls back to the
// text-only state exactly as it would for any other broken image, never a
// broken-image icon.
import { useState } from "react";
import Image from "next/image";
import styles from "./LeadImage.module.css";

const VARIANT_PX: Record<"thumb" | "hero" | "card", { w: number; h: number }> = {
  thumb: { w: 160, h: 160 },
  card: { w: 700, h: 440 },
  hero: { w: 900, h: 506 },
};

function resizedSrc(src: string, variant: "thumb" | "hero" | "card"): string {
  const { w, h } = VARIANT_PX[variant];
  const params = new URLSearchParams({
    url: src.replace(/^https?:\/\//, ""),
    w: String(w),
    h: String(h),
    fit: "cover",
    output: "webp",
    q: "76",
  });
  return `https://wsrv.nl/?${params.toString()}`;
}

export function LeadImage({
  src,
  alt,
  variant = "thumb",
  priority = false,
}: {
  src: string | null;
  alt: string;
  variant?: "thumb" | "hero" | "card";
  /** True for the first couple of above-the-fold cards in the feed list —
   * everything else stays lazy. This can't make the image *discoverable*
   * before the bundle fetch resolves (the URL genuinely doesn't exist in
   * the initial HTML — the bundle is fetched at runtime by design, spec
   * §4.5), but it does mean the browser starts the request the instant
   * the <img> is inserted rather than waiting on a lazy-load viewport
   * check, which is the part actually in this component's control. */
  priority?: boolean;
}) {
  const [failed, setFailed] = useState(false);

  if (!src || failed) return null;

  const eager = variant === "hero" || priority;
  const variantClass = variant === "hero" ? styles.hero : variant === "card" ? styles.card : styles.thumb;

  return (
    <div className={`${styles.wrap} ${variantClass}`}>
      <Image
        src={resizedSrc(src, variant)}
        alt={alt}
        fill
        unoptimized
        // The hero variant is always the top-of-page image on the story
        // detail page — reliably the LCP element there — so it loads
        // eagerly, as do the first couple of feed cards (see `priority`
        // above). Everything else stays lazy: most never enter the
        // viewport on a long feed.
        loading={eager ? "eager" : "lazy"}
        fetchPriority={eager ? "high" : undefined}
        sizes={variant === "hero" || variant === "card" ? "(max-width: 780px) 100vw, 780px" : "104px"}
        style={{ objectFit: "cover" }}
        onError={() => setFailed(true)}
      />
    </div>
  );
}
