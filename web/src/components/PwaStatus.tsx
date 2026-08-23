"use client";

// Mounted once in layout.tsx. Owns the two pieces of "honest offline UX"
// (PWA spec, Part 1): a banner that never lets the reader mistake cached
// data for live data, and a toast that surfaces a new app-shell version
// without ever silently reloading out from under a mid-read.
import { useServiceWorker, useOnlineStatus, useLastSync, useServingFromCache } from "@/lib/pwa";
import { formatDateTime } from "@/lib/format";
import styles from "./PwaStatus.module.css";

export function PwaStatus() {
  const online = useOnlineStatus();
  const servingFromCache = useServingFromCache();
  const lastSync = useLastSync();
  const { updateAvailable, applyUpdate } = useServiceWorker();

  // Two independent signals, either one enough to show the banner:
  // navigator.onLine (the browser's own, sometimes-stale link-layer flag)
  // and direct evidence that the last manifest fetch had to fall back to
  // the service worker's cache (see lib/pwa.ts's useServingFromCache doc
  // comment for why the second one is needed at all).
  const showOfflineBanner = !online || servingFromCache;

  return (
    <>
      {showOfflineBanner && (
        <div className={styles.offlineBanner} role="status">
          <span className={styles.offlineDot} aria-hidden="true" />
          {lastSync
            ? `Showing cached data from ${formatDateTime(new Date(lastSync).toISOString())} — you're offline or your connection dropped.`
            : "You're offline — nothing has synced to this device yet."}
        </div>
      )}
      {updateAvailable && (
        <div className={styles.updateToast} role="status">
          <span>A new version of Observatory is ready.</span>
          <button type="button" className={styles.updateButton} onClick={applyUpdate}>
            Refresh
          </button>
        </div>
      )}
    </>
  );
}
