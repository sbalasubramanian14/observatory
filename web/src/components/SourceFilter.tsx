"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { setMany, toggleId, useSourceFilter } from "@/lib/sourceFilter";
import { formatCategory } from "@/lib/format";
import type { FeedStoryRow, SourceHealth } from "@/lib/types";
import styles from "./SourceFilter.module.css";

// Spec §2's four coverage territories, in the order they're documented --
// anything else (a source synced before territory existed) falls back to
// an "Other" bucket appended at the end.
const TERRITORY_ORDER = ["research", "industry", "policy", "infrastructure"];

interface TerritoryGroup {
  territory: string;
  sourceIds: string[];
  sources: SourceHealth[];
}

function groupByTerritory(sources: SourceHealth[]): TerritoryGroup[] {
  const byTerritory = new Map<string, SourceHealth[]>();
  for (const s of sources) {
    const key = s.territory ?? "other";
    if (!byTerritory.has(key)) byTerritory.set(key, []);
    byTerritory.get(key)!.push(s);
  }
  const known = TERRITORY_ORDER.filter((t) => byTerritory.has(t));
  const rest = Array.from(byTerritory.keys())
    .filter((t) => !TERRITORY_ORDER.includes(t))
    .sort();
  return [...known, ...rest].map((territory) => {
    const group = [...byTerritory.get(territory)!].sort((a, b) => a.id.localeCompare(b.id));
    return { territory, sources: group, sourceIds: group.map((s) => s.id) };
  });
}

export function SourceFilter({
  sources,
  stories,
}: {
  sources: SourceHealth[];
  stories: FeedStoryRow[];
}) {
  const [selected, setSelected] = useSourceFilter();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  const groups = useMemo(() => groupByTerritory(sources), [sources]);

  const sourceCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const s of sources) counts.set(s.id, 0);
    for (const story of stories) {
      for (const id of story.source_ids) {
        counts.set(id, (counts.get(id) ?? 0) + 1);
      }
    }
    return counts;
  }, [sources, stories]);

  const territoryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const group of groups) {
      const idSet = new Set(group.sourceIds);
      let n = 0;
      for (const story of stories) {
        if (story.source_ids.some((id) => idSet.has(id))) n++;
      }
      counts.set(group.territory, n);
    }
    return counts;
  }, [groups, stories]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: PointerEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  function territoryState(group: TerritoryGroup): "all" | "none" | "some" {
    const n = group.sourceIds.filter((id) => selected.has(id)).length;
    if (n === 0) return "none";
    if (n === group.sourceIds.length) return "all";
    return "some";
  }

  return (
    <div className={styles.wrap} ref={rootRef}>
      <button
        type="button"
        className={`${styles.trigger} ${selected.size > 0 ? styles.triggerActive : ""}`}
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen((v) => !v)}
      >
        Sources
        {selected.size > 0 && <span className={styles.count}>{selected.size}</span>}
      </button>

      {open && (
        <>
          <div className={styles.scrim} onClick={() => setOpen(false)} />
          <div className={styles.panel} role="dialog" aria-label="Filter by source or territory">
            <div className={styles.panelHeader}>
              <span className={styles.panelTitle}>Filter sources</span>
              <button type="button" className={styles.closeButton} onClick={() => setOpen(false)}>
                Done
              </button>
            </div>

            <div className={styles.panelBody}>
              {groups.map((group) => {
                const state = territoryState(group);
                return (
                  <fieldset key={group.territory} className={styles.group}>
                    <label className={styles.territoryRow}>
                      <input
                        type="checkbox"
                        checked={state === "all"}
                        ref={(el) => {
                          if (el) el.indeterminate = state === "some";
                        }}
                        onChange={() => setSelected(setMany(selected, group.sourceIds, state !== "all"))}
                      />
                      <span className={styles.territoryLabel}>{formatCategory(group.territory)}</span>
                      <span className={styles.rowCount}>{territoryCounts.get(group.territory) ?? 0}</span>
                    </label>
                    <div className={styles.sourceList}>
                      {group.sources.map((s) => (
                        <label key={s.id} className={styles.sourceRow}>
                          <input
                            type="checkbox"
                            checked={selected.has(s.id)}
                            onChange={() => setSelected(toggleId(selected, s.id))}
                          />
                          <span className={styles.sourceId}>{s.id}</span>
                          <span className={styles.rowCount}>{sourceCounts.get(s.id) ?? 0}</span>
                        </label>
                      ))}
                    </div>
                  </fieldset>
                );
              })}
            </div>

            <div className={styles.panelFooter}>
              <button
                type="button"
                className={styles.clearButton}
                disabled={selected.size === 0}
                onClick={() => setSelected(new Set())}
              >
                Clear all
              </button>
              <span className={styles.footerSummary}>
                {selected.size === 0 ? "Showing all sources" : `${selected.size} selected`}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
