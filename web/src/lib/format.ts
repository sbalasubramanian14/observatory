export function formatAge(iso: string, now: Date = new Date()): string {
  const then = new Date(iso).getTime();
  const diffMs = now.getTime() - then;
  if (Number.isNaN(diffMs)) return "";
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  const years = Math.floor(months / 12);
  return `${years}y ago`;
}

export function formatScore(score: number): string {
  return Math.round(score * 100).toString();
}

/** Importance tier used to pick the semantic colour token for a score. */
export function importanceTier(score: number): "high" | "medium" | "low" {
  if (score >= 0.55) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

export function formatCategory(category: string | null): string {
  if (!category) return "Uncategorized";
  return category
    .split(/[-_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "never";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "unknown";
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Display name for a source id (spec has no separate "display name"
 * field for sources — the id itself, title-cased, is what StoryCard's
 * per-source identity row shows). Reuses the same word-splitting as
 * formatCategory since ids follow the same kebab/snake convention
 * ("github-releases" -> "Github Releases"). */
export function formatSourceName(id: string): string {
  return formatCategory(id);
}

/** A small, deterministic (never random) hash used only to pick a stable
 * avatar colour variant and initials for a source id — same id always
 * renders the same chip, with no per-source colour table to maintain. */
function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

/** One of 4 semantic-token avatar variants (--color-avatar-1..4 in
 * globals.css) — never a literal colour, so the chip re-themes for free. */
export function avatarVariant(id: string): 1 | 2 | 3 | 4 {
  return ((hashString(id) % 4) + 1) as 1 | 2 | 3 | 4;
}

/** Up to 2 initials for a source id's avatar chip: one per hyphen/
 * underscore-separated word (max 2), or the first two characters of a
 * single-word id. */
export function initialsOf(id: string): string {
  const words = id.split(/[-_\s]+/).filter(Boolean);
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return "?";
}

export function hostnameOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
