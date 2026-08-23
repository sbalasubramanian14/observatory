// Per-source identity chip (Reference A: "avatar/logo + name" at the top
// of each card). The bundle has no per-source logo asset (spec §4.2 keeps
// the bundle to references/text, never binary brand assets we'd have to
// host and keep in sync), so this renders deterministic initials on one of
// four semantic-token colour variants instead — see lib/format.ts's
// avatarVariant/initialsOf.
import { avatarVariant, initialsOf } from "@/lib/format";
import styles from "./SourceAvatar.module.css";

const VARIANT_CLASS = { 1: styles.v1, 2: styles.v2, 3: styles.v3, 4: styles.v4 } as const;

export function SourceAvatar({ sourceId, size = 32 }: { sourceId: string; size?: number }) {
  return (
    <span
      className={`${styles.avatar} ${VARIANT_CLASS[avatarVariant(sourceId)]}`}
      style={{ "--avatar-size": `${size}px` } as React.CSSProperties}
      aria-hidden="true"
    >
      {initialsOf(sourceId)}
    </span>
  );
}
