import { formatCategory } from "@/lib/format";
import styles from "./CategoryTag.module.css";

export function CategoryTag({ category }: { category: string | null }) {
  const variant = category === "research" ? styles.research : styles.default;
  return <span className={`${styles.tag} ${variant}`}>{formatCategory(category)}</span>;
}
