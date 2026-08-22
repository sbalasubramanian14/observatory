// TypeScript types for the observatory bundle schema.
// Mirrors the real, on-disk shape at F:/projects/research/public
// (see docs/superpowers/specs/2026-08-22-ai-news-feed-design.md §4.1).
// Do not "improve" these to look nicer than the source data — most stories
// in the current bundle have `category`, `summary`, `kind`, and `analysis`
// as null (Tier 1/2 haven't run on them yet), and the UI must render that
// gracefully rather than assume enrichment always happened.

export interface ManifestPage {
  count: number;
  hash: string;
  page: number;
  path: string;
}

export interface Manifest {
  embedding_dimensions: number;
  embedding_model_id: string;
  embeddings_hash: string;
  /** embeddings_index[i] is the story id whose 384-float vector sits at
   * slice i in the embeddings .bin file. */
  embeddings_index: number[];
  embeddings_path: string;
  generated_at: string;
  pages: ManifestPage[];
  retention_days: number;
  sources_path: string;
  story_count: number;
  version: number;
}

export interface FeedStoryRow {
  category: string | null;
  detail_hash: string;
  detail_path: string;
  id: number;
  item_count: number;
  kind: string | null;
  outlet_count: number;
  score: number;
  summary: string | null;
  title: string;
  updated_at: string;
}

export interface FeedPage {
  page: number;
  page_count: number;
  stories: FeedStoryRow[];
}

export interface Evidence {
  id: number;
  published_at: string;
  source_id: string;
  title: string;
  url: string;
}

export interface ScoreBreakdown {
  authority: number;
  entity: number;
  novelty: number;
  velocity: number;
}

export interface StoryDetail {
  analysis: string | null;
  analysis_provider: string | null;
  category: string | null;
  evidence: Evidence[];
  first_seen: string;
  id: number;
  item_count: number;
  kind: string | null;
  outlet_count: number;
  score: number;
  score_breakdown: ScoreBreakdown;
  summary: string | null;
  title: string;
  updated_at: string;
}

export interface SourceHealth {
  cadence_minutes: number;
  consecutive_failures: number;
  enabled: boolean;
  id: string;
  last_error: string | null;
  last_run_at: string | null;
  plugin: string;
}

export interface SourcesFile {
  generated_at: string;
  sources: SourceHealth[];
}
