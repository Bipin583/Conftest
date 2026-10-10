// Response shapes for the ConfTest API, hand-typed against the FastAPI schemas
// (src/conftest/api). Kept narrow to what the dashboard pages actually read.

export interface Provenance {
  real_labels: boolean;
  detail: string;
}

export interface HeadlineMetric {
  label: string;
  value: string | null;
  note: string;
  source: string;
  measured: boolean;
}

export interface HeadlineResponse {
  provenance: Provenance;
  metrics: HeadlineMetric[];
}

export interface BaselineRow {
  strategy: string | null;
  test_reduction_pct: number | null;
  time_reduction_pct: number | null;
  failure_recall_pct: number | null;
  missed_failure_pct: number | null;
  abstention_rate_pct: number | null;
  escaped_commits: number | null;
}

export interface BaselineResponse {
  rows: BaselineRow[];
  provenance: Provenance;
}

export interface ReliabilityBin {
  bin_idx: number;
  bin_range: [number, number];
  avg_confidence?: number;
  empirical_accuracy?: number;
  sample_count?: number;
}

export interface CalibrationMetric {
  method: string | null;
  ece: number;
  mce: number;
  brier_score: number;
  ece_reduction_pct: number | null;
}

export interface CalibrationResponse {
  best_method: string;
  uncalibrated: CalibrationMetric;
  calibrated: CalibrationMetric | null;
  temperature: number | null;
  metrics_split: string;
  selection_basis: string | null;
  selection_reason: string | null;
  selection_split: string | null;
  resampling_unit: string | null;
  reliability_diagram_bins: ReliabilityBin[];
}

export interface UncertaintyResponse {
  analysis: Record<string, unknown>;
  policy: Record<string, unknown>;
  ensemble: Record<string, unknown>;
}

export interface ExplanationsResponse {
  labels_measured?: boolean;
  model_file?: string;
  dataset_file?: string;
  global_shap_importance?: { feature: string; mean_abs_shap: number }[];
  sample_developer_cards?: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface RankedTest {
  test_id: string;
  raw_score: number;
  calibrated_confidence: number;
  epistemic_uncertainty: number;
  is_selected: boolean;
  reasons: string[];
}

export interface SelectResponse {
  commit_sha: string;
  decision_mode: string;
  abstained: boolean;
  selected_count: number;
  total_count: number;
  test_reduction_pct: number;
  top_confidence: number;
  epistemic_uncertainty: number;
  reasons: string[];
  selected_test_ids: string[];
  ranked_tests: RankedTest[];
}

export interface SelectRequest {
  repository_name: string;
  commit_sha: string;
  commit_message: string;
  changed_files: { file_path: string; change_type: string }[];
  budget_ratio: number;
}
