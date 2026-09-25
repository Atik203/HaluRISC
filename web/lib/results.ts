import fs from "fs";
import path from "path";

/**
 * Server-only data layer: typed readers for every B1-B5 result artifact.
 * Missing files return null (never throw). CSV readers handle quoted fields.
 */

const RESULTS_DIR = path.resolve(process.cwd(), "..", "artifacts", "results");
const MODELS_DIR = path.resolve(process.cwd(), "..", "artifacts", "models");

export function readJson<T>(name: string): T | null {
  const p = path.join(RESULTS_DIR, name);
  if (!fs.existsSync(p)) return null;
  try {
    // Some artifacts were written by Python's json.dumps which emits bare
    // NaN/Infinity tokens (invalid JSON) for undefined metrics (e.g. AUROC of
    // the majority baseline). Tolerate them as null.
    const text = fs
      .readFileSync(p, "utf-8")
      .replace(/-Infinity/g, "null")
      .replace(/\bNaN\b/g, "null")
      .replace(/\bInfinity\b/g, "null");
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

export function readCsv<T>(name: string): T[] | null {
  const p = path.join(RESULTS_DIR, name);
  if (!fs.existsSync(p)) return null;
  try {
    const text = fs.readFileSync(p, "utf-8");
    const rows: string[][] = [];
    let row: string[] = [];
    let field = "";
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') {
            field += '"';
            i++;
          } else inQuotes = false;
        } else field += ch;
      } else if (ch === '"') inQuotes = true;
      else if (ch === ",") {
        row.push(field);
        field = "";
      } else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field);
        field = "";
        if (row.some((f) => f.trim() !== "")) rows.push(row);
        row = [];
      } else field += ch;
    }
    if (field !== "" || row.length > 0) {
      row.push(field);
      if (row.some((f) => f.trim() !== "")) rows.push(row);
    }
    if (rows.length < 2) return null;
    const header = rows[0];
    return rows.slice(1).map((r) => {
      const obj: Record<string, unknown> = {};
      header.forEach((h, i) => {
        const raw = (r[i] ?? "").trim();
        obj[h.trim()] = raw === "" ? null : /^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(raw) ? Number(raw) : raw === "true" ? true : raw === "false" ? false : raw;
      });
      return obj as T;
    });
  } catch {
    return null;
  }
}

export function readText(name: string): string | null {
  const p = path.join(RESULTS_DIR, name);
  if (!fs.existsSync(p)) return null;
  try {
    return fs.readFileSync(p, "utf-8");
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* B2 — corrected baselines                                            */
/* ------------------------------------------------------------------ */

export interface B2ModelRow {
  model: string;
  deterministic?: boolean;
  threshold?: number;
  n_seeds?: number;
  precision_mean?: number;
  recall_mean?: number;
  f1_mean?: number;
  f1_std?: number;
  auroc_mean?: number;
  auroc_std?: number;
  pr_auc_mean?: number;
  mcc_mean?: number;
  ece_mean?: number;
  brier_mean?: number;
  val_f1?: number;
}

export interface B2StatisticalTests {
  mcnemar_vs_best_baseline?: number | null;
  best_baseline?: string;
  bootstrap_xgb_f1_ci?: [number, number];
  bootstrap_xgb_auroc_ci?: [number, number];
  [key: string]: unknown;
}

export interface B2TuningPerSeed {
  best_params: Record<string, number>;
  best_cv_auc: number;
}

/* ------------------------------------------------------------------ */
/* B3 — cross-domain zero-shot                                         */
/* ------------------------------------------------------------------ */

export interface B3DatasetMetrics {
  n_seeds: number;
  precision_mean: number;
  recall_mean: number;
  f1_mean: number;
  f1_std: number;
  auroc_mean: number;
  auroc_std: number;
  pr_auc_mean: number;
  mcc_mean: number;
  ece_mean: number;
  ece_std: number;
  n_rows: number;
  n_groups: number;
  predicted_positive_rate: number;
  label_positive_rate: number;
  [key: string]: number;
}

export type B3DatasetMetricsMap = Record<string, B3DatasetMetrics | null>;

export interface B3TransferRow {
  subset: string;
  n_rows: number;
  n_groups: number;
  f1: number;
  auroc: number;
  delta_f1_vs_in_domain: number;
  delta_auroc_vs_in_domain: number;
  predicted_positive_rate: number;
  label_positive_rate: number;
}

export interface B3SubgroupRow {
  dimension: string;
  subgroup: string;
  n_rows: number;
  n_groups: number;
  reported: boolean;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  ece?: number | null;
  reason?: string | null;
}

export interface B3ErrorCase {
  sample_id: string;
  source_dataset: string;
  task?: string;
  domain?: string;
  generator_model?: string;
  question?: string;
  context?: string;
  answer?: string;
  label: number;
  prediction?: number;
  raw_score?: number;
  text_redacted?: string;
}

/* ------------------------------------------------------------------ */
/* B4 — calibration under shift                                       */
/* ------------------------------------------------------------------ */

export interface B4MethodMetrics {
  n_seeds: number;
  ece_mean: number;
  ece_std: number;
  ace_mean: number;
  brier_mean: number;
  nll_mean: number;
  slope_mean: number;
  intercept_mean: number;
  f1_mean: number;
  auroc_mean: number;
  predicted_positive_rate_mean: number;
  [key: string]: number | null;
}

/** subset -> method(raw|platt|isotonic) -> metrics */
export type B4CalibrationMetricsMap = Record<string, Record<string, B4MethodMetrics>>;

export interface B4TargetCalibration {
  n_calibration_rows: number;
  n_calibration_groups: number;
  n_test_rows: number;
  n_test_groups: number;
  overlapping_groups_removed: number;
  methods: Record<
    string,
    { ece_mean: number; ece_std?: number; brier_mean: number; nll_mean: number; [k: string]: number | null | undefined }
  >;
}

export interface B4SubgroupRow {
  dimension: string;
  subgroup: string;
  n_rows: number;
  n_groups: number;
  reported: boolean;
  platt_ece?: number | null;
  isotonic_ece?: number | null;
  raw_ece?: number | null;
  reason?: string | null;
}

/* ------------------------------------------------------------------ */
/* B5 — explanation reliability                                       */
/* ------------------------------------------------------------------ */

export interface B5FeatureImportance {
  kendall_tau_shap_vs_permutation: number;
  mean_abs_shap: Record<string, number>;
  permutation_importance: Record<string, number>;
  group_mean_abs_shap: Record<string, number>;
  group_ablation_f1_delta: Record<string, number>;
}

export interface B5NeutralizationRow {
  k: number;
  top_features: string[];
  mean_score_delta: number;
  f1_delta: number;
  auroc_delta: number | null;
}

export interface B5Stability {
  n_bootstrap: number;
  top_k: number;
  feature_mean_abs_shap_ci: Record<string, { lo: number; hi: number; mean: number }>;
  topk_set_jaccard: { mean: number; std: number };
}

export interface B5PerturbationAggregate {
  perturbation: string;
  n: number;
  mean_abs_score_delta: number;
  std_score_delta: number;
  large_delta_rate: number;
  top1_flip_rate: number;
  mean_spearman: number;
}

export interface B5ReviewCase {
  sample_id: string;
  question: string;
  context: string;
  answer: string;
  label: number;
  raw_score: number;
  calibrated_score: number;
  top5_shap_features: string;
  reviewer_1?: string;
  reviewer_2?: string;
  agreement?: string;
}

export interface B5FailureCase {
  sample_id: string;
  perturbation: string;
  raw_score: number;
  perturbed_score: number;
  score_delta: number;
  top1_flip: number;
  orig_top1: string;
  pert_top1: string;
}

/* ------------------------------------------------------------------ */
/* B6 — EC-XGB (evidence-consistent XGBoost)                          */
/* ------------------------------------------------------------------ */

export interface B6ComparisonRow {
  variant: string;
  n_features?: number;
  f1_mean?: number;
  f1_std?: number;
  auroc_mean?: number;
  auroc_std?: number;
  mcc_mean?: number;
  ece_mean?: number;
  brier_mean?: number;
}

export interface B6ExternalRow {
  variant: string;
  dataset: string;
  n: number;
  recall?: number;
  precision?: number;
  f1?: number;
  auroc?: number;
  predicted_positive_rate?: number;
  ece?: number;
  brier?: number;
}

export interface B6StrictRun {
  threshold: number;
  val_fpr: number;
  val_recall: number;
  test_f1: number;
  test_recall: number;
  test_fpr: number;
}

export interface B6StrictMode {
  alpha: number;
  per_run: Record<string, B6StrictRun>;
}

export interface B6Calibration {
  calibration_rows: number;
  test_rows: number;
  model: string;
  chosen: string;
  test_raw: { ece: number; brier: number };
  test_platt: { ece: number; brier: number };
  test_isotonic: { ece: number; brier: number };
}

export interface B6RunConfig {
  model_name?: string;
  model_full_name?: string;
  component_map?: Record<string, string>;
  variants?: string[];
  seeds?: number[];
}

/* ------------------------------------------------------------------ */
/* Manifest + legacy VA analysis                                      */
/* ------------------------------------------------------------------ */

export interface Manifest {
  generated_at: string;
  git_commit: string | null;
  source_fingerprint: string | null;
  model_version: string | null;
  feature_version: string | null;
  n_features: number | null;
  nli_model: string | null;
  seeds: number[];
  feature_groups: Record<string, string[]>;
  split_report: {
    n_groups_total?: number;
    leakage_free?: boolean;
    groups_spanning_multiple_splits?: number;
    [k: string]: unknown;
  } | null;
  dataset_sha256: Record<string, string | null>;
  raw_sha256: Record<string, string | null>;
  b_artifacts_sha256: Record<string, string | null>;
  versions: Record<string, string | null>;
  hardware: {
    cpu?: string | null;
    cuda?: boolean;
    gpu_name?: string | null;
    gpu_total_memory_gb?: number | null;
    ram_total_gb?: number | null;
  };
  env: Record<string, string>;
  artifacts: string[];
}

export interface ErrorAnalysis {
  n_test: number;
  n_false_positives: number;
  n_false_negatives: number;
  sampled: { fp: number; fn: number };
  category_counts: {
    false_positive: Record<string, number>;
    false_negative: Record<string, number>;
  };
}

export interface LatencyAnalysis {
  n_samples?: number;
  latency_ms?: Record<string, { p50?: number; p95?: number; mean?: number }>;
  total_per_sample_ms?: { p50?: number; p95?: number };
  model_artifact_mb?: Record<string, number> | number;
  cost_per_1000_predictions_usd?: Record<string, number>;
}

export interface LlmJudgeResults {
  n_samples: number;
  model: string;
  judge: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    latency_ms_p50: number;
    latency_ms_p95: number;
  };
  xgboost_on_same_subset: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
  agreement_with_xgboost: number;
  cost_usd: number;
  cost_per_1000_usd: number;
  mcnemar_p?: number;
  mcnemar_judge_vs_xgboost_p?: number;
}

export interface DashboardData {
  b2: {
    comparison: B2ModelRow[] | null;
    statisticalTests: B2StatisticalTests | null;
    tuning: Record<string, B2TuningPerSeed> | null;
    leakageComparison: unknown | null;
  };
  b3: {
    datasetMetrics: B3DatasetMetricsMap | null;
    transfer: B3TransferRow[] | null;
    subgroups: B3SubgroupRow[] | null;
    bootstrapCis: Record<string, { f1_ci?: number[]; auroc_ci?: number[] }> | null;
    labelSensitivity: Record<string, Record<string, number>> | null;
    errorCases: B3ErrorCase[] | null;
  };
  b4: {
    metrics: B4CalibrationMetricsMap | null;
    target: B4TargetCalibration | null;
    subgroups: B4SubgroupRow[] | null;
  };
  b5: {
    importance: B5FeatureImportance | null;
    neutralization: B5NeutralizationRow[] | null;
    stability: B5Stability | null;
    perturbations: B5PerturbationAggregate[] | null;
    reviewCases: B5ReviewCase[] | null;
    failureCases: B5FailureCase[] | null;
  };
  b6: {
    comparison: B6ComparisonRow[] | null;
    external: B6ExternalRow[] | null;
    strict: B6StrictMode | null;
    calibration: B6Calibration | null;
    runConfig: B6RunConfig | null;
  };
  manifest: Manifest | null;
  legacy: {
    errorAnalysis: ErrorAnalysis | null;
    latency: LatencyAnalysis | null;
    judge: LlmJudgeResults | null;
  };
}

export function loadDashboardData(): DashboardData {
  return {
  b2: {
    comparison: (() => {
      // b2_model_comparison.json is a dict keyed by model name, not a list.
      const raw = readJson<Record<string, Record<string, number | boolean | string>>>(
        "b2/b2_model_comparison.json",
      );
      if (!raw) return null;
      return Object.entries(raw).map(([model, v]) => ({ ...(v as unknown as B2ModelRow), model }));
    })(),
    statisticalTests: readJson<B2StatisticalTests>("b2/b2_statistical_tests.json"),
    tuning: readJson<Record<string, B2TuningPerSeed>>("b2/b2_tuning.json"),
    leakageComparison: readJson("b2/b2_leakage_comparison.json"),
  },
    b3: {
      datasetMetrics: readJson<B3DatasetMetricsMap>("b3/b3_dataset_metrics.json"),
      transfer: readCsv<B3TransferRow>("b3/b3_transfer_comparison.csv"),
      subgroups: readCsv<B3SubgroupRow>("b3/b3_subgroup_metrics.csv"),
      bootstrapCis: readJson<Record<string, { f1_ci?: number[]; auroc_ci?: number[] }>>(
        "b3/b3_bootstrap_cis.json",
      ),
      labelSensitivity: readJson<Record<string, Record<string, number>>>(
        "b3/b3_label_sensitivity.json",
      ),
      errorCases: readJson<B3ErrorCase[]>("b3/b3_error_cases.json"),
    },
    b4: {
      metrics: readJson<B4CalibrationMetricsMap>("b4/b4_calibration_metrics.json"),
      target: readJson<B4TargetCalibration>("b4/b4_target_calibration.json"),
      subgroups: readCsv<B4SubgroupRow>("b4/b4_subgroup_calibration.csv"),
    },
    b5: {
      importance: readJson<B5FeatureImportance>("b5/b5_feature_importance.json"),
      neutralization: readJson<B5NeutralizationRow[]>("b5/b5_neutralization.json"),
      stability: readJson<B5Stability>("b5/b5_stability_bootstrap.json"),
      perturbations: readCsv<B5PerturbationAggregate>("b5/b5_perturbation_aggregates.csv"),
      reviewCases: readJson<B5ReviewCase[]>("b5/b5_review_cases.json"),
      failureCases: readJson<B5FailureCase[]>("b5/b5_failure_cases.json"),
    },
    b6: {
      // pandas wrote the variant name into an unnamed index column.
      comparison: (() => {
        const rows = readCsv<Record<string, unknown>>("b6/b6_model_comparison.csv");
        if (!rows) return null;
        return rows
          .map((r) => ({ ...(r as unknown as B6ComparisonRow), variant: String(r[""] ?? "") }))
          .filter((r) => r.variant);
      })(),
      external: readCsv<B6ExternalRow>("b6/b6_external_metrics.csv"),
      strict: readJson<B6StrictMode>("b6/b6_strict_mode.json"),
      calibration: readJson<B6Calibration>("b6/b6_calibration.json"),
      runConfig: readJson<B6RunConfig>("b6/b6_run_config.json"),
    },
    manifest: readJson<Manifest>("manifest.json"),
    legacy: {
      errorAnalysis: readJson<ErrorAnalysis>("error_analysis.json"),
      latency: readJson<LatencyAnalysis>("latency_analysis.json"),
      judge: readJson<LlmJudgeResults>("llm_judge_results.json"),
    },
  };
}

/* Shared formatting helpers (render-safe for undefined) */

export function fmt(v: number | null | undefined, digits = 4): string {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

export function fmtPct(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`;
}

export function modelVersionFromManifest(m: Manifest | null): string {
  return m?.model_version ?? "unknown";
}

export function featureCount(): number | null {
  const p = path.join(MODELS_DIR, "feature_names.json");
  if (!fs.existsSync(p)) return null;
  try {
    const names = JSON.parse(fs.readFileSync(p, "utf-8"));
    return Array.isArray(names) ? names.length : null;
  } catch {
    return null;
  }
}

export function artifactSizeMb(): number | null {
  try {
    let total = 0;
    for (const p of fs.readdirSync(MODELS_DIR)) {
      const full = path.join(MODELS_DIR, p);
      if (fs.statSync(full).isFile()) total += fs.statSync(full).size;
    }
    return Math.round((total / 2 ** 20) * 10) / 10;
  } catch {
    return null;
  }
}
