import type { AblationRow, BaselineRow, ShiftRow, SlideData, TaskRow } from "@/components/slides/slides";
import { artifactSizeMb, loadDashboardData, readJson } from "@/lib/results";

/**
 * Server-side assembly of the presentation payload. Every value is read from
 * artifacts/ at request time (or bake time for the static route), never hardcoded.
 */

const BASELINE_ORDER: Array<[string, string]> = [
  ["heuristic_overlap", "Overlap heuristic (1 − overlap)"],
  ["lr_full", "Logistic regression"],
  ["rf_full", "Random forest"],
  ["nli_only", "NLI-only"],
  ["tfidf_answer", "TF-IDF (answer-only)"],
  ["tfidf_all", "TF-IDF (full input)"],
  ["xgboost", "XGBoost (standard)"],
];

const ABLATION_LABELS: Record<string, string> = {
  m0: "m0 · Standard XGBoost",
  m1: "m1 · Debias + constraints",
  m2: "m2 · + claim features",
  m3: "m3 · EC-XGB (deployed)",
};

const SHIFT_DATASETS: Array<[string, string]> = [
  ["ragtruth_all_test", "RAGTruth (all tasks)"],
  ["ragtruth_qa_test", "RAGTruth QA (held out)"],
  ["faithbench", "FaithBench"],
];

export function buildSlideData(): SlideData {
  const d = loadDashboardData();
  const manifest = d.manifest;
  const run = d.b6.runConfig;

  const baselines: BaselineRow[] = BASELINE_ORDER.flatMap(([key, label]) => {
    const row = d.b2.comparison?.find((r) => r.model === key);
    if (!row) return [];
    return [
      {
        key,
        label,
        f1: row.f1_mean,
        auroc: row.auroc_mean,
        prAuc: row.pr_auc_mean,
        mcc: row.mcc_mean,
        ece: row.ece_mean,
        brier: row.brier_mean,
        highlighted: key === "xgboost",
      },
    ];
  });

  const ablation: AblationRow[] = ["m0", "m1", "m2", "m3"].flatMap((variant) => {
    const row = d.b6.comparison?.find((r) => r.variant === variant);
    if (!row) return [];
    return [
      {
        variant,
        label: ABLATION_LABELS[variant] ?? variant,
        nFeatures: row.n_features,
        f1: row.f1_mean,
        auroc: row.auroc_mean,
        mcc: row.mcc_mean,
        ece: row.ece_mean,
        highlighted: variant === "m3",
      },
    ];
  });

  const ext = (dataset: string, variant: string) =>
    d.b6.external?.find((r) => r.dataset === dataset && r.variant === variant);
  const shift: ShiftRow[] = SHIFT_DATASETS.flatMap(([dataset, datasetLabel]) => {
    const std = ext(dataset, "m0");
    const ec = ext(dataset, "m3");
    if (!std || !ec) return [];
    return [
      {
        datasetLabel,
        standard: { f1: std.f1, auroc: std.auroc, flagged: std.predicted_positive_rate, ece: std.ece },
        ecxgb: { f1: ec.f1, auroc: ec.auroc, flagged: ec.predicted_positive_rate, ece: ec.ece },
      },
    ];
  });

  const strictRuns = d.b6.strict ? Object.values(d.b6.strict.per_run) : [];
  const rangeText = (pick: (r: (typeof strictRuns)[number]) => number) => {
    if (strictRuns.length === 0) return "—";
    const values = strictRuns.map((r) => pick(r) * 100);
    return `${Math.min(...values).toFixed(1)}%–${Math.max(...values).toFixed(1)}%`;
  };

  const calibration = d.b6.calibration;
  const target = d.b4.target;
  const b4metrics = d.b4.metrics;
  const b4qa = b4metrics?.ragtruth_qa_test;
  const b4hal = b4metrics?.halueval_test;
  const taskBreakdown: TaskRow[] = (
    [
      ["ragtruth_task_qa", "QA"],
      ["ragtruth_task_summarization", "Summarization"],
      ["ragtruth_task_data_to_text", "Data-to-text"],
    ] as Array<[string, string]>
  ).flatMap(([subset, label]) => {
    const row = d.b3.transfer?.find((r) => r.subset === subset);
    if (!row) return [];
    return [
      { label, nRows: row.n_rows, f1: row.f1, auroc: row.auroc, flagged: row.predicted_positive_rate },
    ];
  });
  const stats = d.b2.statisticalTests;
  const importance = d.b5.importance;
  const stability = d.b5.stability;
  const perturbations = d.b5.perturbations ?? [];
  const deltaFor = (needle: string) =>
    perturbations.find((p) => p.perturbation.toLowerCase().includes(needle))?.mean_abs_score_delta ?? null;

  const latency = d.legacy.latency;
  const judge = d.legacy.judge;

  const confusions = readJson<Record<string, { fp?: number }>>("b2/b2_confusion_matrices.json");
  const fpFor = (prefix: string) => {
    if (!confusions) return null;
    const key = Object.keys(confusions).find((k) => k.startsWith(prefix));
    return key ? (confusions[key]?.fp ?? null) : null;
  };
  const fpParts = [
    ["XGBoost", fpFor("xgboost_seed_42")],
    ["random forest", fpFor("rf_full_seed_42")],
    ["logistic regression", fpFor("lr_full_seed_42")],
    ["heuristic", fpFor("heuristic_overlap")],
  ]
    .filter(([, v]) => v != null)
    .map(([name, v]) => `${name} ${v}`);
  const fpNote = fpParts.length ? `False positives at the same threshold (seed 42): ${fpParts.join(" · ")}.` : "";

  const baseFeatures = run?.base_features?.length ?? 26;
  const claimFeatures = run?.claim_features?.length ?? 8;
  const featureGroups: Record<string, string[]> = {
    ...(manifest?.feature_groups ?? {}),
    ...(run?.claim_features ? { claim: run.claim_features } : {}),
  };

  return {
    modelVersion: manifest?.model_version ?? "b6-ec-xgb-v1.0",
    featureVersion: manifest?.feature_version ?? "course-v1.0",
    nliModel: manifest?.nli_model ?? "cross-encoder/nli-deberta-v3-base",
    seedsText: (run?.seeds ?? manifest?.seeds ?? [42, 123, 456]).join(" / "),
    nIter: run?.n_iter_tuning ?? 30,
    baseFeatures,
    claimFeatures,
    totalFeatures: baseFeatures + claimFeatures + 1,
    featureGroups,
    componentMap: run?.component_map ?? {},
    baselines,
    ablation,
    shift,
    taskBreakdown,
    mcnemarP: stats?.mcnemar_vs_best_baseline ?? null,
    mcnemarBest: (stats?.best_baseline as string | undefined) ?? null,
    xgbF1CiLo: stats?.bootstrap_xgb_f1_ci?.[0] ?? null,
    xgbF1CiHi: stats?.bootstrap_xgb_f1_ci?.[1] ?? null,
    displayMethod: calibration?.chosen ?? "platt",
    calibrationRows: calibration?.calibration_rows ?? null,
    b4RawEce: b4qa?.raw?.ece_mean ?? null,
    b4PlattEce: b4qa?.platt?.ece_mean ?? null,
    b4IsoEce: b4qa?.isotonic?.ece_mean ?? null,
    halRawEce: b4hal?.raw?.ece_mean ?? null,
    halPlattEce: b4hal?.platt?.ece_mean ?? null,
    displayRawEce: calibration?.test_raw?.ece ?? null,
    displayRawBrier: calibration?.test_raw?.brier ?? null,
    displayEce: calibration?.test_platt?.ece ?? null,
    displayBrier: calibration?.test_platt?.brier ?? null,
    sourceEce: target?.methods?.raw_reference?.ece_mean ?? null,
    targetEce: target?.methods?.platt?.ece_mean ?? null,
    kendall: importance?.kendall_tau_shap_vs_permutation ?? null,
    jaccard: stability?.topk_set_jaccard?.mean ?? null,
    entityDelta: deltaFor("entity"),
    numberDelta: deltaFor("number"),
    irrelevantDelta: deltaFor("irrelevant"),
    latencyP50: latency?.total_per_sample_ms?.p50 ?? null,
    artifactMb: artifactSizeMb(),
    costPer1k: latency?.cost_per_1000_predictions_usd?.halurisc_local ?? null,
    judgeF1: judge?.judge?.f1 ?? null,
    judgeCostPer1k: judge?.cost_per_1000_usd ?? null,
    judgeModel: judge?.model ?? null,
    judgeN: judge?.n_samples ?? null,
    judgeAcc: judge?.judge?.accuracy ?? null,
    judgePrec: judge?.judge?.precision ?? null,
    judgeRec: judge?.judge?.recall ?? null,
    judgeP50: judge?.judge?.latency_ms_p50 ?? null,
    judgeP95: judge?.judge?.latency_ms_p95 ?? null,
    judgeAgreement: judge?.agreement_with_xgboost ?? null,
    judgeMcnemarP: judge?.mcnemar_judge_vs_xgboost_p ?? null,
    judgeWrongRight: judge?.discordant_pairs?.judge_wrong_xgb_right ?? null,
    judgeRightWrong: judge?.discordant_pairs?.judge_right_xgb_wrong ?? null,
    xgbSameAcc: judge?.xgboost_on_same_subset?.accuracy ?? null,
    leakFree: manifest?.split_report?.leakage_free ?? null,
    leakSpanning: manifest?.split_report?.groups_spanning_multiple_splits ?? null,
    labelBalanceText: (() => {
      const lm = manifest?.split_report?.label_mean_per_split as Record<string, number> | undefined;
      if (!lm) return "—";
      const vals = Object.values(lm);
      return vals.length && vals.every((v) => Math.abs(v - 0.5) < 1e-9)
        ? "50 / 50"
        : vals.map((v) => `${(v * 100).toFixed(0)}%`).join(" · ");
    })(),
    modelJudgeF1: judge?.xgboost_on_same_subset?.f1 ?? null,
    strictAlphaText: d.b6.strict ? `${(d.b6.strict.alpha * 100).toFixed(1)}%` : "5%",
    strictFprText: rangeText((r) => r.test_fpr),
    strictRecallText: rangeText((r) => r.test_recall),
    fpNote,
  };
}
