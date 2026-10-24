# PROJECT_INVENTORY

- Root: selected project directory (absolute path omitted for privacy)
- Symlink content is not opened; inspect link targets manually if they are trusted and needed.
- Scanned: 2026-10-24 13:01
- Files seen: 2841
- Skipped directories: .git, .pytest_cache, .venv, __pycache__, build, node_modules

## Counts
- manuscript: 3
- bib: 1
- results: 214
- code: 1125
- config: 5
- data: 257
- figures: 88
- docs: 21
- office: 0
- secrets: 4
- other: 1123

## README files (paths only; content not copied)
- `README.md` (20.8KB)
- `docs\README.md` (4.0KB)
- `web\app\api\chat\threads\[id]\messages\public\fonts\README.md` (811B)

## LaTeX manuscript
- Main file(s): `Journal_Paper\halurisc.tex`
- Outline of `Journal_Paper\halurisc.tex`:
    Introduction
    Conventional Method
      Preliminaries
      Baseline Method
      Limitations and Research Gap
    Proposed Method
      Working Principle and Task
      Features
      Evidence-Consistent XGBoost
      Calibration and Explainability
      System and Reproducibility
    Experiments
      Experimental Setup and Conditions
      Dataset
      Experimental Procedure
      Parameter Configuration
      Evaluation Metrics
      Additional Protocols
    Results
      Model comparison
      Evidence-consistent XGBoost
      Calibration
      Cross-domain transfer
      Ablation
      Explanation reliability
      Efficiency and the LLM judge
      Error analysis
      Discussion
    Conclusion
    Declarations
    Acknowledgements
- Abstract found in `Journal_Paper\halurisc.tex` (281 words)
- Bib entries: 34; cited keys: 34
- Cited but missing from .bib (0): none
- In .bib but never cited (0): none

## Notebooks
- `colab\HaluRISC_Training_Version_B.ipynb`: 35 code cells, no outputs

## Result and log candidates (check these for the real numbers)
- `Journal_Paper\out\halurisc.log` (51.2KB; content not previewed)
- `artifacts\results\ablation_results.csv` (678B; content not previewed)
- `artifacts\results\b2\b2_bootstrap_cis.json` (172B; content not previewed)
- `artifacts\results\b2\b2_case_meta.json` (859B; content not previewed)
- `artifacts\results\b2\b2_case_scores.csv` (826B; content not previewed)
- `artifacts\results\b2\b2_confusion_matrices.json` (2.1KB; content not previewed)
- `artifacts\results\b2\b2_leakage_comparison.json` (671B; content not previewed)
- `artifacts\results\b2\b2_model_comparison.csv` (2.3KB; content not previewed)
- `artifacts\results\b2\b2_model_comparison.json` (5.4KB; content not previewed)
- `artifacts\results\b2\b2_per_seed_metrics.csv` (3.7KB; content not previewed)
- `artifacts\results\b2\b2_predictions.parquet` (373.2KB; content not previewed)
- `artifacts\results\b2\b2_run_config.json` (1.6KB; content not previewed)
- `artifacts\results\b2\b2_statistical_tests.json` (784B; content not previewed)
- `artifacts\results\b2\b2_tuning.json` (2.2KB; content not previewed)
- `artifacts\results\b2_smoke_test\b2_bootstrap_cis.json` (186B; content not previewed)
- `artifacts\results\b2_smoke_test\b2_confusion_matrices.json` (809B; content not previewed)
- `artifacts\results\b2_smoke_test\b2_leakage_comparison.json` (665B; content not previewed)
- `artifacts\results\b2_smoke_test\b2_model_comparison.csv` (1.5KB; content not previewed)
- `artifacts\results\b2_smoke_test\b2_model_comparison.json` (4.8KB; content not previewed)
- `artifacts\results\b2_smoke_test\b2_per_seed_metrics.csv` (1.0KB; content not previewed)
- `artifacts\results\b2_smoke_test\b2_predictions.parquet` (6.4KB; content not previewed)
- `artifacts\results\b2_smoke_test\b2_run_config.json` (1.3KB; content not previewed)
- `artifacts\results\b2_smoke_test\b2_statistical_tests.json` (834B; content not previewed)
- `artifacts\results\b2_smoke_test\b2_tuning.json` (759B; content not previewed)
- `artifacts\results\b2_test_tmp\b2_bootstrap_cis.json` (125B; content not previewed)
- `artifacts\results\b2_test_tmp\b2_confusion_matrices.json` (809B; content not previewed)
- `artifacts\results\b2_test_tmp\b2_leakage_comparison.json` (674B; content not previewed)
- `artifacts\results\b2_test_tmp\b2_model_comparison.csv` (1.6KB; content not previewed)
- `artifacts\results\b2_test_tmp\b2_model_comparison.json` (4.9KB; content not previewed)
- `artifacts\results\b2_test_tmp\b2_per_seed_metrics.csv` (1.1KB; content not previewed)
- `artifacts\results\b2_test_tmp\b2_predictions.parquet` (6.3KB; content not previewed)
- `artifacts\results\b2_test_tmp\b2_run_config.json` (1.3KB; content not previewed)
- `artifacts\results\b2_test_tmp\b2_statistical_tests.json` (760B; content not previewed)
- `artifacts\results\b2_test_tmp\b2_tuning.json` (760B; content not previewed)
- `artifacts\results\b3\b3_bootstrap_cis.json` (1.5KB; content not previewed)
- `artifacts\results\b3\b3_dataset_metrics.csv` (2.2KB; content not previewed)
- `artifacts\results\b3\b3_dataset_metrics.json` (4.6KB; content not previewed)
- `artifacts\results\b3\b3_error_cases.json` (538.6KB; content not previewed)
- `artifacts\results\b3\b3_label_sensitivity.json` (540B; content not previewed)
- `artifacts\results\b3\b3_predictions.parquet` (623.8KB; content not previewed)
- ... 174 more

## Code
- Source files: 1125; likely entry points: `colab\HaluRISC_Training_Version_B.ipynb`, `src\api\main.py`, `src\data\prepare.py`, `src\data\prepare_unified.py`, `src\models\eval_claims.py`, `src\models\eval_efficiency.py`, `src\models\eval_llm_judge.py`, `src\models\eval_ragtruth.py`, `src\models\run_all_experiments.py`, `src\models\run_b2_baselines.py`, `src\models\run_b3_cross_domain.py`, `src\models\run_b4_calibration_shift.py`, `src\models\run_b5_explanation_reliability.py`, `src\models\run_b6_modified.py`, `src\models\train_baselines.py`, `src\models\train_neural_adapted.py`, `src\models\train_neural_baseline.py`, `src\models\train_pipeline.py`, `tests\test_api.py`, `tests\test_b2_baselines.py`
- Code paths (first 60, filenames only): `colab\build_self_contained.py`, `colab\drive_cache.py`, `colab\HaluRISC_Training_Version_B.ipynb`, `src\__init__.py`, `src\api\main.py`, `src\claims\decompose.py`, `src\claims\judge.py`, `src\claims\verify.py`, `src\data\build_audit_review_sheet.py`, `src\data\download.py`, `src\data\download_faithbench.py`, `src\data\download_ragtruth.py`, `src\data\mappings.py`, `src\data\prepare.py`, `src\data\prepare_unified.py`, `src\data\registry.py`, `src\data\schema.py`, `src\explain\fig_b5_importance.py`, `src\explain\shap_analysis.py`, `src\features\claim_features.py`, `src\features\entity_features.py`, `src\features\extract_external_features.py`, `src\features\extract_features.py`, `src\features\nli_features.py`, `src\features\semantic_features.py`, `src\models\analyze_length_shortcut.py`, `src\models\config.py`, `src\models\error_analysis.py`, `src\models\eval_claims.py`, `src\models\eval_efficiency.py`, `src\models\eval_llm_judge.py`, `src\models\eval_ragtruth.py`, `src\models\export_case_scores.py`, `src\models\fit_display_calibrator.py`, `src\models\fit_ec_xgb_calibrator.py`, `src\models\freeze_ec_xgb_manifest.py`, `src\models\make_manifest.py`, `src\models\plot_paper_figures.py`, `src\models\plot_paper_figures_hires.py`, `src\models\review_baseline_latency.py`, `src\models\review_calibration_protocol.py`, `src\models\review_extras.py`, `src\models\review_judge_rows_comparison.py`, `src\models\review_length_control.py`, `src\models\review_length_matched_repeat.py`, `src\models\review_llm_judge_gpt_ec_xgb.py`, `src\models\review_llm_judge_qwen.py`, `src\models\review_monotone_ablation.py`, `src\models\review_number_audit.py`, `src\models\review_pipeline_latency.py`, `src\models\review_source_ablation.py`, `src\models\review_support_audit.py`, `src\models\review_tally.py`, `src\models\run_all_experiments.py`, `src\models\run_b2_baselines.py`, `src\models\run_b3_cross_domain.py`, `src\models\run_b4_calibration_shift.py`, `src\models\run_b5_explanation_reliability.py`, `src\models\run_b6_modified.py`, `src\models\train_baselines.py`
- ... 1065 additional source paths not shown
- Config and environment: `Dockerfile`, `requirements.txt`, `web\package.json`, `web\.next\package.json`, `web\.next\dev\package.json`
- Reminder: config is not proof of what was run; confirm settings against logs.

## Data files (do not paste private records; describe schema only)
- `artifacts/`: 2 file(s), 437.5KB
- `artifacts\models/`: 2 file(s), 2.5KB
- `artifacts\models\b4/`: 1 file(s), 1.2KB
- `colab/`: 1 file(s), 615B
- `data\processed/`: 13 file(s), 28.6MB
- `data\raw\faithbench/`: 16 file(s), 3.2MB
- `data\raw\halueval/`: 2 file(s), 5.9MB
- `data\raw\ragtruth/`: 1 file(s), 813.1KB
- `data\raw\ragtruth_official/`: 3 file(s), 34.9MB
- `docs/`: 1 file(s), 18.3KB
- `web/`: 1 file(s), 699B
- `web\.next/`: 10 file(s), 196.1KB
- `web\.next\dev/`: 4 file(s), 2.9KB
- `web\.next\dev\cache/`: 1 file(s), 33B
- `web\.next\dev\server/`: 5 file(s), 530B
- `web\.next\dev\server\app\(chat)\chat\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(present)\slide-check\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(present)\slide\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(site)\analyze\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(site)\dashboard\overview\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(site)\dashboard\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(site)\demo\page/`: 5 file(s), 2.0KB
- `web\.next\dev\server\app\(site)\page/`: 5 file(s), 1.9KB
- `web\.next\dev\server\app\api\chat\route/`: 3 file(s), 333B
- `web\.next\dev\server\app\api\chat\threads\[id]\messages\route/`: 3 file(s), 377B

## Figures
- `Journal_Paper\figures/`: 4 file(s), e.g. architecture-flow.png, architecture.png, fig_shap_importance_b5.png, system_architecture.png
- `Journal_Paper\figures\b3/`: 1 file(s), e.g. transfer_score_distributions.png
- `Journal_Paper\figures\b4/`: 2 file(s), e.g. calibration_shift.png, reliability_diagrams.png
- `Journal_Paper\screenshots/`: 10 file(s), e.g. analyze-arctic-model-comparison.png, analyze-arctic-risk-card.png, borderline.png, borderline_shap_model_compare.png, chat-details.png
- `artifacts\figures/`: 15 file(s), e.g. fig_reliability.pdf, fig_reliability.png, fig_roc_pr.pdf, fig_roc_pr.png, fig_shap_importance.pdf
- `artifacts\figures\b3/`: 4 file(s), e.g. context_length_robustness.png, in_domain_vs_ood.png, subgroup_performance.png, transfer_score_distributions.png
- `artifacts\figures\b4/`: 3 file(s), e.g. calibration_shift.png, reliability_diagrams.png, subgroup_calibration.png
- `docs\figures/`: 6 file(s), e.g. calibration_shift.png, fig_roc_pr.png, in_domain_vs_ood.png, reliability_diagrams.png, shap_importance_b5.png
- `web\.next\dev\static\media/`: 1 file(s), e.g. icon.0ae4m59k0su59.svg
- `web\.next\static\media/`: 1 file(s), e.g. icon.0ae4m59k0su59.svg
- `web\app/`: 1 file(s), e.g. icon.svg
- `web\app\api\chat\threads\[id]\messages\public\fonts/`: 3 file(s), e.g. fa-brands-400.svg, fa-regular-400.svg, fa-solid-900.svg
- `web\artifacts\figures/`: 15 file(s), e.g. fig_reliability.pdf, fig_reliability.png, fig_roc_pr.pdf, fig_roc_pr.png, fig_shap_importance.pdf
- `web\artifacts\figures\b3/`: 4 file(s), e.g. context_length_robustness.png, in_domain_vs_ood.png, subgroup_performance.png, transfer_score_distributions.png
- `web\artifacts\figures\b4/`: 3 file(s), e.g. calibration_shift.png, reliability_diagrams.png, subgroup_calibration.png

## Notes and documents (paths only; content not copied)
- `AGENTS.md` (18.0KB)
- `blueprint.md` (31.1KB)
- `Review_Checklist.md` (16.1KB)
- `roadmap.md` (57.6KB)
- `.agents\AGENTS.md` (1.4KB)
- `docs\01-project-and-methods.md` (37.5KB)
- `docs\02-experiments-and-results.md` (24.3KB)
- `docs\03-reproduction-and-defense.md` (24.2KB)
- `docs\04-interface-guide.md` (15.5KB)
- `docs\05-solving-hallucination-plan.md` (13.0KB)
- `docs\b5-explanation-reliability.md` (6.9KB)
- `docs\b6-reproducibility.md` (5.1KB)
- `docs\b7-research-ui.md` (11.0KB)
- `docs\prompt.md` (18.9KB)
- `docs\speech.md` (9.1KB)
- `docs\title_explanations.md` (12.7KB)
- `docs\writing-guideline.md` (33.7KB)
- `docs\archive\2026-09-17_Review_Checklist.md` (36.6KB)

## Candidate writing samples for the voice profile
Provenance is UNKNOWN for all of these. Use only text the author wrote; ask which files those are, and exclude AI-written or third-party text.
- `docs\writing-guide.pdf` (modified 2026-09-27)
- `Journal_Paper\halurisc.tex` (10395 words, modified 2026-10-24)
- `Journal_Paper\out\halurisc.pdf` (modified 2026-10-24)
- `AGENTS.md` (2287 words, modified 2026-10-19)
- `blueprint.md` (4278 words, modified 2026-09-26)
- `Review_Checklist.md` (2323 words, modified 2026-10-24)
- `roadmap.md` (8458 words, modified 2026-10-21)
- `.agents\AGENTS.md` (191 words, modified 2026-08-04)
- `docs\01-project-and-methods.md` (4947 words, modified 2026-09-26)
- `docs\02-experiments-and-results.md` (3655 words, modified 2026-10-19)
- `docs\03-reproduction-and-defense.md` (3471 words, modified 2026-09-26)
- `docs\04-interface-guide.md` (2556 words, modified 2026-09-25)
- `docs\05-solving-hallucination-plan.md` (2038 words, modified 2026-09-25)
- `docs\b5-explanation-reliability.md` (941 words, modified 2026-09-17)
- `docs\b6-reproducibility.md` (617 words, modified 2026-09-17)
- `docs\b7-research-ui.md` (1438 words, modified 2026-09-25)
- `docs\prompt.md` (3032 words, modified 2026-09-23)
- `docs\speech.md` (1658 words, modified 2026-10-19)
- `docs\title_explanations.md` (1930 words, modified 2026-10-19)
- `docs\writing-guideline.md` (4866 words, modified 2026-10-23)
- `docs\archive\2026-09-17_Review_Checklist.md` (5499 words, modified 2026-10-20)

## Possible secrets (listed, NOT read)
- `.env`
- `.env.example`
- `web\.env.local`
- `web\.npmrc`

## Next steps for the agent
1. Read trusted project documentation directly; use this inventory for paths and structure only.
2. Read result files directly only when needed and permitted; inspect schemas first and never copy private rows into summaries.
3. Fill `templates/project-brief.md` with source pointers; list conflicts and open questions.
4. Build `templates/voice-profile.md` from author-written samples only.
