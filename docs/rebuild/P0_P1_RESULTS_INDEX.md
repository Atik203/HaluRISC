# P0/P1 results index and reproduction

Every number added in the rebuild comes from one of the files below. The paper
locations are the current `Journal_Paper/halurisc.tex` labels.

## 1. P0 experiments

| Result | Artifact | Paper location |
|---|---|---|
| Fixed-hp m3 feature-group ablation | `artifacts/results/rebuild/p0/feature_group_ablation.csv` | Table 7, §5.2 |
| Per-seed ablation runs | `artifacts/results/rebuild/p0/feature_group_ablation_runs.csv` | supporting |
| Cross-seed SHAP stability | `artifacts/results/rebuild/p0/shap_seed_stability.csv` | §5.5 |
| CPU latency | `artifacts/results/rebuild/p0/m3_latency_cpu.json` | §5.5 |
| Length-matched judges | `artifacts/results/rebuild/p0/judge_length_matched.csv` | Table 17 lower panel |
| Prompt sensitivity | `artifacts/results/rebuild/p0/prompt_sensitivity.csv` | §5.5 |
| Annotation package | `docs/rebuild/p0/` | §3.4, §5.6, supplement |

Run:

```powershell
& .venv\Scripts\python.exe src\models\p0_feature_ablation.py
& .venv\Scripts\python.exe src\models\p0_shap_stability.py
& .venv\Scripts\python.exe src\models\rebuild_latency.py --limit 200 --out artifacts/results/rebuild/p0/m3_latency_cpu.json
& .venv\Scripts\python.exe src\models\rebuild_p0_judge.py length-matched
& .venv\Scripts\python.exe src\models\rebuild_p0_judge.py prompt-sensitivity
```

## 2. P1 experiments

| Result | Artifact | Paper location |
|---|---|---|
| External-style baseline | `artifacts/results/rebuild/p1/external_baseline.csv` | Table 5, §5.1 |
| MedHallu second corpus | `artifacts/results/rebuild/p1/medhallu.csv` | Table 11, §5.3 |
| HaluBench non-overlap subset | `artifacts/results/rebuild/p1/halubench.csv` | Table 11, §5.3 |
| HaluEval dialogue/summarization task shift | `artifacts/results/rebuild/p1/task_shift.csv` | Table 11, §5.3 |
| Retrieval stress | `artifacts/results/rebuild/p1/retrieval_stress.csv` | §5.3 |
| Split and class-conditional conformal | `artifacts/results/rebuild/p1/conformal.json` | Table 9, §5.3 |
| Multicalibration | `artifacts/results/rebuild/p1/multicalibration.csv` | §5.3 |
| Number audit | `artifacts/results/rebuild/review/p1_number_audit.csv` | build gate |

Run:

```powershell
& .venv\Scripts\python.exe src\models\p1_external_baseline.py
& .venv\Scripts\python.exe src\models\rebuild_medhallu.py
& .venv\Scripts\python.exe src\models\p1_halubench.py
& .venv\Scripts\python.exe src\models\p1_task_shift.py
& .venv\Scripts\python.exe src\models\p1_retrieval_stress.py
& .venv\Scripts\python.exe src\models\p1_conformal.py
```

## 3. Build and audit gates

```powershell
cd Journal_Paper
latexmk -pdf -interaction=nonstopmode -outdir=out halurisc.tex
cd ..
& .venv\Scripts\python.exe src\models\rebuild_number_audit.py
& .venv\Scripts\python.exe src\models\review_number_audit.py
& .venv\Scripts\python.exe src\models\p1_number_audit.py
& .venv\Scripts\python.exe src\models\float_proximity_audit.py
& .venv\Scripts\python.exe -m pytest tests -q
```

Current gate values:

- `rebuild_number_audit.py`: 116/116 match.
- `review_number_audit.py`: 298/298 match.
- `p1_number_audit.py`: 108/108 match.
- `float_proximity_audit.py`: 29/29 within two pages.
- PDF: 24 pages, 38 references, 0 overfull, 0 undefined.

## 4. Key results in one block

| Result | Value |
|---|---:|
| HaluEval m3 F1 / AUROC | 0.9855 / 0.9984 |
| Fixed-hp m3 ablation reference F1 / AUROC | 0.9861 / 0.9985 |
| Ablation ΔF1: minus lexical / length / NLI | -0.0171 / -0.0066 / -0.0040 |
| SHAP Spearman vs seed 42: 123 / 456 | 0.9717 / 0.9501 |
| CPU m3 latency mean / p50 / p95 | 2002 / 975 / 7565 ms |
| RAGTruth QA deployed AUROC / target ECE | 0.558 / 0.125 |
| Conformal QA coverage / positive coverage, 90% | 0.944 / 0.685 |
| Conformal FaithBench coverage, 90% | 0.320 |
| MedHallu m3 pair accuracy / AUROC / ECE | 0.867 / 0.821 / 0.493 |
| HaluBench m3 AUROC: overall / covidQA / DROP / FinanceBench | 0.600 / 0.833 / 0.513 / 0.527 |
| Task shift m3 AUROC: dialogue / summarization | 0.763 / 0.717 |
| Retrieval recall@1 / recall@5 / MRR | 0.94 / 1.00 / 0.959 |
| Judge natural QA GPT / Qwen F1 | 0.565 / 0.602 |
| Judge length-matched EC-XGB / GPT F1 | 0.907 / 0.865 |
| External baseline F1 / AUROC | 0.9482 / 0.9850 |

## 5. Author review

Author verification sign-off is complete (2026-11-12). The labels originated as
AI pre-annotations; the corresponding author checked all 310 rows and accepted
them on behalf of the author team. One author reviewed each main sheet, and all
three reviewed the 60-item blind overlap:

- `docs/rebuild/p0/author_review/AUTHOR_SIGNOFF.md` - sign-off record.
- `docs/rebuild/p0/author_review/AUTHOR_VERIFICATION.csv` - 310-row review sheet, all accepted.
- `docs/rebuild/p0/author_review/AI_REVIEW_RESULTS.md` - agreement, kappa, corrections, caveats.
- `docs/rebuild/p0/author_review/ingest/` - reviewed sheets in the ingest layout.

Result: AI-author agreement taxonomy 0.940 (Cohen κ 0.923), explanation 0.980
(κ 0.965), claims 0.880 (κ 0.788); blind inter-author Fleiss κ 0.868 / 0.940 /
0.761 (mean 0.857). The paper reports these as the author-verified review.

Rerun after any label edit:

```powershell
& .venv\Scripts\python.exe src\models\merge_p0_author_review.py
& .venv\Scripts\python.exe src\models\p0_ingest_author_review.py --docs-dir docs/rebuild/p0/author_review/ingest
```

Then copy the agreement and kappa values from
`artifacts/results/rebuild/p0/author_review_report.md` into the audit paragraph.
