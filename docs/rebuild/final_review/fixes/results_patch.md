# Results patch — exact LaTeX replacements

**Target:** `Journal_Paper/halurisc.tex` (current 25-page build; PDF checked at pp. 10, 14, 16). Line numbers below refer to that file.
**Sources:** `docs/rebuild/final_review/03_statistics.md` (S2, S3, S4, S5, S9), `docs/rebuild/final_review/04_results.md` (I1, I2, I3, I4, I5, I8), `docs/rebuild/P0_P1_RESULTS_INDEX.md`, and the artifacts cited under each edit.
**Rule:** apply only the `old` → `new` blocks below. No manuscript, PDF, or artifact was modified. Each `Current` block is a unique literal string in the current file; each `Replace with` block is the complete replacement.

---

## 1. P0 operating point and maximum target-Platt score

### 1A — P0 flag rate (L668)

**Current:**

```latex
On HaluEval test, recall is 99.0--99.5\% and FPR stays near 5\%, but on RAGTruth QA the same threshold flags 97.4\% of the tuned checkpoint, so P0 is not usable.
```

**Replace with:**

```latex
On HaluEval test, recall is 99.0--99.5\% and FPR stays near 5\%, but on RAGTruth QA the same threshold flags all 900 rows of the deployed seed-42 checkpoint, because every score exceeds 0.095, so P0 is not usable.
```

**Evidence:** `artifacts/results/rebuild/fixed_hp/b6_strict_mode.json` gives `m3_seed_42.threshold = 0.095`; `artifacts/results/rebuild/fixed_hp/b6_predictions.parquet` has 900/900 QA scores at or above 0.095 (minimum 0.1026). The 97.4% is the tuned B6 run's flag rate at tau = 0.5 in Table 10, not the P0 rate. 03_statistics S2.

### 1B — maximum target-Platt score (L668)

**Current:**

```latex
At 0.5, P1 flags no RAGTruth QA rows, and the maximum target-Platt score (0.313) is below the API's lowest band (0.35); the card therefore relies on the per-claim verdict adjustment.
```

**Replace with:**

```latex
At 0.5, P1 flags no RAGTruth QA rows; for the deployed seed-42 calibrator, the maximum target-Platt score is 0.355 (35.4\% before rounding, 35\% on the card), the median is 0.331, and 16\% of QA rows reach 0.35, so the card usually relies on the per-claim verdict adjustment.
```

**Evidence:** the seed-42 Platt coefficients in `artifacts/results/rebuild/calibration/calibration_report.json` (2.7803, -3.3771) applied to `b6_predictions.parquet` give max 0.3548, median 0.3311, and 16.4% at or above 0.35; the 35.4% score value, rounded to 35% on the card, is at L464. Keep 0.313 only in the P2 sentence, where it is the B2 target threshold. 03_statistics S3.

---

## 2. Table 10 checkpoint label

### 2A — caption (L741)

**Current:**

```latex
\caption{External and adaptation performance of B6 m0 and EC-XGB m3, mean over three seeds.}
```

**Replace with:**

```latex
\caption{External and adaptation performance of B6 m0 and EC-XGB m3 from the tuned B6 run, mean over three seeds.}
```

**Evidence:** the F1, AUROC, flag-rate, and ECE cells of Table 10 reproduce from `artifacts/results/b6/b6_external_metrics.csv` (pooled 0.4753/0.5815; QA 0.5491/97.4%; FaithBench 0.4500); the PR-AUC column is recomputed from `b6_predictions.parquet` and `external_features.parquet` (04_results check 9). The fixed-hp rebuild is the separate file `artifacts/results/rebuild/fixed_hp/b6_external_metrics.csv`. 04_results I4; 03_statistics S1.

### 2B — protocol sentence (L737, append to the last sentence)

**Current:**

```latex
ECE uses raw scores, F1 uses a fixed 0.5 threshold, and Flagged is the share above it.
```

**Replace with:**

```latex
ECE uses raw scores, F1 uses a fixed 0.5 threshold, and Flagged is the share above it. Tables~\ref{tab:ablation}, \ref{tab:target}, \ref{tab:external}, and \ref{tab:judge} report the fixed-hp refit; Table~\ref{tab:shift} reports the tuned B6 checkpoint.
```

**Evidence:** 04_results I4; fixed-hp artifacts `p0/feature_group_ablation.csv` (Table 7), `calibration/calibration_report.json` (Table 8), `p1/medhallu.csv`, `p1/halubench.csv`, and `p1/task_shift.csv` (Table 11), and `judge/comparison_rows.csv` (Table 17); tuned artifact `b6/b6_external_metrics.csv` (Table 10).

---

## 3. §5.1 encoder comparison

### 3 — per-task AUROCs (L557)

**Current:**

```latex
A source-only encoder wins in-domain (F1 0.9912 vs 0.9855) but is near chance on each RAGTruth task (AUROC 0.458 and 0.491 vs 0.582 for EC-XGB); pooling gives 0.337 because it mixes between-task score shifts with within-task ranking.
```

**Replace with:**

```latex
A source-only encoder wins in-domain (F1 0.9912 vs 0.9855) but is near chance on each RAGTruth task (AUROC 0.458 on summarization and 0.491 on data-to-text, against 0.736 and 0.770 for EC-XGB on those tasks); pooling gives 0.337 for the encoder and 0.582 for EC-XGB, because the encoder's pooled score mixes between-task score shifts with within-task ranking.
```

**Evidence:** `docs/rebuild/P0_P1_REVIEWER_RESPONSE.md` P1-8 gives 0.458 summarization, 0.491 data-to-text, and 0.337 pooled seed mean; the EC-XGB per-task values are 0.7358 and 0.7705 in Table 10, recomputed from `artifacts/results/b6/b6_predictions.parquet` and `external_features.parquet` (04_results check 9). 04_results I1.

### 3C1 — companion: 0.694 label in §5.1 (L557)

**Current:**

```latex
An adapted encoder on the same multi-source rows reaches 0.694 on the seen tasks, above EC-XGB's 0.582
```

**Replace with:**

```latex
An adapted encoder on the same multi-source rows reaches 0.694 on the pooled RAGTruth test set, above EC-XGB's 0.582
```

**Evidence:** `artifacts/results/review/neural_adapted/neural_adapted_summary.json` gives `external.ragtruth_all_test.raw_auroc.mean = 0.6940995` on 2,700 rows (QA plus summarization and data-to-text), so it is a pooled value, not a seen-task-only value; 04_results I5.

### 3C2 — companion: same label in §5.5 (L927)

**Current:**

```latex
the adapted encoder reaches 0.694 on seen RAGTruth tasks, above EC-XGB's 0.582
```

**Replace with:**

```latex
the adapted encoder reaches 0.694 on the pooled RAGTruth test set, above EC-XGB's 0.582
```

**Evidence:** same artifact as 3C1; 04_results I5.

---

## 4. WP5 audit count

### 4 — seven false positives (L955)

**Current:**

```latex
The WP5 audit covers all eight false positives and 12 of the 35 false negatives, chosen to cover the codebook.
```

**Replace with:**

```latex
The WP5 audit covers seven of the eight false positives and 12 of the 35 false negatives, chosen to cover the codebook.
```

**Evidence:** `artifacts/results/rebuild/explanations/m3_audit_summary.json` has `role_counts_m3.fp = 7` and `fn = 12`; `m3_audit_case_roles.csv` has seven `m3_fp_*` rows; the model has eight false positives in `artifacts/results/rebuild/fixed_hp/b6_confusion_matrices.json` (`m3_seed_42.fp = 8`). 04_results I3.

---

## 5. Author-review agreement swap

### 5 — explanation and taxonomy values (L854)

**Current:**

```latex
agreement was 0.94 for explanation, 0.98 for taxonomy, and 0.88 for claims, so the labels are author-verified.
```

**Replace with:**

```latex
agreement was 0.98 for explanation, 0.94 for taxonomy, and 0.88 for claims, so the labels are author-verified.
```

**Evidence:** `artifacts/results/rebuild/p0/author_review_summary.json` gives explanation 0.98 (kappa 0.9645, corrected rate 0.02), taxonomy 0.94 (kappa 0.9227, corrected rate 0.06), and claims 0.88 (kappa 0.7878, corrected rate 0.12). The manuscript has no other kappa or correction-rate wording to adjust, so the swap is the only change needed. 04_results I2.

---

## 6. Conditional statistical wording

### 6A — Table 8 slope bracket (caption L672, cell L681)

**Current caption:**

```latex
\caption{Calibration of the fixed-hp m3 model and the conventional target-Platt reference on the RAGTruth QA test set (n = 900, 17.8\% positive).}
```

**Replace with:**

```latex
\caption{Calibration of the fixed-hp m3 model and the conventional target-Platt reference on the RAGTruth QA test set (n = 900, 17.8\% positive). $^{\ast}$Slope: three-seed mean; bracket: envelope of the three per-seed 95\% intervals, not a pooled interval.}
```

**Current cell:**

```latex
    m3 raw          & none            & 0.731 & 0.689 & $-$3.71 & 0.18 [0.12, 0.27] & 96.8\% \\
```

**Replace with:**

```latex
    m3 raw          & none            & 0.731 & 0.689 & $-$3.71 & 0.18 [0.12, 0.27]$^{\ast}$ & 96.8\% \\
```

**Evidence:** `artifacts/results/rebuild/calibration/calibration_slope.csv` gives per-seed intervals [0.1212, 0.1962], [0.1183, 0.1948], and [0.1876, 0.2651]; the mean slope is 0.1806 and the envelope is [0.12, 0.27]. 03_statistics S4; 04_results I8.

### 6B — McNemar b/c reversal (L823)

**Current:**

```latex
    EC-XGB vs conventional (fixed hp) & 6  & 8  & 0.789            & ---              \\
```

**Replace with:**

```latex
    EC-XGB vs conventional (fixed hp) & 8  & 6  & 0.789            & ---              \\
```

**Evidence:** `artifacts/results/rebuild/fixed_hp/b6_predictions.parquet`, seed 42 at tau = 0.5, gives b = 8 (EC-XGB right, conventional wrong) and c = 6; the continuity-corrected p is 0.7893. 03_statistics S5.

### 6C — conformal finite-sample correction (L697–L709)

**Current block:**

```latex
\begin{equation}
  \hat{q} = \mathrm{Quantile}_{1-\alpha}\left(\{s(x_i,y_i)\}_{i=1}^{n}\right),
  \qquad
  C(x) = \{y : s(x,y) \le \hat{q}\}.
  \label{eq:conformal}
\end{equation}
A class-conditional version replaces the single quantile with one quantile per label,
\begin{equation}
  \hat{q}_y = \mathrm{Quantile}_{1-\alpha}\left(\{s(x_i,y_i) : y_i = y\}\right),
  \qquad
  C(x) = \{y : s(x,y) \le \hat{q}_y\}.
  \label{eq:conformal_class}
\end{equation}
```

**Replace with:**

```latex
\begin{equation}
  \hat{q} = s_{(k)}, \quad k = \left\lceil (n+1)(1-\alpha) \right\rceil,
  \qquad
  C(x) = \{y : s(x,y) \le \hat{q}\},
  \label{eq:conformal}
\end{equation}
where $s_{(1)} \le \cdots \le s_{(n)}$ are the ordered calibration scores; this finite-sample order statistic yields coverage of at least $1-\alpha$ for the prediction set. A class-conditional version replaces the single quantile with one quantile per label,
\begin{equation}
  \hat{q}_y = s_{y,(k_y)}, \quad k_y = \left\lceil (n_y+1)(1-\alpha) \right\rceil,
  \qquad
  C(x) = \{y : s(x,y) \le \hat{q}_y\},
  \label{eq:conformal_class}
\end{equation}
where $s_{y,(1)} \le \cdots \le s_{y,(n_y)}$ are the ordered scores in class $y$ and $n_y$ is the number of calibration rows with label $y$.
```

**Evidence:** `artifacts/results/rebuild/p1/conformal.json` records `protocol.quantile = "ceil((n+1)(1-alpha))-th smallest calibration score"`; `src/models/p1_conformal.py:209-216` implements that rank and lines 595-596 apply the same function to the positive and negative calibration scores. 03_statistics S9.

---

## Not patched — author decisions

| Item | Why it is not in the exact patch | Status |
|---|---|---|
| Table 8 slope estimand | The ready fix keeps the three-seed mean 0.18 and labels the bracket as a per-seed envelope. Replacing the point with the seed-42 slope 0.159 [0.121, 0.196], or computing a two-level CI for the mean, changes the estimand and needs a choice. | `[AUTHOR DECISION]` |
| Table 10 regeneration | The ready fix labels Table 10 as tuned. Regenerating it from `artifacts/results/rebuild/fixed_hp/b6_external_metrics.csv` would require recomputing PR-AUC and the per-task rows, then rewriting the §5.3 pooled, per-task, QA, and FaithBench sentences (pooled AUROC 0.494→0.590, QA AUROC 0.556→0.558 and flag 96.8%, FaithBench ECE 0.453). | `[AUTHOR DECISION]` |
| Conformal class-conditional S10 | The 0.887 class-conditional positive coverage uses the full 5,034 calibration rows, while the 0.685 comparison uses the disjoint split. A like-for-like comparison needs the class-conditional quantiles rerun on the disjoint half (0.660→0.887), not a wording change. | `[AUTHOR DECISION]` |
| Table 11 and Table 17 captions | The new §5.3 sentence labels both as fixed-hp, but their own captions still omit the label. Adding “fixed-hp m3” to those captions is exact but outside the requested Table 10 edit. | `[AUTHOR DECISION — scope]` |
| McNemar item clustering and Holm family | S6 (row-level McNemar while the bootstrap unit is the item pair) and S7 (two TF-IDF comparisons outside the Holm family) need new tests or a pre-specification statement, not a number swap. | `[AUTHOR DECISION]` |

---

## Summary

- **11 required exact edits are ready** across the six issues: 1A, 1B, 2A, 2B, 3, 4, 5, 6A-caption, 6A-cell, 6B, and 6C.
- **2 companion exact edits are ready** for the same encoder claim: 3C1 in §5.1 and 3C2 in §5.5. They are confirmed by 04_results I5 but are outside the six-item list.
- **No number is invented.** Every replacement traces to an artifact or to a value recomputed from `b6_predictions.parquet` / `calibration_report.json`.
- **Decisions needed:** the Table 8 slope estimand (seed-42 value versus a two-level CI for the mean), whether to regenerate Table 10 instead of labelling it, and whether to rerun the class-conditional conformal check on the disjoint half. Table 11/17 caption labels and the S6/S7 tests are scope decisions.
