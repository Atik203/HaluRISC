# WP6 report: judge and encoder at natural prevalence

Date: 2026-11-11
Branch: `rebuild`
Status: pilot complete. The full 900-row judge results are not included in
this report; the exact commands are below.
Scope: LLM-as-judge and encoder comparison only. No human evaluation was
performed, and none is claimed.

## What ran

| Component | Rows | Repeats | Status |
|---|---|---|---|
| GPT-5.6-Luna judge | 30 (natural-prevalence pilot) | 3 seeds | complete |
| Qwen 3.5 4B judge (Ollama) | 30 (same rows) | 3 seeds | complete |
| EC-XGB (m3) comparison rows | 900 test rows, 3 seeds | thresholds from 5,034 calibration rows | complete |
| Length-only baseline | fit on 5,034 calibration rows, tested on 900 | single fit | complete |
| RoBERTa-base timing | 200 rows | batch 8, fp16, CUDA | complete |

The 30-row pilot is a stratified sample of the 900 RAGTruth QA test rows:
5 hallucinated and 25 grounded (prevalence 0.1667), sampling seed 42. The same
30 rows were used for both judges. Pilot numbers are unstable because only 5
positives are in the sample. They are not final and must not be used as the
WP6 table.

## Judge metrics on the pilot

Natural prevalence in the full test split is 0.178. `PPR` is the predicted
positive (hallucinated) rate. Brackets are 95% bootstrap CIs over rows
(1,000 resamples, seed 777).

### GPT-5.6-Luna

| Seed | Accuracy | Precision | Recall | F1 | AUROC | PPR |
|---|---|---|---|---|---|---|
| 42 | 0.667 | 0.333 | 1.000 | 0.500 [0.200, 0.750] | 0.932 [0.821, 1.000] | 0.500 |
| 123 | 0.733 | 0.385 | 1.000 | 0.556 [0.222, 0.800] | 0.880 [0.730, 0.983] | 0.433 |
| 456 | 0.733 | 0.364 | 0.800 | 0.500 [0.167, 0.769] | 0.800 [0.559, 0.969] | 0.367 |
| mean +/- sd | 0.711 +/- 0.038 | 0.361 +/- 0.026 | 0.933 +/- 0.115 | 0.519 +/- 0.032 | 0.871 +/- 0.066 | 0.433 +/- 0.067 |

Confusion counts: seed 42 is TP 5, FP 10, TN 15, FN 0; seed 123 is TP 5,
FP 8, TN 17, FN 0; seed 456 is TP 4, FP 7, TN 18, FN 1. The judge over-flags
relative to the 0.167 pilot prevalence (PPR 0.433 on average) and buys recall
at the cost of precision.

Run-to-run agreement:

| Pair | Exact match | Cohen's kappa |
|---|---|---|
| 42 vs 123 | 0.867 | 0.733 |
| 42 vs 456 | 0.867 | 0.733 |
| 123 vs 456 | 0.800 | 0.585 |

Three-way exact agreement is 0.767; Fleiss' kappa across the three repeats is
0.683, and the mean of the three pairwise Cohen's kappas is 0.684. The
majority vote over three seeds agrees with the label on 0.700 of
rows (Cohen's kappa 0.372). The three repeats were unanimous on 23 of 30 rows.
Against EC-XGB (m3, seed 42, threshold 0.5), exact agreement is 0.533, 0.467,
and 0.400 for seeds 42, 123, and 456, with kappa 0.067, 0.051, and 0.039.

### Qwen 3.5 4B (local, Q4_K_M)

| Seed | Accuracy | Precision | Recall | F1 | AUROC | PPR |
|---|---|---|---|---|---|---|
| 42 | 0.800 | 0.429 | 0.600 | 0.500 [0.000, 0.824] | 0.720 [0.428, 0.962] | 0.233 |
| 123 | 0.800 | 0.429 | 0.600 | 0.500 [0.000, 0.824] | 0.720 [0.428, 0.962] | 0.233 |
| 456 | 0.800 | 0.429 | 0.600 | 0.500 [0.000, 0.824] | 0.720 [0.428, 0.962] | 0.233 |
| mean +/- sd | 0.800 +/- 0.000 | 0.429 +/- 0.000 | 0.600 +/- 0.000 | 0.500 +/- 0.000 | 0.720 +/- 0.000 | 0.233 +/- 0.000 |

Confusion counts are TP 3, FP 4, TN 21, FN 2 for every seed. The local model
flags less often than GPT (PPR 0.233 vs 0.433), which raises accuracy on this
pilot but lowers recall.

Run-to-run agreement is exact match 1.000 and Cohen's kappa 1.000 for all
three pairs; three-way exact agreement is 1.000 and Fleiss' kappa is 1.000
(mean pairwise Cohen's kappa 1.000). This is a property of the temperature-0
local setup, which returned identical labels and confidences for all three
seeds. It is not evidence that the model is robust across prompts or samples. The majority vote agrees with the label
on 0.800 of rows (kappa 0.379). Against EC-XGB, exact agreement is 0.267 for
every seed (kappa 0.021).

## Token counts, latency, and cost

### GPT-5.6-Luna pilot

| Quantity | Value |
|---|---|
| Calls | 90 |
| Prompt tokens | 145,938 |
| Cached prompt tokens | 130,241 (cache hit rate 0.892) |
| Uncached prompt tokens | 15,697 |
| Completion tokens | 14,240 |
| Reasoning tokens (inside completion) | 9,519 |
| Total tokens | 160,178 |
| Mean prompt / completion tokens per call | 1,621.5 / 158.2 |
| Latency mean / p50 / p95 | 2,764 / 2,505 / 4,940 ms |
| Cost | $0.02283 |
| Cost per 1,000 calls | $0.2537 |

Assumed prices for the cost calculation:

- input, uncached: $0.20 per million tokens;
- input, cached: $0.02 per million tokens;
- output: $1.20 per million tokens.

The input prices come from AGENTS.md. The output price is the value already
used in `src/models/eval_llm_judge.py`. These are assumptions, not a provider
invoice. Override them with `HALU_JUDGE_INPUT_USD_PER_MTOK`,
`HALU_JUDGE_CACHED_INPUT_USD_PER_MTOK`, and
`HALU_JUDGE_OUTPUT_USD_PER_MTOK`. At the pilot rate, 2,700 calls (900 rows x 3
seeds) would cost about $0.69. With no cache hits the same run would cost
about $1.39. The actual full-run cost will be recorded from the API usage, not
from this projection.

### Qwen 3.5 4B pilot

| Quantity | Value |
|---|---|
| Calls | 90 |
| Prompt tokens | 149,529 |
| Completion tokens | 3,750 |
| Total tokens | 153,279 |
| Mean prompt / completion tokens per call | 1,661.4 / 41.7 |
| Latency mean / p50 / p95 | 3,057 / 2,890 / 3,339 ms |
| API cost | $0.00 (local inference; electricity and hardware excluded) |

The Qwen maximum latency was 9,181 ms on one call with a long generation.
The first call also included a 3.7 s model load. At the pilot median, 2,700
calls with concurrency 1 would take about 2.2 hours, which fits an overnight
run.

## Comparison rows on the 900 test rows

All rows use the same 900 RAGTruth QA test rows (prevalence 0.178). The m3
operating threshold for 5% FPR is the smallest observed calibration negative
score whose empirical negative tail is at most 5% on the 5,034-row QA
calibration split, applied unchanged to the test rows. This tie-safe rule
keeps the calibration FPR at or below the target. AUROC does not depend on a
threshold; F1 is reported at 0.5 and at the 5% FPR operating point.

| Row | AUROC | F1 @ 0.5 | PPR @ 0.5 | F1 @ 5% FPR | FPR @ 5% |
|---|---|---|---|---|---|
| EC-XGB m3, seed 42 | 0.554 [0.508, 0.600] | 0.310 [0.274, 0.344] | 0.963 | 0.057 [0.019, 0.106] | 0.058 |
| EC-XGB m3, seed 123 | 0.540 [0.494, 0.586] | 0.308 [0.271, 0.341] | 0.978 | 0.073 [0.027, 0.123] | 0.068 |
| EC-XGB m3, seed 456 | 0.579 [0.535, 0.624] | 0.312 [0.276, 0.346] | 0.962 | 0.066 [0.020, 0.117] | 0.061 |
| EC-XGB m3, mean +/- sd | 0.558 +/- 0.020 | 0.310 +/- 0.002 | 0.968 +/- 0.009 | 0.066 +/- 0.008 | 0.062 +/- 0.005 |
| Length-only (log1p n_words) | 0.712 [0.667, 0.758] | 0.257 [0.177, 0.330] | 0.073 | 0.256 [0.178, 0.330] | 0.042 |
| Flag everything | undefined | 0.302 | 1.000 | undefined | undefined |

The FPR column is the realized test FPR at the calibration-selected threshold,
not the target. It exceeds 5% because the test score distribution shifts
relative to the calibration split. The threshold itself is fixed before the
test rows are scored. On the calibration negatives, the tie-safe threshold
yields 4.99% FPR for all three m3 seeds and for the length baseline.

The length-only baseline is a logistic regression on `log1p(n_words)` fit on
the 5,034 calibration rows. Its coefficient is 1.426 and its intercept is
-7.560. Its calibration AUROC is 0.707. It beats m3 on the 900 test rows
(AUROC 0.712 vs 0.558). m3 flags 96.8% of rows at threshold 0.5, so its F1 at
0.5 is essentially the flag-everything F1 (0.302). At the 5% FPR operating
point its F1 drops to 0.066. This is the expected negative QA result, not a
reason to tune the threshold on test.

Calibration-split AUROC for m3 is 0.570 (seed 42), 0.568 (seed 123), and 0.599
(seed 456).

## Encoder timing

RoBERTa-base (`roberta-base`, sequence-classification head with a randomly
initialized 2-class output) on 200 test rows:

| Quantity | Value |
|---|---|
| Device / dtype | CUDA / fp16 (RTX 3060 Laptop, 6 GB) |
| Batch size | 8 |
| Max sequence length | 512 |
| Rows timed | 200 (25 batches) |
| Model load | 3.476 s |
| Tokenization | 0.302 s |
| Timed forward pass | 1.394 s |
| Per row | 6.97 ms |
| Throughput | 143.5 rows/s |
| Batch latency mean / p50 / p95 | 55.76 / 44.30 / 102.05 ms |
| Peak GPU memory | 322.3 MB |
| Total script elapsed | 5.724 s |
| Rows reaching the 512-token cap | 81 of 200 |

The run is well under the 10-minute budget. The truncation count is a real
limitation: 40.5% of the 200 rows reach RoBERTa's 512-token limit, so an
encoder detector would lose part of the answer or context unless it uses a
longer-context model or a chunking strategy. These are timing numbers only;
no encoder accuracy is claimed.

## Exact full overnight command

The full run uses `--limit 0` (all 900 rows), three seeds, and resumable
JSONL checkpoints. Run from the repository root in PowerShell:

```powershell
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend openai --limit 0 --repeats 3 --concurrency 6 --out-tag full
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend qwen --limit 0 --repeats 3 --concurrency 1 --out-tag full
```

The Qwen pilot returned identical outputs for all three seeds at temperature
0. If that holds on the full 900 rows, one repeat is enough for the Qwen point
estimates, and `--repeats 1` cuts the local run from about 2.2 hours to about
45 minutes. The three-repeat command is kept so the GPT and Qwen tables have
the same run-to-run structure.

If a run is interrupted, re-running the same command resumes from
`calls_<backend>_full.jsonl`. Use `--fresh` only when deliberately discarding
a checkpoint. The OpenAI key is read from the root `.env` with
`load_dotenv(".env")`; it is never written to an artifact.

## Files created

- `src/models/rebuild_judge.py`: judge runner, metrics, comparison rows,
  encoder timing.
- `docs/rebuild/judge_prompt.md`: exact system and user prompts, decoding,
  seed handling, parsing rules, cost formula.
- `artifacts/results/rebuild/judge/REPORT.md`: this report.
- `artifacts/results/rebuild/judge/calls_openai_pilot.jsonl`
- `artifacts/results/rebuild/judge/calls_qwen_pilot.jsonl`
- `artifacts/results/rebuild/judge/predictions_openai_pilot.csv`
- `artifacts/results/rebuild/judge/predictions_qwen_pilot.csv`
- `artifacts/results/rebuild/judge/metrics_openai_pilot.json`
- `artifacts/results/rebuild/judge/metrics_qwen_pilot.json`
- `artifacts/results/rebuild/judge/comparison_rows.csv`
- `artifacts/results/rebuild/judge/comparison_rows.json`
- `artifacts/results/rebuild/judge/encoder_timing.json`
- `artifacts/results/rebuild/judge/encoder_run.log`
- `artifacts/results/rebuild/judge/pilot_openai_run.log`
- `artifacts/results/rebuild/judge/pilot_qwen_run.log`

## Blockers and limitations

- No blocker for the OpenAI or Qwen paths. The root `.env` key works, and
  Ollama is running with `qwen3.5:4b` (Q4_K_M) installed.
- The full 900-row judge runs were not executed in this session by design;
  the task asked for a 30-row pilot. The working full command is above.
- The pilot has only 5 positives, so its CIs are wide and its AUROC is
  unstable. GPT's pilot AUROC (0.871) and Qwen's (0.720) will move on 900
  rows.
- Qwen returned identical outputs at temperature 0 across the three seeds.
  Its agreement is therefore perfect by construction of the decoding setup,
  not because the judge is independently stable.
- No human evaluation, adjudication, or expert annotation was performed.
  The judge comparison is an automated evaluation against RAGTruth labels.
- The cost figures use assumed list prices and exclude any provider discounts,
  taxes, or local electricity.
