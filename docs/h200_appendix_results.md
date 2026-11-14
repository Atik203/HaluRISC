# H200 Controls — Verified Results

Numbers only, produced on `uiu-h200` (`H200 1g.18gb`, driver `580.178.04`).
There is no appendix. The four controls sit in Sections 5.1, 5.5, and 5.6 of the
manuscript (Table 6, Figure 10, Table 19, Table 21). Prose for the manuscript is the
author's to write; this file is the source of truth for the numbers. Every table below is
reproducible from the script named with it.

Careful reading notes are at the end. Do not lift a number without the caveat that
belongs to it.

---

## E1 — Encoder sensitivity (in-domain, m2, 34 features)

Source: `artifacts/results/h200/encoder_ablation.{csv,json}`, from
`run_b6_modified.py` over three feature matrices and `h200_encoder_ablation.py`.
Protocol unchanged: grouped 70/15/15 split, 30-iteration randomized search, 5-fold
`StratifiedGroupKFold`, seeds 42/123/456.

| Encoder | F1 (mean ± std) | AUROC (mean ± std) | ΔF1 | ΔAUROC | McNemar p |
|---|---|---|---|---|---|
| DeBERTa-v3-base + MiniLM (deployed) | 0.9852 ± 0.0013 | 0.9986 ± 0.0001 | — | — | — |
| DeBERTa-v3-large + MiniLM | 0.9875 ± 0.0012 | 0.9978 ± 0.0001 | +0.0022 | −0.0008 | 0.63 |
| DeBERTa-v3-base + BGE-base | 0.9855 ± 0.0005 | 0.9984 ± 0.0000 | +0.0002 | −0.0002 | 0.77 |

Reproduction check: the `base` row (0.9852 / 0.9986) matches the frozen tuned m2
row (0.9857 / 0.9984) within 0.0005 F1 and 0.0002 AUROC, so the second machine
follows the same pipeline.

**Reading:** no encoder change is significant. The larger NLI checkpoint buys
+0.0022 F1 and loses 0.0008 AUROC; McNemar across three seeds does not separate it
from the deployed stack. AUROC is effectively saturated (0.998+), so encoder choice
is not the lever.

---

## E2 — Length-matched external control (frozen m3, no retraining)

Source: `artifacts/results/h200/length_matched.{csv,json}`, from
`h200_length_matched.py`. Frozen m3 boosters and frozen MedHallu predictions are
read as-is. The length-only baseline is logistic regression on `log1p(n_words)`
fitted on the 5,034 RAGTruth QA calibration rows.

| Dataset | Slice | EC-XGB AUROC | Length-only AUROC | Δ vs full |
|---|---|---|---|---|
| HaluBench | full | 0.6032 ± 0.0068 | 0.5099 | — |
| HaluBench | length-matched | 0.5882 ± 0.0088 | 0.4753 | −0.0151 |
| MedHallu | full | 0.8177 ± 0.0018 | 0.2604 | — |
| MedHallu | length-matched | 0.7707 ± 0.0034 | 0.4007 | −0.0470 |

MedHallu pair accuracy: full **0.8607**, length-matched **0.8290**.

**Reading:** the external numbers are not a length artifact. Equalising answer
length costs 1.5 AUROC points on HaluBench and 4.7 on MedHallu, and the model stays
far above the length-only baseline in both. The honest qualification is that
HaluBench sits near chance even before matching (0.603), so its transfer value is
weak regardless of length.

---

## E3 — Local judge at natural prevalence

Source: `artifacts/results/h200/judge_qwen9b_metrics.json` (copy of
`artifacts/results/rebuild/judge/metrics_qwen_qwen9b.json`).
900-row RAGTruth QA test split, 3 repeats, concurrency 1, zero failed calls.

| Judge | Precision | Recall | F1 | AUROC | Flagged | p50 (ms) | $/1K |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B (H200 slice) | 0.519 | 0.700 | 0.5957 | 0.7838 | 24.0% | 1466 | 0 |
| Qwen3.5-4B (frozen, RTX 3060) | 0.570 | 0.638 | 0.6018 | 0.7783 | 19.9% | 3050 | 0 |

Frozen reference rows already in the paper (`tab:judge`): GPT-5.6-Luna F1 0.565 /
AUROC 0.910 at $0.234 per 1,000; EC-XGB F1 0.310 / AUROC 0.558.

**Reading:** scale does not help the local judge. The 9B model trades 0.006 F1
down and 0.006 AUROC up, i.e. flat, and it still sits far below the API judge's
0.910 AUROC. It also shifts the operating point (recall 0.700, precision 0.519,
24.0% flagged). Agreement with EC-XGB is 0.274 exact match, so the two tools are
not interchangeable. Latency is **not** comparable across the two rows because
they ran on different GPUs.

---

## E4 — Throughput and VRAM on the MIG slice

Source: `artifacts/results/h200/efficiency.{csv,json}`, from `h200_efficiency.py`
with 2,000 rows, fp16, warmup pass before timing.

| Encoder | Batch | NLI rows/s | Peak VRAM (GB) |
|---|---|---|---|
| DeBERTa-base + MiniLM | 256 | 309 | 6.1 |
| DeBERTa-large + MiniLM | 256 | 122 | 10.3 |
| DeBERTa-base + BGE-base | 256 | 333 | 6.1 |
| DeBERTa-base + MiniLM | 128 | **361** | **3.2** |
| DeBERTa-base + MiniLM | 512 | 285 | 11.8 |

**Reading:** batch 128 is the throughput optimum on this slice, 17% faster than 256
at roughly half the memory, so the configured 256 is already past the knee.
DeBERTa-large costs 2.5x throughput and 1.7x VRAM for the null result in E1.

---

## Caveats that must travel with these numbers

1. **Single GPU slice, shared node.** Timings are from `1g.18gb` on a shared host.
   Report them as slice measurements, not H200 measurements.
2. **In-domain AUROC is saturated.** Any encoder comparison at 0.998 AUROC has no
   room to show an improvement; frame E1 as a null result, not a ranking.
3. **E1 uses m2** (26 base + 8 claim features) so the external RAGTruth features do
   not confound the encoder comparison. Do not present E1 as a test of the deployed
   m3 model.
4. **Length-only F1 is not reported** in E2. Its probability scale does not transfer
   across corpora with very different answer lengths, so only AUROC is comparable.
5. **HaluBench is near chance** (0.603 full, 0.588 matched). Do not describe the
   matched result as evidence of strong cross-domain transfer.
6. **E3 replaces the judge runtime, not the judge protocol.** If the 9B numbers
   differ from the frozen 4B row, that is a model-size effect, not a protocol change.
