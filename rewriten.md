# Rewrite ledger

Every sentence added to `Journal_paper/halurisc.tex` in this pass, section by section, with the
exact position so a rewritten paragraph can be dropped back in place.

**Nothing you wrote was changed.** A diff against `main` shows 59 insertions and 1 deletion. The
deletion is a non-printing preamble comment (`18 tables and 11 figures` to `21 tables and 12 figures`).
No sentence of yours was edited, reordered, or deleted.

Floats were renumbered automatically by `\ref` because four new floats were inserted. Where your
existing sentences point at a float, the number changed by itself; the wording did not.

## How to use this file

Rewrite a paragraph as a whole and send it back. The anchor line tells me exactly where it goes, so
I replace the block between the two neighbouring sentences and keep your surrounding text intact.

| Block | Section | Drop-in position (anchor) | Float | Words |
|---|---|---|---|---|
| A | 5.1 Model Comparison | immediately before `\subsection{EC-XGB and Ablation}` | Table 6 | 125 |
| B | 5.5 Explanation, Efficiency, and Judges | immediately before the paragraph that starts "LLM judges are not fair evaluators" | Figure 10 | 124 |
| C | 5.5 Explanation, Efficiency, and Judges | immediately before `\subsection{Error Analysis and Discussion}` | Table 19 | 146 |
| D | 5.6 Error Analysis and Discussion | immediately before `\section{Conclusion}` | Table 21 | 130 |

Anchors by neighbouring text, for precision:

- **A** follows the sentence ending "...the feature model preserves its relative ordering on unseen tasks."
- **B** follows the sentence ending "...and the encoder time is at 512 tokens as an upper bound for the 256-token baseline."
- **C** follows the `Table 18` float (judge stress tests).
- **D** follows the sentence ending "...and Table 9 refits Platt without that selection."

Authorship note: this is draft text (rung C), built from verified artifacts. Rewrite it in your own
voice and check that the numbers survive. Keep this file as part of any AI-use disclosure record.

Numbers trace to `artifacts/results/h200/encoder_ablation.json`,
`artifacts/results/h200/length_matched.json`, `artifacts/results/h200/efficiency.csv`,
`artifacts/results/h200/judge_qwen9b_metrics.json`, and `artifacts/results/h200/judge_qwen4b_metrics.json`.

---

## Block A. Section 5.1, closing paragraph, introduces Table 6 (125 words)

The encoder choice is the last in-domain check. Table 6 replaces one encoder at a time and repeats the grouped split, the thirty-iteration search, and the five-fold cross-validation with the same three seeds. The base row reproduces the frozen m2 row within 0.0005 F1 and 0.0002 AUROC, which confirms that the pipeline is stable across machines. A larger NLI checkpoint adds 0.0022 F1 and loses 0.0008 AUROC, and the BGE embedding adds 0.0002 F1. McNemar tests return p = 0.63 and p = 0.77, so neither change separates from the deployed stack. AUROC already sits at 0.998, and a comparison at that level has little room to show a gain. The larger checkpoint is not worth the extra cost.

**Caption, Table 6.** In-domain encoder comparison on HaluEval QA (m2, 34 features, mean over seeds 42/123/456). Delta and McNemar p are against the deployed stack.

---

## Block B. Section 5.5, throughput paragraph, introduces Figure 10 (124 words)

The throughput measurements come from a second machine, an H200 MIG slice with 15.9 GB of usable memory, so they describe the feature pipeline rather than the results above. Figure 10 reports the sweep. Batch 128 is the fastest point for the deployed stack at 361 rows per second and 3.2 GB of memory. Batch 256 gives 309 rows per second at 6.1 GB, so the configured value is already past the knee. The large NLI checkpoint runs at 122 rows per second and 10.3 GB, about 2.5 times the extraction time of the deployed stack. That cost buys the null result in Table 6, and it is the practical reason to keep the base encoder in the deployed system.

**Caption, Figure 10.** NLI throughput and peak memory for three encoder stacks on the H200 MIG slice, with batch size varied for the deployed stack.

---

## Block C. Section 5.5, judge-scale paragraph, introduces Table 19 (146 words)

A local judge of the same family tests whether model size changes the picture. Table 19 adds Qwen3.5-9B on the same 900 rows. The 9B model reaches F1 0.5957 and AUROC 0.7838, against 0.6012 and 0.7755 for the 4B row, so F1 differs by 0.005 and AUROC by 0.008 between the two sizes, and neither closes the gap to the API judge at 0.910 AUROC. The 9B model flags 24.0% of the rows at recall 0.700 and precision 0.519. Cost stays at zero for both. Both rows come from the same slice, and the 4B model is about 1.5 times faster at the median. The 4B latency in Tables 17 and 18 was measured on the RTX 3060, which is why the same model is faster here. Exact agreement with EC-XGB is 0.274, so the judge and the feature model answer different questions and are not interchangeable.

**Caption, Table 19.** Local judges on the 900-row RAGTruth QA test split at natural prevalence, mean over three repeats, both measured on the same MIG slice.

---

## Block D. Section 5.6, closing paragraph, introduces Table 21 (130 words)

The in-domain check leaves the external corpora open, because their answers have a different length distribution. Table 21 repeats the two external comparisons after answer length is equalised. For HaluBench, each source keeps an equal number of PASS and FAIL answers inside each length bin. For MedHallu, the matched slice keeps the pairs whose two answers differ least in length. Matching costs 1.5 AUROC points on HaluBench and 4.7 on MedHallu, and pair accuracy falls from 0.8607 to 0.8290. The model stays well above the length-only baseline in both slices, at 0.588 against 0.475 on HaluBench and 0.771 against 0.401 on MedHallu. HaluBench is near chance in both slices, so its transfer value is weak with or without matching.

**Caption, Table 21.** External AUROC before and after answer-length matching, mean over seeds 42/123/456. Length-only is a logistic regression on log1p(n_words).

---

## Table and figure bodies (numbers only, no prose to rewrite)

Table 6 has three rows: DeBERTa-base + MiniLM, DeBERTa-large + MiniLM, DeBERTa-base + BGE-base.
Table 19 has two rows: Qwen3.5-9B and Qwen3.5-4B.
Table 21 has four rows: HaluBench full and length-matched, MedHallu full and length-matched.
Figure 10 has two panels: NLI throughput against batch size, and peak VRAM against batch size.

## Numbers you must not change while rewriting

| Value | Meaning |
|---|---|
| 0.9852 / 0.9986 | base encoder F1 / AUROC |
| 0.9875 / 0.9978 | large-NLI F1 / AUROC |
| 0.9855 / 0.9984 | BGE F1 / AUROC |
| +0.0022 / −0.0008 | F1 and AUROC change for large NLI |
| +0.0002 / −0.0002 | F1 and AUROC change for BGE |
| 0.63 / 0.77 | McNemar p for the two changes |
| 0.0005 / 0.0002 | reproduction gap against the frozen m2 row |
| 0.6032 → 0.5882 | HaluBench AUROC, full to matched |
| 0.8177 → 0.7707 | MedHallu AUROC, full to matched |
| 0.5099 / 0.4753 | HaluBench length-only AUROC |
| 0.2604 / 0.4007 | MedHallu length-only AUROC |
| 0.8607 → 0.8290 | MedHallu pair accuracy |
| 361 / 309 / 122 | rows per second at batch 128, 256, and for large NLI |
| 3.2 / 6.1 / 10.3 | peak GB in the same three runs |
| 15.9 | usable slice memory in GB |
| 0.5957 / 0.7838 | 9B judge F1 / AUROC |
| 0.6012 / 0.7755 | 4B judge F1 / AUROC |
| 0.700 / 0.519 | 9B recall / precision |
| 24.0% / 19.6% | flagged rows for 9B and 4B |
| 1466 / 976 | median call in ms for 9B and 4B |
| 1.5 | 4B speedup over 9B at the median |
| 3050 | 4B latency on the RTX 3060, quoted in Tables 17 and 18 |
| 0.274 | exact agreement between the 9B judge and EC-XGB |
| 0.910 | AUROC of the API judge on the same rows |
