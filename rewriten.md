# Draft text for author rewrite

Every sentence added to `Journal_Paper/halurisc.tex` in this pass, formatted as paragraphs so each
block can be copied and rewritten whole.

Scope: Appendix A only. Two subsections, five body paragraphs, three table captions, one figure
caption. One preamble comment changed (`18 tables and 11 figures` to `20 tables and 12 figures`) and
it does not print. Every existing sentence in Sections 1 to 6, and every existing table and figure,
is unchanged.

Authorship rung C: this is draft text built from verified artifacts, not finished prose. Rewrite it
in your own voice and check that the numbers survive. If your institution requires an AI-assistance
disclosure, keep this file as part of that record.

Numbers trace to `artifacts/results/h200/encoder_ablation.json`,
`artifacts/results/h200/length_matched.json`, `artifacts/results/h200/efficiency.csv`, and
`artifacts/results/rebuild/judge/metrics_qwen_qwen9b.json`.

---

## Appendix A title

Additional Controls and Checks

## Subsection A.1 title

Encoder and Length Controls

## Subsection A.2 title

Local Judge Scale

---

## Appendix A, paragraph 1 (setup, 95 words)

The deployed model uses DeBERTa-v3-base for the NLI features and MiniLM for the cosine features. Both are small checkpoints. A reader may ask whether the reported results depend on that choice. Two controls answer the question. The first replaces one encoder at a time and repeats the full protocol on the new feature matrix. The second removes the length shortcut from the external corpora. The deployed model is not retrained in either control. All runs reuse the grouped split, the thirty-iteration search, and the five-fold cross-validation with seeds 42, 123, and 456, so only the encoder changes.

---

## Appendix A, paragraph 2 (Table 19, encoder comparison, 120 words)

Table 19 reports the in-domain comparison. The base row reproduces the frozen m2 row within 0.0005 F1 and 0.0002 AUROC, which confirms that a second machine follows the same pipeline. A larger NLI checkpoint adds 0.0022 F1 and loses 0.0008 AUROC. The BGE embedding adds 0.0002 F1. McNemar tests across the three seeds return p = 0.63 and p = 0.77, so neither change separates from the deployed stack. AUROC already sits at 0.998, and a comparison at that level has little room to show a gain. The larger checkpoint is not worth the extra cost.

---

## Appendix A, paragraph 3 (Table 20, length-matched control, 129 words)

Table 20 repeats the two external comparisons after answer length is equalised. For HaluBench, each source keeps an equal number of PASS and FAIL answers inside each length bin. For MedHallu, the matched slice keeps the pairs whose two answers differ least in length. Matching costs 1.5 AUROC points on HaluBench and 4.7 on MedHallu. Pair accuracy falls from 0.8607 to 0.8290. The model stays well above the length-only baseline in both slices, at 0.588 against 0.475 on HaluBench and 0.771 against 0.401 on MedHallu. HaluBench is near chance in both slices, so its transfer value is weak with or without matching.

---

## Appendix A, paragraph 4 (Figure 12, throughput and memory, 112 words)

Figure 12 reports throughput and memory for the three encoder stacks on one H200 MIG slice with 15.9 GB of usable memory. The measurements come from a second machine, not the RTX 3060 used for the main experiments. Batch 128 is the fastest point for the deployed stack at 361 rows per second and 3.2 GB of memory. Batch 256 gives 309 rows per second at 6.1 GB, so the configured value is already past the knee. The large NLI checkpoint runs at 122 rows per second and 10.3 GB. The null result in Table 19 therefore costs about 2.5 times the extraction time, which is the practical argument for keeping the deployed encoder.

---

## Appendix A, paragraph 5 (Table 21, local judge scale, 113 words)

Table 21 adds a larger local judge on the same 900-row QA split. Qwen3.5-9B reaches F1 0.5957 and AUROC 0.7838, and it flags 24.0% of the rows. The frozen 4B row sits at F1 0.6018 and AUROC 0.7783. The two sizes differ by 0.006 F1 and 0.006 AUROC in opposite directions, and the larger model does not recover the ranking that the API judge keeps at 0.910 AUROC. The 9B model moves recall up to 0.700 and precision down to 0.519, so it marks more rows as hallucinated. Cost stays at zero and the median call takes 1.47 s on the slice. Its exact agreement with EC-XGB is 0.274, which means the two tools answer different questions and should not be substituted for one another.

---

## Caption, Table 19

In-domain encoder comparison on HaluEval QA (m2, 34 features, mean over seeds 42/123/456). Delta and McNemar p are against the deployed stack.

---

## Caption, Table 20

External AUROC before and after answer-length matching, mean over seeds 42/123/456. Length-only is a logistic regression on log1p(n_words).

---

## Caption, Table 21

Local judges on the 900-row RAGTruth QA test split at natural prevalence, mean over three repeats. Dagger marks the frozen row from the RTX 3060, so its latency is not comparable.

---

## Caption, Figure 12

NLI throughput and peak memory for three encoder stacks on the H200 MIG slice, with batch size varied for the deployed stack.

---

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
| 0.6018 / 0.7783 | 4B judge F1 / AUROC (frozen row) |
| 0.700 / 0.519 | 9B recall / precision |
| 24.0% / 19.9% | flagged rows for 9B and 4B |
| 1466 / 3050 | median call in ms for 9B and 4B |
| 0.274 | exact agreement between the 9B judge and EC-XGB |
| 0.910 | AUROC of the API judge on the same rows |
