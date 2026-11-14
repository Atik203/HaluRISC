# Draft text for author rewrite

This file lists every sentence added to `Journal_Paper/halurisc.tex` in this pass so it can be
rewritten in the author's own words.

Scope of this pass:

- New: Appendix A (`\section{Encoder and Length Controls}`), 4 body paragraphs, 2 table captions,
  1 figure caption.
- Changed: one preamble comment (`18 tables and 11 figures` to `20 tables and 12 figures`). This is
  a comment and does not print.
- Unchanged: every existing sentence in Sections 1 to 6, all existing tables, all existing figures.

Authorship rung C: the text below was drafted from verified artifacts. It is a draft, not finished
prose. Rewrite it in your own voice, then check that every number survives. If your institution
requires an AI-assistance disclosure, keep a record of this file as part of that record.

Numbers are traceable to `artifacts/results/h200/encoder_ablation.json`,
`artifacts/results/h200/length_matched.json`, and `artifacts/results/h200/efficiency.csv`.

---

## Appendix A. Encoder and Length Controls

### Paragraph 1 (setup)

1. The deployed model uses DeBERTa-v3-base for the NLI features and MiniLM for the cosine features.
2. Both are small checkpoints.
3. A reader may ask whether the reported results depend on that choice.
4. Two controls answer the question.
5. The first replaces one encoder at a time and repeats the full protocol on the new feature matrix.
6. The second removes the length shortcut from the external corpora.
7. The deployed model is not retrained in either control.
8. All runs reuse the grouped split, the thirty-iteration search, and the five-fold cross-validation
   with seeds 42, 123, and 456, so only the encoder changes.

### Paragraph 2 (Table 19, encoder comparison)

1. Table 19 reports the in-domain comparison.
2. The base row reproduces the frozen m2 row within 0.0005 F1 and 0.0002 AUROC, which confirms that
   a second machine follows the same pipeline.
3. A larger NLI checkpoint adds 0.0022 F1 and loses 0.0008 AUROC.
4. The BGE embedding adds 0.0002 F1.
5. McNemar tests across the three seeds return p = 0.63 and p = 0.77, so neither change separates
   from the deployed stack.
6. AUROC already sits at 0.998, and a comparison at that level has little room to show a gain.
7. The larger checkpoint is not worth the extra cost.

### Paragraph 3 (Table 20, length-matched control)

1. Table 20 repeats the two external comparisons after answer length is equalised.
2. For HaluBench, each source keeps an equal number of PASS and FAIL answers inside each length bin.
3. For MedHallu, the matched slice keeps the pairs whose two answers differ least in length.
4. Matching costs 1.5 AUROC points on HaluBench and 4.7 on MedHallu.
5. Pair accuracy falls from 0.8607 to 0.8290.
6. The model stays well above the length-only baseline in both slices, at 0.588 against 0.475 on
   HaluBench and 0.771 against 0.401 on MedHallu.
7. HaluBench is near chance in both slices, so its transfer value is weak with or without matching.

### Paragraph 4 (Figure 12, throughput and memory)

1. Figure 12 reports throughput and memory for the three encoder stacks on one H200 MIG slice with
   15.9 GB of usable memory.
2. The measurements come from a second machine, not the RTX 3060 used for the main experiments.
3. Batch 128 is the fastest point for the deployed stack at 361 rows per second and 3.2 GB of memory.
4. Batch 256 gives 309 rows per second at 6.1 GB, so the configured value is already past the knee.
5. The large NLI checkpoint runs at 122 rows per second and 10.3 GB.
6. The null result in Table 19 therefore costs about 2.5 times the extraction time, which is the
   practical argument for keeping the deployed encoder.

---

## Captions

### Table 19

In-domain encoder comparison on HaluEval QA (m2, 34 features, mean over seeds 42/123/456). Delta and
McNemar p are against the deployed stack.

### Table 20

External AUROC before and after answer-length matching, mean over seeds 42/123/456. Length-only is a
logistic regression on log1p(n_words).

### Figure 12

NLI throughput and peak memory for three encoder stacks on the H200 MIG slice, with batch size varied
for the deployed stack.

---

## Still to add

A local-judge row (Qwen3.5-9B and Qwen3.5-4B on the 900-row RAGTruth QA split) is not in the
manuscript yet. It will add one small table and about 90 words, and it will be listed here in the
same format once the run finishes.

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
