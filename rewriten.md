# Rewrite ledger

Rewrite each paragraph and caption below in your own words. Keep every number exactly as written.
Each block gives the section and the exact drop-in position.

---

## Block A. Section 5.1 Model Comparison

Drop in immediately before `\subsection{EC-XGB and Ablation}` (after the Figure 3 float).

In addition to domain checks, the encoder selection was the last. Table 6 replaced an encoder at a time (one encoder per run), repeated the grouped splits, thirty iterations of the search procedure, and five-fold cross-validation using the same three random seeds. The base row reproduced the m2 row that was previously frozen with an F1 score difference of less than or equal to 0.0005, and an AUROC score difference of less than 0.0002. This confirmed that this particular pipeline was consistent across multiple machines. The larger NLI checkpoint improved the model by adding 0.0022 to F1 but lost 0.0008 to AUROC. The BGE embedding added 0.0002 to F1. McNemar tests were used to determine if either improvement separated from the deployed stack; both tests had p = 0.63 and p = 0.77 respectively. Therefore neither improvement provided enough value to separate from what was currently being produced. At current levels of AUROC (0.998) there is very limited room for additional improvements.

> Status: author rewrite applied to `Journal_Paper/halurisc.tex` (Block A, Section 5.1). Numbers fixed: the draft said `.022` F1 and `.008` AUROC, corrected to 0.0022 and 0.0008 (artifact values). `production stack` -> `deployed stack`. The closing sentence "The larger checkpoint is not worth the extra cost" was omitted in the author rewrite; say the word to restore it.

Caption. In-domain encoder comparison on HaluEval QA (m2, 34 features, mean over seeds 42/123/456). Delta and McNemar p are against the deployed stack.

---

## Block B. Section 5.5 Explanation, Efficiency, and Judges

Drop in immediately before the paragraph starting "LLM judges are not fair evaluators" (after the Table 17 float).

Throughput was measured on a second machine, an H200 MIG slice that has 15.9 GB of usable RAM; therefore, these measurements describe the throughput of the feature pipeline rather than the results presented earlier. The batch sweep for the deployed stack is shown in Figure 10. Batch 128 gives the maximum throughput for the deployed stack at 361 rows per second and 3.2 GB of RAM. For batch 256, the throughput is 309 rows per second at 6.1 GB, which indicates that the configured batch size has already passed the knee. The large NLI checkpoint runs at 122 rows per second and 10.3 GB, about 2.5 times the extraction time of the deployed stack. This cost buys the null result in Table 6, and it is the practical reason for keeping the base encoder in the deployed system.

> Status: author rewrite applied to `Journal_Paper/halurisc.tex` (Block B, Section 5.5). Fixed: em dash -> semicolon; typo `pipepline`; pronoun `we` removed; "2.5 times slower" -> "2.5 times the extraction time" (the artifact reports 2.5x the time). Numbers unchanged.

Caption. NLI throughput and peak memory for three encoder stacks on the H200 MIG slice, where batch size varies for the deployed stack.

---

## Block C. Section 5.5 Explanation, Efficiency, and Judges

Drop in immediately before `\subsection{Error Analysis and Discussion}` (after the Table 18 float).

Another related group of judges (a local judge) assesses how much the picture depends on the model's size. In addition to Qwen3.5-4B (Table 19), the 900 rows of Qwen3.5-9B are added to create the same set. The 9B model reaches an F1-score of 0.5957 and an AUROC of 0.7838, whereas the 4B row reaches an F1-score of 0.6012 and an AUROC of 0.7755. Therefore, the F1-scores vary by 0.005 while the AUROCs vary by 0.008, but neither result brings either size closer to the AUROC of the API judge, which reaches 0.910. The 9B model flags 24.0% of all rows at recall 0.700 and precision 0.519. Again, costs for both are $0.00. Although they are from the same slice, the 4B model runs approximately 1.5 times faster than the 9B model at the median. Since the 4B model ran on the RTX 3060 in Tables 17 and 18, this explains why the same model is faster here. An exact match with EC-XGB is only 0.274, so the judge and the feature model answer distinct questions and cannot be used interchangeably.

> Status: author rewrite applied (Block C). Fixed: leading zeros on all decimals; `image` -> `picture`; "no speedup" -> "the same model is faster here" (4B is 1.5x faster on this slice); `we can conclude` removed. Refs restored.

> Status: author rewrite applied to `Journal_Paper/halurisc.tex` (Block C, Section 5.5). Fixed wrong claim: the draft said the 4B latency was "1.5x greater than the average latency", but the artifact shows 4B is about 1.5 times FASTER at the median (976 ms vs 1466 ms). `image` -> `picture`; `lines` -> `rows`; `RTX3060` -> `RTX 3060`. Refs restored: `Table~\ref{tab:judge9b}`, `Tables~\ref{tab:cost}` and `~\ref{tab:judge}`. Numbers unchanged.

Caption. Local judges on the 900-row RAGTruth QA test split at natural prevalence, mean over three repeats, both measured on the same MIG slice.

---

## Block D. Section 5.6 Error Analysis and Discussion

Drop in immediately before `\section{Conclusion}` (after the Figure 12 float).

The external test sets, MedHallu and HaluBench, contain a different answer-length distribution from the in-domain evaluations, so a second evaluation was conducted on these test sets. In the second evaluation, answer lengths were normalised as in Table 21. Accordingly, an additional evaluation of the model on the two external test sets was conducted. For HaluBench, the number of PASS and FAIL answers by source in each length bucket remained unchanged. Only pairs of answers whose difference in length was very small were chosen for the MedHallu test set. Pairing resulted in a drop of 1.5 AUROC points for HaluBench and 4.7 for MedHallu, as well as a drop in pair accuracy from 0.8607 to 0.8290. Despite the drop in accuracy, both sources still outperformed a simple length-only baseline, at 0.588 versus 0.475 for HaluBench and 0.771 versus 0.401 for MedHallu. The performance on HaluBench was only slightly better than random chance, which suggests that transferability is far from robust however much pairing may help.

> Status: author rewrite applied (Block D). Fixed: semicolon -> period; `significantly outperformed` -> `outperformed` (no significance test was run for this comparison); `PASS/FAIL` -> `PASS and FAIL`; `difference ... were` -> `was`; `Table 21` ref restored. Numbers unchanged.

> Status: author rewrite applied (Block D). Fixed a wrong claim: the draft said the external sets "need to change", but the manuscript says they remain open because the length distributions differ. `Table 21` ref restored. Numbers unchanged.

> Status: author rewrite applied (Block D). Fixed wrong claim: the draft said pairs "differed by less than one unit in length", but the method selects pairs that differ LEAST in length. Also `has costed` -> `costs`, leading zeros, and `Table 21` refs restored. Caption unchanged from the earlier author rewrite.

> Status: author rewrite applied to `Journal_Paper/halurisc.tex` (Block D, Section 5.6). Fixed: pronoun `we` removed (impersonal); `Table 21` ref restored; `.8607` -> `0.8607`. Caption fixed: the draft said "all three test sets", but Table 21 has only two (HaluBench and MedHallu).

Caption. AUROC before and after answer-length normalisation on the two external sets, mean over seeds 42/123/456. Length-only is a logistic regression on log1p(n_words).
