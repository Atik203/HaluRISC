# Section 4 Review: Experiments

**Manuscript:** `Journal_Paper/halurisc.tex`, lines 476-553 (`\section{Experiments}` through the end of `\subsection{Evaluation Metrics}`)
**Review date:** 11 November 2026
**Reviewer role:** adversarial journal reviewer, Section 4 only

---

## Verdict

**Major revision.**

The section is compact and most of its numbers are correct. I verified the HaluEval split, the RAGTruth counts, the pooled test size, all five positive rates, the 26+8+1 feature count, the seed list, the installed package versions, and the seed-42 hyperparameters against the artifacts. Those all check out.

The revision is major because the section misdescribes several things a reader needs in order to interpret the results:

1. Three evaluation sets are presented as if they were complete corpora when they are samples (HaluBench 900 of 3,000 rows, HaluEval dialogue/summarization 1,000 of 10,000 pairs each).
2. FaithBench labels are called "native" when they are severity annotations reduced by a worst-severity rule.
3. RAGTruth is described as if it received a 70/15/15 split, but it uses the official document-level split.
4. The latency benchmark is described as "200 test rows", but the checked-in script times the first 200 rows of `qa_clean.parquet`, which are 112 train, 40 validation, and 48 test rows.
5. The judge timing is described as "the same rows", but the judge runs on the 900 RAGTruth QA rows, not the 200 feature-latency rows.
6. NLL is defined and its precision is set, but no table or sentence reports it.

None of these requires a new modeling experiment. They require accurate description, paragraph splits, and about 100 words of cuts.

---

## Strengths

- **The core dataset arithmetic and positive rates are right.** HaluEval 20,000 rows / 10,000 items / 14,000-3,000-3,000; RAGTruth QA 5,934 rows / 989 groups / 900 test / 5,034 calibration; RAGTruth full 17,790 / 2,965; pooled test 2,700. The rates are 50.0% HaluEval, 34.93% pooled RAGTruth, 17.78% RAGTruth QA test, 31.07% QA calibration, and 68.13% FaithBench. I recomputed each from `data/processed/unified_records.parquet` and `artifacts/split_indices.json`; all match, and the paper rounds them correctly.
- **The feature count is right.** `FEATURE_GROUPS` has 26 columns, `claim_features.FEATURE_COLUMNS` has 8, and `SOURCE_COLUMN` adds 1, for 35. The m0/m1/m2/m3 feature counts of 26/26/34/35 are consistent.
- **The protocol table is clear and useful.** P0/P1/P2 separate the calibrator, the threshold, and the zero-shot status cleanly, and the note correctly states that P1/P2 read target labels and that the 900 test rows are disjoint from the 5,034 calibration rows.
- **The baseline set is well chosen for the claim.** The heuristic, length-only control, 26-feature logistic and random forest, three feature controls, RoBERTa, external ensemble, majority, and two judges cover the obvious alternative explanations. The length-only control is exactly the right artifact-control baseline for HaluEval.
- **Reproducibility is above average.** Pinned versions match the installed `.venv`, the seed-42 conventional configuration matches `artifacts/results/b2/b2_tuning.json`, the 30-iteration randomized search matches `n_iter`, and the repository ships split indices, download revisions, and SHA-256 hashes.

---

## Issues

Severity key: **Major** = a claim is wrong, unsupported, or cannot be reproduced from the text; **Minor** = clarity, precision, style, or text weight. Exact replacements for Major issues are in the next section.

| ID | Severity | Line | Location (quote) | Issue | Why it matters | Fix |
|---|---|---|---|---|---|---|
| MAJ-1 | Major | 510 | "RAGTruth span labels are mapped to binary answer labels, while HaluEval and FaithBench use their native labels." | FaithBench does not use a native binary label. `src/data/mappings.py` aggregates severity strings (`Benign`, `Questionable`, `Unwanted`, `Unwanted.Intrinsic`, `Unwanted.Extrinsic`) by worst severity, and `Questionable` counts as positive. | The 68.1% positive rate, the FaithBench F1 of 0.120, and every FaithBench transfer claim depend on this mapping. Calling it "native" hides a modeling choice. | Replace with the three-sentence version in MAJ-1 below. |
| MAJ-2 | Major | 486 | "HaluBench \cite{ravi2024lynx} gives 900 rows from DROP, FinanceBench, and covidQA, 300 each at 50.0\% positive" | 900 is a stratified 30% sample of 3,000 non-overlapping rows (150 PASS + 150 FAIL per source), not the corpus. | A reader cannot reconstruct the 900-row set from the original HaluBench release. The 50.0% rate is a sampling choice, not a corpus property. | Replace with the sample wording in MAJ-2 below. |
| MAJ-3 | Major | 486 | "HaluEval dialogue and summarization give 1,000 pairs each (2,000 rows per task, 50.0\% positive)" | Each HaluEval task has 10,000 items. The paper evaluates on a seeded 1,000-pair sample, not on the full subset. | The table's 2,000 rows look like the corpus size. The result is not reproducible without the sampling seed and rule. | Replace with the seeded-sample wording in MAJ-3 below. |
| MAJ-4 | Major | 510 | "HaluEval groups by item and RAGTruth by source document; the 70/15/15 indices are frozen." | RAGTruth uses its official document-level split (15,090 train / 2,700 test), not a 70/15/15 split. | The sentence conflates the two split protocols. A reader may try to reproduce a 70/15/15 RAGTruth split that does not exist. | Replace with the split-specific wording in MAJ-4 below. |
| MAJ-5 | Major | 510 | "Answer lengths are log1p-transformed." | Only m1 and later variants log-transform `n_chars` and `n_words`. m0 and the conventional B2 model keep raw counts, as the Table 6 note itself says. | The preprocessing paragraph contradicts the ablation table and misdescribes the conventional baseline. | Replace with the variant-specific wording in MAJ-5 below. |
| MAJ-6 | Major | 510 | "Claims are split into at most six atomic clauses and matched to the four best context sentences with a 400-character pair limit." | The four-sentence selection is per claim and only when the context has more than four sentences; short contexts use every sentence. The 400-character cap applies to each side of the pair, so a pair can be 800 characters. "Atomic" overstates a regex splitter that also splits on commas, semicolons, colons, and conjunctions. | The claim features and the deployed verifier depend on this pairing. A reader cannot tell whether four sentences are selected per answer or per claim, or what the 400-character limit applies to. | Replace with the precise wording in MAJ-6 below. |
| MAJ-7 | Major | 510 | "Missing context is a special condition: lexical overlap is 0 and NLI probabilities are uniform." | No sentence describes rows without a question. Every FaithBench row and every RAGTruth summarization/data-to-text row has an empty question. `overlap_answer_question` is 0 there, but `cosine_q_ans` is the cosine between the answer and the encoding of an empty string, which ranges from -0.10 to 0.24 for the 12,606 rows without a question after joining `external_features.parquet` to the record text. | This affects 750 FaithBench rows, 10,056 RAGTruth training rows, and 1,800 RAGTruth non-QA test rows. The feature has a different meaning on the multi-source training rows and on the external tasks, which matters for the transfer claims. | Replace with the wording in MAJ-7 below, or set the feature to 0 for empty questions and re-extract. |
| MAJ-8 | Major | 510 | "External corpora use the same extractors, with the MedHallu source indicator set to 1 and HaluBench to 0." | The reported runs also set FaithBench to 1 and the HaluEval task-shift sets to 0. Two values are omitted. The feature is described in Section 3.1 as marking a "natural response corpus", but FaithBench is model-generated summarization and still receives 1. | The indicator is a model input. A reader cannot reproduce the external scores without its value for every corpus, and the current definition does not explain why FaithBench is treated as natural. | Replace with the complete value list in MAJ-8 below, and define what the indicator marks. |
| MAJ-9 | Major | 514, 546, 551 | "Metrics are reported to four decimals, NLL to two, and rates to one." / "Negative log-likelihood rewards confident correct probabilities and penalizes confident wrong ones, and lower is better." | NLL is defined in prose and in the `gather` block, and its precision is set, but no table or sentence reports NLL. Accuracy is also discussed but never reported. | The section promises two metrics that do not appear anywhere in the paper. This is exactly the kind of methods-results mismatch a reviewer checks. | Delete NLL and accuracy text, or add an NLL column to Table 5. See MAJ-9 below. |
| MAJ-10 | Major | 514 | "Twelve models are tested. ... Three controls isolate the feature design: NLI-only, answer-only TF-IDF, and full-input TF-IDF." | The enumeration names 10 baselines plus EC-XGB, which is 11. The twelfth row in Table 5 is the length-only control, which is not introduced in the prose; it is defined only in the Table 5 note. | The model count does not match the prose, and a table row appears without a definition. | Replace with the length-only introduction in MAJ-10 below. |
| MAJ-11 | Major | 533 | "Latency is measured on 200 test rows on the reference laptop. The same rows are timed with GPT 5.6 Luna under the LLM-as-a-judge protocol \cite{zheng2023judge}" | `src/models/rebuild_latency.py` defaults to `data/processed/qa_clean.parquet` and takes `head(200)`. Those rows are 112 train, 40 validation, and 48 test. The judge script loads the 900 RAGTruth QA test rows, not the 200 HaluEval rows. | Both sentences misstate the benchmark. The 127.5 ms figure is not a test-only measurement, and the judge cost/latency is not on the same rows as the feature latency. | Replace with the accurate protocol in MAJ-11 below, or re-run the feature latency on 200 test rows. |
| MAJ-12 | Major | 510 | "Feature extraction runs on the reference GPU with fp16 NLI and SBERT; XGBoost inference is device-agnostic." | Core and entity features run on the CPU; only NLI and SBERT use GPU fp16. The reported m3 booster was trained with `grow_gpu_hist` and records `device: cuda:0` in its config, so "device-agnostic" is not supported for a CPU-only deployment. | The first clause overstates the hardware path. The second is a portability claim that the saved artifact does not demonstrate, and the `xgb_device()` note in `train_pipeline.py` warns that CUDA-trained boosters do not port across platforms. | Replace with the accurate hardware sentence in MAJ-12 below. |
| MIN-1 | Minor | 486 | "Three further corpora test external validity without fitting or threshold selection." | The paragraph names MedHallu, HaluBench, and the two HaluEval task-shift sets (four datasets). Section 5.3 counts "three external corpora and the task-shift pair." | The count conflicts with the Results section and with the table note's "five zero-shot corpora." | "Two further corpora and a task-shift pair test external validity without fitting or threshold selection." |
| MIN-2 | Minor | 484 | "About 85\% of HaluEval was generated by ChatGPT \cite{haluEval,bang-etal-2025-hallulens}" | I could not verify 85% from the repo artifacts or a cited table. HaluEval's own description says the benchmark samples are ChatGPT-generated. | An unverified percentage in a dataset description invites a reviewer query and weakens trust in the other numbers. | Cite the exact HaluLens table/section for 85%, or delete the percentage and keep the citation. |
| MIN-3 | Minor | 546 | "The second group measures ranking, calibration and paired error patterns." | MCC is threshold-dependent, NLL is a proper scoring rule, Wilson is an interval, and "paired error patterns" is not among the listed metrics. ECE and Brier are reported in the model and calibration tables but defined only in Section 3.4. | The taxonomy is inaccurate and the two most important calibration metrics are absent from the Evaluation Metrics subsection. | "The second group covers ranking, probability quality, and interval estimation. ECE and the Brier score, defined in Section 3.4, measure probability quality." |
| MIN-4 | Minor | 537 | "It is the operating constraint under distribution shift, where most natural answers are correct." | False for FaithBench, where 68.1% of rows are positive. It is true for RAGTruth QA, where 82.2% are correct. | The sentence generalizes a QA-specific prevalence to all shifted corpora. | "It is the operating constraint on RAGTruth QA, where 82.2\% of the test answers are correct." |
| MIN-5 | Minor | 514 | "No question appears in both training and evaluation." | The checked guarantee is stronger and narrower: HaluEval items are grouped, and RAGTruth source documents are disjoint. Cross-corpus duplicates are not checked. | The global wording claims more than the evidence supports. | "No HaluEval item and no RAGTruth source document are shared between training and evaluation." |
| MIN-6 | Minor | 533 | "P1 fits the calibrator on the target split and uses 0.5; P2 also selects the threshold on the target split at 5\% FPR." | "uses 0.5" is telegraphic, and the "also" is ambiguous because P1 also fits a calibrator. | A reader has to infer that 0.5 is the threshold. | "P1 fits the calibrator on the target split and uses a fixed 0.5 threshold. P2 fits the same calibrator and selects the threshold on the target split at a 5\% FPR budget." |
| MIN-7 | Minor | 533 | "The test rows are unseen at calibration time, and the deployed system uses P1; P2 is the thresholded alternative." | "uses P1" is approximate: the API uses the P1 calibrator but applies separate risk bands (0.35/0.60), and Section 5.3 says the deployed card relies on per-claim adjustment on RAGTruth. | A reader may think the deployment threshold is 0.5. | "The test rows are unseen at calibration time. The deployed system uses the P1 calibrator, and P2 is the thresholded alternative." |
| MIN-8 | Minor | 510 | "Licenses are HaluEval, RAGTruth, MedHallu, and the HaluEval dialogue and summarization subsets (MIT); FaithBench (CC BY-NC-SA 4.0); and HaluBench (CC BY-NC 2.0). Raw data is not redistributed; the repository ships download scripts, source revisions, and SHA-256 hashes." | "Licenses are HaluEval" is ungrammatical, the semicolons are a mix of clause joins and list separators, and the content duplicates the Declarations. | It reads like a checklist and spends about 50 words on material already in the Data availability statement. | Shorten to: "Licenses and the data-availability statement are in the Declarations." If detail stays, use: "HaluEval, RAGTruth, and MedHallu are MIT licensed. FaithBench is CC BY-NC-SA 4.0, and HaluBench is CC BY-NC 2.0. Raw data is not redistributed. The repository ships download scripts, source revisions, and SHA-256 hashes." |
| MIN-9 | Minor | 510 | "All text is whitespace-normalized." | The checked-in pipeline trims leading and trailing whitespace (`prepare.py`, `prepare_unified.py`); I found no code that collapses internal whitespace runs. | A small reproducibility claim is stronger than the implementation. | "Leading and trailing whitespace is trimmed, and lexical matching is case-folded." |
| MIN-10 | Minor | 510 | "NER and NLI use the original case." | NER reads the original text, but `entity_features.py` lowercases entity strings before matching, so entity overlap is case-insensitive. | The sentence is half true and hides the matching rule. | "NER and NLI read the original text, and entity strings are lowercased for matching." |
| MIN-11 | Minor | 510 | "The logistic baselines standardize features on their fit split, while XGBoost does not scale." | Only the feature-based logistic models are standardized. The TF-IDF logistic models use sparse features and are not standardized. | The plural suggests all logistic models, including the TF-IDF controls. | "The feature-based logistic baselines standardize their inputs on the training split. The TF-IDF models use sparse features, and XGBoost does not scale." |
| MIN-12 | Minor | 480 | "A randomized search was performed over 30 hyperparameter configurations with grouped five-fold cross-validation." | The search space is not reported. Only the seed-42 winner is given, although each seed/variant is tuned separately. | A reader cannot reproduce the tuning from the text. The artifact has the grid, but the paper does not. | Add one sentence: "Each tuned run used 30 randomized configurations under grouped five-fold cross-validation, varying depth 3--7, learning rate 0.01--0.2, 100--500 trees, subsample 0.7--1.0, and column sampling 0.7--1.0." |
| MIN-13 | Minor | 480 | "Each seed is tuned separately for the main comparison, and the component ablation ... holds one configuration fixed across variants so that the feature effects are not confounded with tuning." | "Each seed" could include the non-XGBoost baselines, which are not tuned. The sentence is also long. | Ambiguity about what is tuned. | "Each XGBoost seed is tuned separately for the main comparison. The component ablation ... holds one configuration fixed across variants, so feature effects are not confounded with tuning." |
| MIN-14 | Minor | 480 | "Each run can be reproduced because of saved split indices and centralized seeds." | Awkward causal phrasing. | Small style issue. | "Saved split indices and fixed seeds make each run reproducible." |
| MIN-15 | Minor | 500-501 | "MedHallu & 2{,}000 & 1{,}000 pairs & zero-shot external" / "HaluBench subset & 900 & 3 sources & zero-shot external" | The "Groups" column holds text descriptions, not group counts. HaluBench has 900 independent rows, not 3 groups. | The table mixes units and makes the group counts hard to compare. | Use 1,000 and 900 in the Groups column, and put "pairs" and "three sources" in the Use column or the note. |
| MIN-16 | Minor | 510/514/533 | The section uses 16 semicolons in prose, and 14 of them join independent clauses, for example "Lexical features use lowercased token sets; NER and NLI use the original case" and "Logistic regression and a 300-tree random forest are the classical comparators; the conventional XGBoost model ...". | `docs/prompt.md` bans semicolons outside complex lists. | Style guide violation; also makes the prose feel list-like. | Split each into two sentences, as shown in the exact replacements. |
| MIN-17 | Minor | 476-553 | No paragraph is shorter than 100 words; the shortest is 105 words / 5 sentences. | `docs/prompt.md` asks for at least one 2-sentence paragraph per section and varied paragraph length. | The section reads as a continuous block of fact-dense paragraphs, which lowers burstiness and readability. | Split the preprocessing paragraph and the baselines paragraph; the licenses/data-availability sentence can become a 2-sentence paragraph or move to Declarations. |
| MIN-18 | Minor | 478-553 | The G9 sequence "Experimental Procedure" and "Parameter Configuration" has no separate heading; the protocol table sits inside "Baselines and Experimental Procedure". | The content is present, but the heading compresses three G9 slots into one. | A reviewer following the journal's expected 4.1-4.5 structure may see a missing subsection. | Either split 4.3 into "Baselines" and "Experimental Procedure" or record the deviation in the writing-guideline mapping. |

---

## Exact replacements for Major issues

### MAJ-1: FaithBench label mapping

**Old (line 510):**
> RAGTruth span labels are mapped to binary answer labels, while HaluEval and FaithBench use their native labels.

**New:**
> A RAGTruth answer is positive when it carries at least one annotated hallucination span. FaithBench severity annotations are reduced to a binary label by a worst-severity rule, and a Questionable annotation counts as positive. HaluEval uses its native label.

### MAJ-2: HaluBench is a sample

**Old (line 486):**
> HaluBench \cite{ravi2024lynx} gives 900 rows from DROP, FinanceBench, and covidQA, 300 each at 50.0\% positive; its HaluEval, RAGTruth, and PubMedQA rows are excluded.

**New:**
> HaluBench \cite{ravi2024lynx} contributes a stratified 30\% sample of its DROP, FinanceBench, and covidQA rows: 900 rows, 300 per source, balanced at 150 PASS and 150 FAIL. Its HaluEval, RAGTruth, and PubMedQA rows are excluded.

### MAJ-3: HaluEval task-shift sets are samples

**Old (line 486):**
> HaluEval dialogue and summarization give 1,000 pairs each (2,000 rows per task, 50.0\% positive) as a task-shift test, not a second independent QA corpus.

**New:**
> A seeded sample of 1,000 pairs from each HaluEval task gives 2,000 rows per task at 50.0\% positive. This is a task-shift test, not a second independent QA corpus.

### MAJ-4: Split protocol

**Old (line 510):**
> HaluEval groups by item and RAGTruth by source document; the 70/15/15 indices are frozen.

**New:**
> HaluEval is split 70/15/15 by question item, RAGTruth keeps its official document-level split, and both index sets are frozen.

### MAJ-5: log1p scope

**Old (line 510):**
> Answer lengths are log1p-transformed.

**New:**
> In m1 and later variants, the two length counts are log1p-transformed; m0 and the conventional model keep raw counts.

### MAJ-6: Claim, context, and pair limits

**Old (line 510):**
> Claims are split into at most six atomic clauses and matched to the four best context sentences with a 400-character pair limit.

**New:**
> Claims are split into at most six clause-level segments. When the context has more than four sentences, each claim is matched to its four best sentences by token overlap, and each side of a pair is truncated to 400 characters.

### MAJ-7: Empty questions

**Old (line 510):**
> Missing context is a special condition: lexical overlap is 0 and NLI probabilities are uniform.

**New (describes the current code):**
> A missing context is a special case: lexical overlap is 0 and NLI probabilities are uniform. FaithBench and RAGTruth summarization and data-to-text have no question, so question overlap is 0 and question similarity is the cosine between the answer and the encoded empty string.

**Preferred alternative if the authors change the extractor:** set `cosine_q_ans` to 0 for empty questions, re-extract, and write "question overlap and question similarity are set to 0" instead. The current behavior is defensible only if it is documented.

### MAJ-8: Source indicator values

**Old (line 510):**
> External corpora use the same extractors, with the MedHallu source indicator set to 1 and HaluBench to 0.

**New:**
> External corpora use the same extractors. The source indicator is 1 for FaithBench and MedHallu and 0 for HaluBench and the task-shift sets; Section~\ref{sec:results} reports a sensitivity check with the MedHallu value set to 0.

**Also add one sentence defining the feature**, because "natural response" does not explain why FaithBench, a model-generated summarization corpus, receives 1. For example: "The indicator marks rows from the RAGTruth-style external corpora rather than HaluEval-style benchmark rows." Use whatever definition the authors intend, but make it match the values above.

### MAJ-9: NLL and accuracy are never reported

**Old (line 514):**
> Metrics are reported to four decimals, NLL to two, and rates to one.

**New:**
> Metrics are reported to four decimals and rates to one.

**Delete (line 546):**
> Negative log-likelihood rewards confident correct probabilities and penalizes confident wrong ones, and lower is better.

**Delete the NLL line in the `gather` block (line 551).**

If the authors prefer to keep NLL, they must add it to Table 5 or to a calibration table and report it to two decimals. The current text defines a metric that appears nowhere.

### MAJ-10: Length-only baseline and the twelve-model count

**Old (line 514):**
> Twelve models are tested. The heuristic sets risk to one minus lexical overlap. Logistic regression and a 300-tree random forest are the classical comparators; the conventional XGBoost model \cite{xgboost} uses the 26 base features. Three controls isolate the feature design: NLI-only, answer-only TF-IDF, and full-input TF-IDF.

**New:**
> Twelve models are tested. The heuristic sets risk to one minus lexical overlap, and a length-only logistic control uses the two log-scaled length counts. Logistic regression and a 300-tree random forest are the classical comparators. The conventional XGBoost model \cite{xgboost} uses the 26 base features. Three further controls isolate the feature design: NLI-only, answer-only TF-IDF, and full-input TF-IDF.

### MAJ-11: Latency and judge protocol

**Old (line 533):**
> Latency is measured on 200 test rows on the reference laptop. The same rows are timed with GPT 5.6 Luna under the LLM-as-a-judge protocol \cite{zheng2023judge}, and costs are computed from the recorded token counts.

**New:**
> Feature latency is measured on 200 HaluEval rows on the reference laptop. The 900 RAGTruth QA rows are scored by two judges, GPT-5.6 Luna and Qwen 3.5 4B, under the LLM-as-a-judge protocol \cite{zheng2023judge}, and costs are computed from the recorded token counts.

If the authors want the stronger "200 test rows" claim, re-run `rebuild_latency.py` with a test-only input and record the input path in the artifact. The current `m3_latency.json` does not record which file was used.

### MAJ-12: Hardware and XGBoost portability

**Old (line 510):**
> Feature extraction runs on the reference GPU with fp16 NLI and SBERT; XGBoost inference is device-agnostic.

**New:**
> NLI and SBERT inference ran on the reference GPU in fp16, while the lexical and entity features run on the CPU. The deployed classifier is a 35-feature tree model.

If the authors want to claim portability, add the evidence: "The released boosters were re-saved with the CPU histogram method and load on a CPU-only host." Do not claim device-agnostic inference for the current CUDA-trained m3 booster without that check.

---

## Text-weight and readability assessment

**Overall:** about 1,240 words of prose plus about 260 words of table and equation text. The section is not bloated for an 18-page paper. The problem is density, not total length.

| Block | Words | Sentences | Assessment |
|---|---:|---:|---|
| 4.1 Setup | 168 | 7 | Fine. One config-dump sentence is 37 words; acceptable but could be split. |
| 4.2 Dataset, paragraph 1 | 175 | 9 | Borderline. The five-rate sentence is the part to move to the table. |
| 4.2 Dataset, paragraph 2 | 105 | 5 | Fine. The "three further corpora" count is the only issue. |
| 4.2 Preprocessing | 178 | 12 | Too dense. Twelve consecutive facts with almost no short sentence. Split into two paragraphs. |
| 4.3 Baselines and procedure | 169 | 12 | Overloaded. It contains the model list, tuning, calibration, significance, and protocols. Split before "EC-XGB's display calibrator...". |
| 4.3 Protocol paragraph | 106 | 6 | Repeats Table 4. Cut to two sentences after the table. |
| 4.4 Metrics, paragraph 1 | 181 | 12 | Over-explained. Precision/recall/F1 get interpreted twice. Cut about 40 words. |
| 4.4 Metrics, paragraph 2 | 158 | 9 | Over-explained and contains two dead metrics (accuracy, NLL). Cut about 45 words. |

**Is the dataset paragraph still too long?** Not in absolute terms, but it carries too many numbers for one paragraph. The five positive rates in one sentence are hard to parse. Move them to Table 3's note and keep only the 17.8% QA test rate in 4.4, where it justifies PR-AUC. That removes one dense sentence and one duplication.

**Is the metrics section over-explained?** Yes. It is 339 words and 21 sentences for eleven metrics, two of which are never reported. Sentences that can be cut without losing a definition:

- "The confusion matrix supplies the four counts used below, TP, FP, FN, and TN."
- ", and a high value means a reviewer can trust an alert"
- ", and a high value means few bad answers pass"
- ", and it punishes a model that is strong on one metric and weak on the other"
- "Accuracy is misleading on an imbalanced test set, so MCC is the safer single-number summary."
- "Negative log-likelihood rewards confident correct probabilities and penalizes confident wrong ones, and lower is better."
- "A flag rate near 100\% means the detector sends every answer to a human."
- ", and it is more sensitive than AUROC when the positive class is rare"

That is about 100 words, or roughly 8% of the section's prose.

**Other cuts:**

- The protocol paragraph after Table 4 repeats the table. Keep only: "The test rows are unseen at calibration time. The deployed system uses the P1 calibrator, and P2 is the thresholded alternative."
- The license and raw-data sentences duplicate the Declarations. Replace with "Licenses and the data-availability statement are in the Declarations." Saves about 40 words.
- The "Each seed is tuned separately" sentence and the "Seeds 42, 123, and 456 are repeated" sentence can be merged: "Each XGBoost seed is tuned separately, and seeds 42, 123, and 456 are repeated where a seed applies."

**Paragraph rhythm:** Every paragraph is 105-181 words. There is no 2-sentence paragraph. The preprocessing paragraph has 12 sentences of similar length, which is the least human-sounding part of the section. Splitting it after the split/label sentence and making the license/availability sentence its own short paragraph would fix both the rhythm and the density.

**Flow:** The order dataset -> baselines -> protocol -> metrics is logical and matches the content G9 asks for. The two flow problems are (1) the preprocessing paragraph mixes feature prep, splits, labels, hardware, scaling, external values, licenses, and availability, and (2) 4.3 mixes baselines, tuning, calibration, significance, protocols, SHAP, and latency. Both need paragraph breaks, not reordering.

---

## Preprocessing and completeness checklist

Status key: **Pass** = described and verified; **Partial** = described but imprecise or incomplete; **Missing** = not described; **Incorrect** = described differently from the code.

| Step | Claimed in Section 4? | Verified in code/artifacts? | Status | Note |
|---|---|---|---|---|
| Whitespace normalization | "All text is whitespace-normalized" | Only `.strip()` in `prepare.py` and `prepare_unified.py`; no internal whitespace collapse found | Partial | Trim, not full normalization. |
| Lowercasing | "Lexical features use lowercased token sets" | `.lower()` in lexical, hedging, and entity matching; NLI raw | Pass | Numeric regex is case-insensitive by construction. Entity matching also lowercases. |
| NER case | "NER ... use the original case" | NER input raw; entity strings lowercased for matching | Partial | Case is preserved for the model, not for matching. |
| NLI case | "NLI use the original case" | Raw context/answer passed to the cross-encoder | Pass | Correct. |
| Missing context | "lexical overlap is 0 and NLI probabilities are uniform" | `_neutral_row()` returns 1/3 each; lexical returns 0 | Pass | Correct. |
| Missing question | Not mentioned | `cosine_q_ans` encodes the empty string; `overlap_answer_question` is 0 | Missing | Affects FaithBench and RAGTruth non-QA rows. |
| Answer-length transform | "Answer lengths are log1p-transformed" | `feature_matrix(..., log_lengths=variant != "m0")`; B2 raw | Incorrect | Applies to m1+ only. |
| Claim splitting | "at most six atomic clauses" | `MAX_CLAIMS=6`; regex splitter on punctuation and conjunctions | Partial | "Atomic" overclaims a heuristic splitter. |
| Context sentence limit | "four best context sentences" | `MAX_CONTEXT_SENTENCES=4`; top-k per claim only when `len(sentences) > 4` | Partial | Per claim, not per answer; short contexts use all sentences. |
| Pair character limit | "400-character pair limit" | `sentence[:400]`, `claim[:400]` | Partial | 400 per side, so a pair can be 800 characters. |
| Grouped splits | "HaluEval groups by item and RAGTruth by source document; the 70/15/15 indices are frozen" | HaluEval item-level 70/15/15; RAGTruth official source-level split | Partial | Conflates the two protocols. |
| HaluEval label mapping | "native labels" | correct=0, hallucinated=1 | Pass | Native binary labels. |
| RAGTruth label mapping | "mapped to binary answer labels" | Any annotated span -> 1, none -> 0 | Partial | The rule is not stated. |
| FaithBench label mapping | "native labels" | Worst-severity aggregation; `Questionable` counts positive | Incorrect | Not native binary labels. |
| Feature extraction precision | "reference GPU with fp16 NLI and SBERT" | NLI/SBERT fp16 on CUDA; core/entity on CPU | Partial | Hardware path is broader than the code. |
| Scaling | "logistic baselines standardize features ... XGBoost does not scale" | Feature LR scaled; TF-IDF LR not; XGBoost not | Partial | "Logistic baselines" is too broad. |
| Source indicator | "MedHallu 1 and HaluBench 0" | FaithBench 1, MedHallu 1, HaluBench 0, task-shift 0 | Partial | Two values and the feature definition are missing. |
| Licenses | Full sentence | Manifest supports HaluEval/RAGTruth/FaithBench; external artifacts support MedHallu/HaluBench | Pass with style issues | Ungrammatical and duplicates Declarations. |
| Data availability | "Raw data is not redistributed; ... download scripts, source revisions, SHA-256 hashes" | `dataset_license_manifest.json` has revisions and hashes | Pass | Duplicates Declarations. |
| Deduplication / decontamination | Not mentioned | HaluBench excludes overlapping sources; no cross-corpus near-duplicate check | Missing | The "no question overlap" claim is split-level, not corpus-level. |
| Empty answers / invalid spans | Not mentioned | Build filters empty answers and counts invalid spans | Missing | Low priority, but easy to state. |
| TF-IDF settings | Not mentioned | `ngram_range=(1,2)`, `min_df=2`, `max_features=100000`, `sublinear_tf=True` | Missing | A reviewer will ask for at least the n-gram range and vocabulary cap. |
| Random forest settings | "300-tree random forest" | `n_estimators=300`, `min_samples_leaf=5` | Partial | Leaf size missing. |
| RoBERTa settings | "up to 256 tokens, three epochs" | LR 2e-5, AdamW, weight decay 0.01, warmup 10%, batch/accum in code | Partial | Learning rate and batch size missing. |
| XGBoost device portability | "device-agnostic" | m3 booster trained with `grow_gpu_hist`, `device: cuda:0` | Incorrect | Unsupported for CPU-only deployment. |

---

## Top 3 fixes

1. **Correct the dataset and label description first.** FaithBench's severity mapping, HaluBench's 30% sample, the HaluEval task-shift samples, and RAGTruth's official split are the claims a reviewer will check against the original dataset releases. If these stay wrong, the external and task-shift results look better documented than they are.

2. **Rewrite the preprocessing paragraph and split it in two.** State the m1+ log transform, the per-claim/four-sentence/per-side truncation, the empty-question behavior, and all four source-indicator values. Move licenses and data availability to the Declarations or a one-line cross-reference. This is the reproducibility core of the section.

3. **Fix the protocol and metrics text.** Replace "200 test rows" and "the same rows" with the actual latency and judge protocols, introduce the length-only baseline so the twelve-model count is true, and delete the NLL/accuracy text or report NLL. Then cut the remaining metric hand-holding. This removes two false claims and about 100 words.

---

## What I verified

- `artifacts/split_indices.json`: `n_train=14000`, `n_val=3000`, `n_test=3000`, seed 42.
- `data/processed/unified_records.parquet`: HaluEval 20,000 / 10,000 groups / 50.0% positive; RAGTruth 17,790 / 2,965 groups; RAGTruth QA 5,934 / 989 groups; QA test 900 / 150 groups / 17.78% positive; QA calibration 5,034 / 31.07% positive; pooled test 2,700 / 34.93% positive; FaithBench 750 / 68.13% positive.
- `data/processed/medhallu_records.parquet`: 2,000 rows, 1,000 positive.
- `data/processed/halubench_subset.parquet`: 900 rows, 450 positive, 300 per source.
- `artifacts/results/rebuild/p1/task_shift.json`: 1,000 pairs per task, 2,000 rows per task, `sample_seed=42`.
- `artifacts/results/rebuild/p1/halubench.json`: 150 PASS + 150 FAIL per source, `source_indicator=0.0`.
- `src/features/claim_features.py`: `MAX_CLAIMS=6`, `MAX_CONTEXT_SENTENCES=4`, `MAX_PAIR_CHARS=400`, per-side truncation.
- `src/models/run_b6_modified.py`: `FEATURE_GROUPS` totals 26, `CLAIM_COLUMNS` totals 8, `SOURCE_COLUMN` adds 1; `log_lengths = variant != "m0"`.
- `artifacts/results/b2/b2_tuning.json`: seed-42 params depth 5, lr 0.05, 300 trees, subsample 0.7, colsample 0.7.
- `TUNING_GRID`: depth 3-7, lr 0.01/0.05/0.1/0.2, trees 100/200/300/500, subsample 0.7-1.0, colsample 0.7/0.9/1.0.
- Installed versions: Python 3.12, scikit-learn 1.9.0, XGBoost 3.3.0, SHAP 0.52.0, spaCy 3.8.14, sentence-transformers 5.6.1, torch 2.11.0+cu128.
- `src/models/rebuild_latency.py`: default input `data/processed/qa_clean.parquet`, `head(200)`; those rows are 112 train / 40 validation / 48 test.
- `src/models/rebuild_judge.py`: judge evaluation on the 900 RAGTruth QA test rows.
- `artifacts/models/b6/xgboost_m3_seed_42.joblib`: booster config `grow_gpu_hist`, `device: cuda:0`.
- `artifacts/results/b6/b6_external_metrics.csv`: m3 FaithBench AUROC 0.594, F1 0.120, flag rate 7.3%, matching Table 11 and consistent with source indicator 1.

---

## Summary

**Major revision.** The section's arithmetic is mostly reliable, and the protocol table and baseline set are strong. The problems are descriptive: three evaluation sets are samples presented as corpora, FaithBench's label mapping is called native when it is an aggregated severity rule, RAGTruth's split is misdescribed, the latency and judge protocols are wrong, and NLL is defined but never reported. The preprocessing paragraph also omits the empty-question behavior and two source-indicator values, and it contradicts the ablation table on the log transform. Fix the dataset/label paragraph and the preprocessing paragraph first, then cut the dead metrics and the duplicated protocol text. No new modeling experiment is required. A latency re-run is needed only if the authors keep the "200 test rows" claim.
