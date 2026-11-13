# AI Detect Text — Turnitin Flagged Paragraphs for Rewrite

Source: `halurisc_AI.pdf` (Turnitin, Submission ID `trn:oid:::22779:158271170`)
Manuscript: `Journal_Paper/halurisc.tex`
Overall: **34% detected as AI**. Blue highlight in PDF = rewrite target.

## How to use (paragraph mode)

1. Rewrite each `PARA` block below **as a full paragraph in your own words**.
2. Return it as `PARA-ID: your rewritten paragraph`.
3. I split your paragraph into sentences and update `Journal_Paper/halurisc.tex` sentence-by-sentence.
4. On update I only fix: numbers, claims, citations, LaTeX, wrong words. I do **not** add `we / our / ours / us`. Formal sections stay impersonal per `docs/prompt.md` and `docs/writing-guideline.md`.
5. Do **not** change metric values, ECE/Brier/AUROC/F1, row counts, thresholds, citations, or conclusions.
6. Captions are in `CAP` section — rewrite those too, keep all `n =`, `%`, dataset names exact.

Blocks marked ✅ APPLIED are already patched into `halurisc.tex` — text below is what is now in the manuscript. All PARA blocks are applied; only `CAP` captions are pending.

## Priority

| Priority | Section | tex location | Est. flagged words | Share of doc (14,605) | Share of Turnitin 34% (~4,966 words) |
|---|---|---|---|---|---|
| P0 — first ✅ | 5.3 Calibration, Shift, and External Validity | `Journal_Paper/halurisc.tex:638-799` | ~1,264 | ~8.65% | ~25.5% |
| P1 ✅ | Proposed block 3.x + Intro tail (INT1, INT2, 31, 32, 33, 34, 35A, 35B) | `Journal_Paper/halurisc.tex:294-296,353-464` | ~1,077 | ~7.37% | ~21.7% |
| P2 | 5.1 + 5.2 + 5.4 (comparison, ablation, stats) | `Journal_Paper/halurisc.tex:542-542,584-625,802-804` | ~565 | ~3.87% | ~11.4% |
| P3 ✅ | 5.5 Explanation, Efficiency, and Judges | `Journal_Paper/halurisc.tex:844-943` | ~536 | ~3.67% | ~10.8% |
| P4 ✅ | 4.2 Dataset | `Journal_Paper/halurisc.tex:472-496` | ~455 | ~3.12% | ~9.2% |
| P5 | Rest (4.4, 5.6, Declarations) | see blocks | ~277 | ~1.90% | ~5.6% |

Method: word counts run from the verbatim PARA blocks against Turnitin totals (14,605 words, 34% = ~4,966 words). Shares are estimates, not Turnitin per-section scores (Turnitin only gives the 34% overall).

---

## P0 — 5.3 Calibration, Shift, and External Validity ✅ APPLIED

### PARA-53A ✅ — In-domain calibration — `Journal_Paper/halurisc.tex:640`

> Values of ECE for the raw, Platt-scaled, and isotonic variants of the conventional model are 0.0045, 0.0091, and 0.0070, respectively, on HaluEval, demonstrating that calibration does not help to decrease ECE in this particular example. Figure~\ref{fig:reliability} explains this fact. On HaluEval, the points at the extreme ends lie on the diagonal, and there are only a few distinct values for the mid-score range; therefore, a small number of stable bins are used. FaithBench is aggregated because its scores concentrate near 1.0. The Platt and isotonic curves overlap the raw curve at both ends. The m3 raw score is already well calibrated in-domain; hence, the ECE value is quite low at 0.0078, which leaves little room for further calibration on HaluEval. Platt scaling applies a two-parameter logistic transformation where $s$ (the raw score in the figures) is transformed into a calibrated probability. (Eq. 13 follows unchanged in tex.)

### PARA-53B ✅ — Target recalibration part 1 — `Journal_Paper/halurisc.tex:659`

> The source Platt method cannot repair the base-rate shift. The fixed-hp m3 raw score is severely overconfident on RAGTruth QA: ECE 0.731 and Brier 0.689, with a Cox slope of 0.18 on the 5{,}034 calibration rows. Target Platt lowers ECE to 0.125 and Brier to 0.161, but the Brier skill score against the test prior is $-$0.10, and a constant predictor at the test prevalence has Brier 0.146, lower than every calibrated row. Calibration fixes the scale, not the ranking; AUROC stays at 0.558. The conventional target-Platt row behaves the same way at ECE 0.134 and Brier 0.164. Figure~\ref{fig:calibshift} shows the raw and target-recalibrated scores.

### PARA-53C ✅ — Target recalibration part 2 (operating point) — `Journal_Paper/halurisc.tex:659`

> The source-only operating point (P0) selects the threshold on the HaluEval validation split at a 5\% false-positive budget. On HaluEval test, recall varies between 99.0--99.5\% while FPR stays near 5\%, but on RAGTruth QA the same threshold flags all 900 rows of the deployed seed-42 checkpoint, because every score exceeds 0.095, so P0 is not usable. Table~\ref{tab:target} reports both models on the same test set. A prevalence-only predictor at 0.1778 reaches ECE 0.000 and Brier 0.146 and predicts no positives; at the calibration prevalence of 0.3107 it gives ECE 0.133 and Brier 0.164, suggesting that the calibrator sets the level to the calibration prevalence and the residual ECE is dominated by the prevalence gap. The conventional target-thresholded mode (P2) selects 0.313 at a 5\% false-positive budget and attains 5.0\% recall at 5.8\% FPR (precision 15.7\%, F1 0.076). At 17.8\% prevalence, random flagging at 5.8\% FPR yields 5.8\% recall, so P2 is chance-level.

### PARA-53D ✅ — Target recalibration part 3 (display score) — `Journal_Paper/halurisc.tex:659`

> At 0.5, P1 flags no RAGTruth QA rows; the deployed seed-42 calibrator has a maximum target-Platt score of 0.355 (35.4\% before rounding, 35\% on the card), the median is 0.331, and 16\% of QA rows reach 0.35, so the card usually relies on the per-claim verdict adjustment. Figure~\ref{fig:operating} shows the operating point and label budget: mean ECE is 0.130 at 50 rows, 0.124 at 252 rows, and 0.125 at 5{,}034 rows, though a single 252-row draw can miss the full-split ECE by about 0.06. All calibration metrics refer to the pre-verdict score; the displayed percentage is a design aid, not a validated probability.

### PARA-53E ✅ — Conformal — `Journal_Paper/halurisc.tex:687-702`

> Table~\ref{tab:conformal} summarizes the coverage. Split conformal prediction offers a coverage guarantee for the score. (Equations 15-16 unchanged in tex.) The main run fits the Platt map and the quantile on the same 5{,}034 rows; thus, its coverage serves as a construction check and gives 0.940 and 0.967 coverage on QA and 0.900 and 0.950 on the calibration rows. A disjoint sensitivity fits Platt on 2{,}514 rows and the quantile on 2{,}520 rows. On that split, the 900 QA test rows reach 0.944 coverage at 90\%, which shows that the marginal guarantee holds. Nevertheless, the marginal guarantee hides the class split: positive coverage is 0.685, while negative coverage is 1.000. A class-conditional quantile on the full calibration rows raises positive coverage to 0.887, still slightly below nominal. FaithBench coverage falls to 0.320 and positive coverage to 0.003, because a Platt map fitted on QA sends almost every FaithBench score to the negative side. Conformal prediction therefore certifies the marginal rate only when the calibrator and the quantile are fitted on disjoint rows, and it does not transfer across the task shift.

### PARA-53F ✅ — Multicalibration — `Journal_Paper/halurisc.tex:702`

> Multicalibration by answer length and claim count shows where the average ECE hides error. On the 900 QA test rows, the target-Platt score has overall ECE 0.125. The worst length bin is 60 words or fewer (n=154), with ECE 0.275; the worst claim-count bin is a single claim (n=36), with ECE 0.346. The Brier score does not follow the ECE ranking, so averaging is not a safe summary for short or one-claim answers. FaithBench is not improved by the QA calibrator: its target-Platt ECE is 0.629 against 0.453 for the raw score.

### PARA-53G ✅ — Adaptation vs shift — `Journal_Paper/halurisc.tex:729`

> The conventional model m0 is trained only on HaluEval, while EC-XGB also sees RAGTruth summarization and data-to-text rows, so those two pairs measure adaptation, not general shift (Table~\ref{tab:shift}). On the pooled RAGTruth test split, the flag rate decreases from 99.9\% to 61.3\% and AUROC increases from 0.475 to 0.582. The improvement is task-specific, with summarization AUROC rising from 0.634 to 0.736 and data-to-text AUROC rising from 0.567 to 0.771. Held-out QA remains near chance with AUROC from 0.547 to 0.549. FaithBench AUROC rises from 0.509 to 0.594, but F1 falls from 0.813 to 0.120 and raw ECE rises from 0.292 to 0.450, so the raw scores transfer poorly. The conventional F1 matches the all-positive baseline because almost every row is flagged. ECE uses raw scores, F1 uses a fixed 0.5 threshold, and Flagged is the share above it.

### PARA-53H ✅ — Task-shift — `Journal_Paper/halurisc.tex:757`

> Task-shift is the weakest test in the experiment. For the dialogue task, m3 flags 99.3\% of rows and its F1 equals the all-positive baseline, with AUROC 0.763; the length-only baseline reaches 0.724. For summarization, m3 reaches AUROC 0.717 and flags 37.5\%, while the length-only baseline reaches 0.788 and flags every row. The conventional m0 model flags 99.2\% of dialogue rows and 97.7\% of summarization rows, with AUROC 0.741 and 0.605. On dialogue, both models collapse to the all-positive baseline in F1, though their AUROCs stay above chance; on summarization, both rank below the length-only model. Figure~\ref{fig:transferscore} shows the raw score distributions of the conventional model: the HaluEval panel separates the classes, while the external panels pile both classes near 1.0, so the 0.5 threshold flags almost every row. This is the visual form of the 99.9\% flag rate in Table~\ref{tab:shift}.

### PARA-53I ✅ — External + index stress — `Journal_Paper/halurisc.tex:784-798`

> (Anchor sentences unchanged in tex: Table~\ref{tab:external} reports the two external corpora and the task-shift pair. The fixed-hp m3 model ranks the paired hallucinated answer above the grounded answer in 86.7\% of 1{,}000 medical pairs [84.5, 88.7], with AUROC 0.821 [0.806, 0.837] on the 2{,}000 rows; equation and length sentences unchanged.) The ranking transfers while the scale does not: m3 flags every row, giving an F1 of 0.667 (all-positive) and an ECE of 0.493. The source indicator gives MedHallu a one and HaluBench a zero; a source-zero sensitivity on MedHallu changes pair accuracy and AUROC by at most 0.003 and 0.002, so the indicator does not drive the result. HaluBench is a 30\% stratified sample. Its overall AUROC is 0.600, with covidQA 0.833, DROP 0.513, and FinanceBench 0.527; the model transfers to the biomedical subset but is at chance on the other two. An indexing stress test investigates whether a poor held-out QA result is caused by a retrieval issue. The hybrid BM25 and FAISS index covers the 989 RAGTruth QA source documents. On a 300-row, 50-group sample, recall@1 is 0.94, recall@5 and recall@10 are 1.00, and MRR is 0.959. Replacing the gold context with the top-5 retrieved passages changes AUROC from 0.545 to 0.521, F1 from 0.345 to 0.349, and ECE from 0.706 to 0.704. The AUROC delta is $-$0.023 [$-0.061$, 0.011], which includes zero. Retrieval is therefore not the bottleneck: the model is near chance on RAGTruth QA even with the gold source passage. The qrels are the known source passage, not human web-search judgments, so this measures index recall, not end-to-end search quality. The sample has 50 source groups, so the interval is wide.

---

## P1 — 4.2 Dataset ✅ APPLIED

### PARA-42A ✅ — Main dataset — `Journal_Paper/halurisc.tex:474`

> HaluEval QA \cite{haluEval} has 10{,}000 question-answer pairs, with a knowledge passage, a correct answer, and a hallucinated answer. Data is divided with a 70/15/15 split by item so that both answers associated with the same question are kept together. HaluEval is entirely generated by ChatGPT \cite{haluEval,bang-etal-2025-hallulens}, somewhat blurring the difference between factual inaccuracies and hallucinations. RAGTruth \cite{ragtruth} has natural answers with human span annotations, and FaithBench \cite{faithbench} offers hallucinations for contemporary LLM summaries. RAGTruth maintains its official document-level partition: summarization and data-to-text provide training rows and a held-out test set each, and QA provides 5{,}034 calibration rows and a held-out test set. RAGTruth QA is a subset of the full corpus, so its rows are not additive. The pooled RAGTruth test set has three held-out test sets. Table~\ref{tab:datasets} lists the corpora, their sizes, and positive rates. MedHallu pqa\_labeled \cite{pandit2025medhallu} adds medical QA pairs with human-annotated gold answers and paired LLM hallucinations. HaluBench \cite{ravi2024lynx} provides a stratified 30\% sample of its DROP, FinanceBench, and covidQA rows, balanced between PASS and FAIL; its HaluEval, RAGTruth, and PubMedQA rows are excluded. Seeded samples of HaluEval tasks are used as task-shift sets. External corpora are never used for fitting or threshold selection.

### PARA-42B ✅ — Preprocessing — `Journal_Paper/halurisc.tex:496`

> Leading and trailing whitespace is trimmed, and lexical matching is done in a case-insensitive manner. NER and NLI are performed on the original text, with entity strings converted to lowercase for matching. Missing context is a special case where lexical overlap is zero and NLI probabilities are uniformly distributed. In m1 and later variants, both length counts are transformed with log1p; in m0 and the conventional model, raw counts are used. Claims are segmented into no more than six clause-level segments, with conjunctions split into clauses. If the context has more than four sentences, each claim is matched to its four best sentences by token overlap, and both sides of a pair are truncated to 400 characters. Both index sets are frozen. A RAGTruth answer is considered positive if it includes at least one annotated hallucination span. Severity annotations in FaithBench are condensed by a worst-severity principle, and a Questionable annotation is counted as positive. HaluEval uses its own label scheme. NLI and MiniLM inference ran on the reference GPU in fp16, while lexical and entity features were calculated on the CPU. The feature-based logistic baselines standardize their inputs on the training split; TF-IDF models use sparse features, and XGBoost does not scale. The source indicator takes a value of 1 for FaithBench and MedHallu, and 0 for HaluBench and the task-shift sets. FaithBench, RAGTruth summarization, and data-to-text have no question; therefore, question overlap is 0, and question similarity is defined as the cosine similarity between the answer and the empty string encoding. Licenses and the data availability statement are in the Declarations.

---

## P2 — 5.5 Explanation, Efficiency, and Judges ✅ APPLIED

### PARA-55A ✅ — Explanation setup — `Journal_Paper/halurisc.tex:846`

> The explanation uses the fixed-hyperparameter m3 seed-42 checkpoint. Figure~\ref{fig:shapimp} ranks the top-15 features by mean absolute SHAP value over 1{,}000 stratified HaluEval test rows. Context overlap is the strongest driver, followed by word count, question-answer cosine, character count, and question-answer Jaccard similarity. The claim-contradiction mean appears at rank eight. The top ten features cover 77.9\% of the attributions, and exact TreeSHAP matches the model margin up to $9.2\times10^{-6}$. The attributions are calculated for the raw margin, not the display score. The m3 rankings are highly correlated with the frozen B2 rankings (Spearman's $\rho = 0.878$, Kendall's $\tau = 0.730$); therefore, the claim features refine the explanation, rather than replace the original drivers. Cross-seed Spearman correlation is 0.972 and 0.950; nine of the top ten features are shared, and the top five features are identical. The explanation is resilient to the choice of the seed but not to content editing.

### PARA-55B ✅ — Perturbation + audit — `Journal_Paper/halurisc.tex:846`

> Table~\ref{tab:perturb} presents a 20-case perturbation pilot: content edits move the score, structural edits barely shift it, and the top-1 feature is flipped in 54.5\% of entity perturbations and none of the other operators. The entity flipping rate is higher than in B2, so the explanation is less stable when the answer entity changes; the per-operator sample size is small. The structured AI-assisted audit of 26 error-enriched cases re-scored under m3 evaluated each explanation for consistency with answer-context evidence, rather than the dataset label. Thirteen explanations were considered implausible, three uncertain, and ten plausible, so the explanations are used as diagnostic aids, not faithful attributions. False positives carry the length confound, and false negatives frequently rely on entities repeated in the context. As the sample is enriched for errors, the percentages characterize the audit, not the test set. The labels are author-checked pre-annotations: LLM tools assisted the annotation, and the authors reviewed and verified the final labels. Agreement with the pre-annotations was 0.98 for explanation, 0.94 for taxonomy, and 0.88 for claims. Cohen's $\kappa$ was 0.965, 0.923, and 0.788; the authors changed 2\%, 6\%, and 12\% of rows, and the blind Fleiss $\kappa$ was 0.940, 0.868, and 0.761.

### PARA-55C ✅ — Judges — `Journal_Paper/halurisc.tex:919`

> LLM judges are not fair evaluators, and their preferences depend on the answer order \cite{wang2024fair}. Table~\ref{tab:judge} compares EC-XGB with two judges on the RAGTruth QA rows at natural prevalence. GPT-5.6 Luna reaches AUROC 0.910 compared to 0.558 for EC-XGB, and Qwen 3.5 4B also beats EC-XGB, meaning that both judges are superior QA detectors, while the feature model advantage is only in cost. The target-fitted length-only baseline also ranks the QA rows higher than the feature model. Under matched multi-source training, the adapted encoder achieves 0.694 on the pooled RAGTruth test set, outperforming EC-XGB at 0.582. The feature model has less degradation on new tasks, but both models approach random guessing. The length-matched panel changes the ranking: on the exact-length subset, EC-XGB performs similarly to the stronger judge in AUROC and better in F1, meaning that the natural-prevalence difference is due to the domain shift and not the length difference. The paired interval for GPT minus EC-XGB is $-$0.052 [$-0.127$, 0.024], and the AUROC interval is $+$0.008 [$-0.034$, 0.048]. Both include zero. Prompt choice moves GPT's F1 from 0.814 to 0.854 across three prompts, about the size of the model gap. GPT uses default decoding and three seeds, while Qwen is a single deterministic run, so the judge ranking is not stable.

---

## P3 — Remaining flagged paragraphs (all ✅ APPLIED)

### PARA-INT1 ✅ — Model overview — `Journal_Paper/halurisc.tex:294`

> The EC-XGB model uses 26 features divided into seven groups, eight claim-level NLI aggregates, and a source indicator inside a tuned XGBoost classifier. The score is mapped to a probability with Platt calibration, and explanations are produced with Shapley additive explanations (SHAP). Grouped splits with length and overlap controls are used to reduce shortcut learning. Source calibration is fitted only on validation data. Experiments with target calibration check the reliability of predictions. The log transformation compresses the length scale but keeps the cue. The claim aggregates support per-claim explanations, but the seen-task gain on RAGTruth comes from the multi-source rows and source indicator, not the claim aggregates. Transfer limits are set up on RAGTruth and FaithBench. The evaluation adds MedHallu \cite{pandit2025medhallu}, a HaluBench subset \cite{ravi2024lynx} with no training overlap, the HaluEval task shifts, conformal prediction, and judge stress tests. Results are reported for F1, ranking, calibration, and paired errors, separating ranking from calibration: the model transfers ranking to medical pairs but not its probability, and it fails under the task shift.

### PARA-INT2 ✅ — Data overview — `Journal_Paper/halurisc.tex:296`

> HaluEval QA contains 10,000 question pairs, each with a correct and a hallucinated answer, for training and in-domain testing. RAGTruth provides natural responses for adaptation and recalibration. The target application is question answering with a reference context, and FaithBench serves only as an out-of-domain summarization stress test. The fixed-hyperparameter EC-XGB stack (m3) reaches F1 0.9854 and AUROC 0.9986 on HaluEval QA. RAGTruth shows the same pattern: the flag rate falls on tasks seen in training, while held-out QA stays near chance. On MedHallu, the same model ranks the hallucinated answer above the grounded one in 86.7\% of 1,000 pairs but flags every row. On HaluEval dialogue it flags almost every row, and on HaluEval summarization it ranks below a length-only model. Target calibration on 5,034 RAGTruth QA calibration rows lowers ECE from 0.731 to 0.125 without changing the ranking. One answer takes 127.5~ms on GPU (2.00~s on CPU), and the approximate cost is about \$0.001 per 1,000 predictions.

### PARA-31 ✅ — Working principle — `Journal_Paper/halurisc.tex:353`

> EC-XGB produces a raw score from the features, which is calibrated by a Platt calibrator trained on RAGTruth QA calibration rows to obtain the displayed probability. The model outputs a risk score, verdicts at the claim level, and SHAP values. The bottom part of the figure shows the deployment pipeline, where the Next.js front end calls the FastAPI service, which loads the model, calibrator, explainer, and feature models at startup. The right block lists the evaluation datasets. HaluEval evaluates in-domain performance, while RAGTruth summarization and data-to-text measure performance on task types seen in training. RAGTruth QA and FaithBench are held out.

### PARA-32 ✅ (kept — fixed terms) — Features — `Journal_Paper/halurisc.tex:364`

> No wording change applied: `driver / rows / rank / cover / raw margin / novel / logical support / easy-to-detect / hedging words / finds / usually operate` and `\cite{ru2024ragchecker}` are fixed claims. Rephrase only by splitting or reordering sentences around these keywords.

### PARA-33 ✅ — Training — `Journal_Paper/halurisc.tex:416`

> The classifier does not see RAGTruth QA rows and thus performs zero-shot QA ranking. In contrast, the display calibrator sees labeled QA rows and produces a non-zero-shot display score. The 35-feature setup is called EC-XGB, with seed 42 as the deployed checkpoint and other seeds as stability checks. The ablation described in Section~\ref{sec:results} shows that the claim block makes no aggregate F1 contribution, and the claim count $K$ is kept because answers with more claims are harder to verify. Algorithm~\ref{alg:ecxgb} describes the procedure: steps 1 and 2 generate the feature matrix, and steps 3 to 9 perform the model fitting for the three seeds. Each iteration creates the 35-feature input, trains on HaluEval plus the RAGTruth non-QA rows, fits the display calibrator on the 5{,}034 QA calibration rows, and sets the source and target thresholds.

### PARA-34 ✅ — Explainability — `Journal_Paper/halurisc.tex:451`

> (TreeExplainer sentence unchanged.) Global feature importance is defined as the mean absolute SHAP value over a stratified test sample. The explanation endpoint outputs five features with the largest absolute SHAP values for a given prediction. Post-hoc attribution scores can be unstable despite a stable underlying model \cite{kelodjou2024shap}, so the pilot evaluates them directly \cite{adebayo2018}. A 20-case pilot evaluates the effects of six operators: entity, number, and date replacements, support removal, context insertion, and clause shuffling. (Final `It measures...` sentence unchanged.)

### PARA-35A ✅ — System — `Journal_Paper/halurisc.tex:455`

> The method runs in a web application with a Next.js frontend and a FastAPI backend. (Remaining sentences unchanged: browser calls, analyze route, At startup FastAPI loads.)

### PARA-35B ✅ — Example + contribution — `Journal_Paper/halurisc.tex:464`

> For the example of the hallucinated number, the context says 5 days per decade whereas the answer says 10. The raw score is 0.999 and the target-calibrated score is 35.4\%, which the card rounds to 35\% and places in the medium band (0.35 to 0.60). The claim view marks the number claim as unsupported, so the chat card reports medium risk. Though the two labels agree in this case, they come from different rules. The claim verdict audit is covered in Section~\ref{sec:results}. The requirements file fixes all dependencies; split indices are stored with the artifacts; and the audit scripts check that rebuilt and reported values match the frozen artifacts. Audits run after each build and fail if there is any mismatch. The contribution of this work is in the training and evaluation protocol, not in the development of a new learner. EC-XGB makes three standard changes: log-scaled length counts with monotone constraints, claim-level NLI aggregations for explanation, and multi-source training with a source indicator. Section~\ref{sec:results} evaluates what each change contributes.

### PARA-44 ✅ — Metrics intro (first sentence) — `Journal_Paper/halurisc.tex:521`

> Two types of metrics are provided here. (Remaining sentences unchanged.)

### PARA-51 ✅ (kept — fixed terms) — Model comparison — `Journal_Paper/halurisc.tex:542`

> No wording change applied: `ranking columns / comparable trend / takes first place / paints / takes the lowest / weak controls / is much stronger / lowers / is behind / descriptive / underlie / three-seed averages` are fixed claims. Rephrase only by splitting or reordering sentences around these keywords.

### PARA-52A ✅ — Ablation result — `Journal_Paper/halurisc.tex:584`

> The claim aggregation changes F1 by -0.0004 at m2 compared to m1, and the entire m3 stack changes F1 by +0.0001; neither is an aggregate gain. On the pooled RAGTruth split, AUROC improves from 0.494 to 0.590 and raw ECE reduces from 0.635 to 0.275; for the held-out QA split the comparison is 0.556 vs 0.558, still within seed noise. McNemar yields p = 0.789 for m0 vs m3, so the ablation does not resolve a difference. The gain does not come from the claim features, but appears when the multi-source rows and source indicator are added. The MedHallu source-zero check suggests that the indicator is not the main driver of the gain, but a seen-task source ablation would settle it. The deployed checkpoint achieves 0.9855 with its own tuned hyperparameter configuration, 0.0001 above the fixed ablation. (User substitutes in this block changed fixed claims — kept precise wording; only `still` applied.)

### PARA-52B ✅ — Group ablation — `Journal_Paper/halurisc.tex:602-625`

> Table 7 block kept (fixed terms: reproduces / fixed-hp / reaches / followed by / costs / resolved). Mean-ordering applied: `The mean ordering is m3 > m1 > m2 > m0; however, the per-seed ranks flip and the differences are below seed noise. The ECE sign is stable: m0 is lower than m3 on every seed, so the ablation does not establish a calibration advantage for m3 in this sample.` (concluding clause restored — your version dropped it).

### PARA-54A ✅ — Stats — `Journal_Paper/halurisc.tex:802`

> Applied one structural change: `Under the deployed checkpoints, neither EC-XGB nor the conventional method exhibits a significant difference (p = 0.45), whereas ...` (fronted adverbial; all other terms fixed: conventional / deployed checkpoints / same test rows / primary / part / baseline / single / defined / shows / with / does not produce). Unflagged methods + Table-stats sentences untouched.

### PARA-54B ✅ — Bootstrap — `Journal_Paper/halurisc.tex:804`

> The F1 gain comes mainly from flagging almost every row, not from improved ranking. (Rest of block unchanged in tex: +0.0451 / +0.0962 / All the paired fixed-minus-tuned differences include zero / rebuild does not affect / 1,500 + 150 + 450 groups / unresolved + adaptation sentences kept.)

### PARA-56 ✅ — Error — `Journal_Paper/halurisc.tex:947`

> On HaluEval, the m3 seed-42 model makes 43 errors (Figure 10), which amounts to an error rate of 1.4%. WP5 includes seven out of eight false positives and 12 out of 35 false negatives chosen to cover the codebook. An additional 50-instance worksheet, reviewed by one author, brings 31 artifact cases and 19 content cases distributed into 12 categories.

### PARA-DC ✅ (one sentence) — Declarations — `Journal_Paper/halurisc.tex:997-1006`

> All bullets kept except last acknowledgement sentence, now: `The authors thank the curators of HaluEval, RAGTruth, FaithBench, MedHallu, and HaluBench for releasing their datasets publicly for research.` (license/procedure bullets are boilerplate with fixed legal meaning — kept exact).

---

## CAP — Flagged captions ⏭️ SKIPPED (per author: 1-2 lines each, low weight)

Keep dataset names, `n`, `%`, and meaning exact. Only change wording order.

- [CAP-01] Figure 1 — `Journal_Paper/halurisc.tex:358` — Original: `End-to-end pipeline, deployment path, and evaluation corpora of EC-XGB.`
- [CAP-02] Table 3 — `Journal_Paper/halurisc.tex:478` — Original: `Datasets used for training and evaluation.` Plus flagged row text: `train / validation / test`, `held-out test + calibration`, `zero-shot external`, `task-shift test`, and rows `HaluBench subset`, `HaluEval dialogue`, `HaluEval summarization`, positives `50.0%`, `17.8 / 31.1%`. Rewrite caption only; do not change table data.
- [CAP-03] Table 8 (target) — `Journal_Paper/halurisc.tex:663` — Original: `Calibration of fixed-hp m3 and conventional target-Platt on the RAGTruth QA test set (n = 900, 17.8% positive). *Slope: three-seed mean, bracket: envelope of per-seed 95% intervals.`
- [CAP-04] Table 9 (conformal) — `Journal_Paper/halurisc.tex:706` — Original: `Conformal coverage on RAGTruth QA and FaithBench, mean over seeds 42/123/456. Class-conditional rows use the full 5,034 rows; split rows use the disjoint halves.`
- [CAP-05] Table 10 (shift) — `Journal_Paper/halurisc.tex:733` — Original: `External and adaptation performance of B6 m0 and EC-XGB m3 from the tuned B6 run, mean over three seeds.`
- [CAP-06] Table 12 rows — `Journal_Paper/halurisc.tex:814-819` — Original row labels: `EC-XGB vs conventional`, `EC-XGB vs conventional (fixed hp)`, `Conventional vs random forest`, `Conventional vs logistic regression`, `Conventional vs heuristic`, `Conventional vs NLI-only`. Keep labels; only the surrounding paragraph (PARA-54A) needs rewrite.
- [CAP-07] Table 13 — `Journal_Paper/halurisc.tex:828` — Original: `Two-level bootstrap deltas for the fixed-hyperparameter ablation (2,000 draws; groups within seed, then seeds). Intervals are 95%.`
- [CAP-08] Table 17 — `Journal_Paper/halurisc.tex:923` — Original: `Judge stress tests on the 900 RAGTruth QA rows and the 158-row length-matched HaluEval set.`

---

## Rewrite rules

- Keep every number, threshold, ECE/Brier/AUROC/F1, row count, and citation exactly the same.
- One idea per sentence. Mix short (6-10 words) and medium (12-18 words). No three long sentences in a row.
- Simple student words. No `delve, robust, comprehensive, furthermore, moreover, crucially`. No em-dash. Max one transition per 2-3 paragraphs.
- No first-person plural in formal sections: no `we / our / ours / us`.
- Active voice. Inline citations stay with the claim they support.
- Do not touch equations, Algorithms 1-2, tables data, figures, or `ref.bib`.

## Next step

Done. All PARA blocks applied; CAP skipped. Re-run Turnitin and compare against the 34% baseline.
