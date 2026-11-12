# Section 3 Review: Proposed Method

- **Reviewer:** independent subagent (journal reviewer, Section 3)
- **Manuscript:** `Journal_Paper/halurisc.tex`, lines 351-477 (`\section{Proposed Method}` through the paragraph before `\section{Experiments}`). The manuscript was edited while this review was in progress; the line numbers in the issues table are from the review copy and run four lines lower than the current file. The quoted text is the stable locator.
- **Guides read:** `docs/prompt.md`, `docs/writing-guideline.md`
- **Code cross-checked:** `src/models/run_b6_modified.py`, `src/features/claim_features.py`, `src/features/nli_features.py`, `src/features/semantic_features.py`, `src/features/entity_features.py`, `src/models/fit_ec_xgb_calibrator.py`, `src/api/main.py`, `src/claims/verify.py`, `web/components/assistant-ui/auto-risk-card.tsx`, `web/app/(site)/analyze/page.tsx`
- **Artifacts cross-checked:** `artifacts/results/rebuild/fixed_hp/b6_feature_names.json`, `artifacts/results/b6/b6_calibration.json`, `artifacts/models/b6/ec_xgb_display_calibrator.joblib`, `artifacts/results/b5/b5_perturbation_aggregates.csv`, `docs/rebuild/p0/author_review/ingest/p0_claim_verdicts.csv`, `docs/rebuild/p0/author_review/AI_REVIEW_RESULTS.md`
- **Figures checked:** `Journal_Paper/figures/revision/architecture-flow-large.png`, `Journal_Paper/figures/revision/system-architecture-large.png`, the UI screenshot (removed from the manuscript)
- **Constraints respected:** no edits to the manuscript, artifacts, or any existing file. Only this report was written.

## Verdict

**Major revision.** The method is sound and the feature counts check out, but the section's framing describes a different contribution and a different deployment than the artifacts. The "three key aspects" framing omits multi-source training, which the Results identify as the source of the seen-task gain, and it oversells the claim features that the Results show add no aggregate F1. The claim aggregation is not reproducible as written: $C(k)$ and $E(k)$ are undefined, the verdict rule appears only in §3.5, and Algorithm 2 omits the calibration rows and the deployed seed. The deployed-system description also disagrees with the code: the source indicator is fixed to 1 at serving, the figure shows the `/predict` card while the text describes the `/verify` card, and the 100-claim audit is a result placed in the Method. These are text and structure fixes, not new experiments. With R1-R10, the section would be credible to a methods reviewer.

## Strengths

- **The high-level order is right.** Working principle → features → EC-XGB → calibration/explainability → system follows the spirit of G8's proposed-method flow, and the pipeline figure gives the reader a map before the details. The section does not wander into a literature review.
- **The feature rationale is honest.** Length is presented as a strong but fragile signal, lexical overlap as the strongest but shallow signal, NLI as the only base group that captures logical support, and the claim features as explanation aids. The section does not pretend the model is a general detector.
- **The three modifications are technically sensible and traceable to code.** `log1p` is applied to `n_chars` and `n_words`, 13 features carry monotone constraints, the eight claim features are computed by `claim_features.py`, and the 35-feature count matches `b6_feature_names.json` m3.
- **The calibration and explanation coverage is conceptually complete.** Platt and isotonic, ECE, Brier, reliability diagrams, SHAP, and the six-operator perturbation pilot are the right ingredients, and the citations (Guo, Platt, isotonic, Gasparin, Kelodjou, Adebayo) are appropriate. The deployed display calibrator is Platt, which matches the artifact.
- **The contribution paragraph's "not a new learner" framing is the right level of honesty.** The system subsection is grounded in the real Next.js/FastAPI deployment rather than a generic architecture diagram, and the risk-card example gives the reader a concrete case.

## Issues

| # | Severity | Exact location (quote) | Issue | Why it matters | Suggested fix |
|---|---|---|---|---|---|
| C1 | **Critical** | Line 405: `EC-XGB alters three key aspects of the conventional XGBoost learner.` … line 413: `The third modification is the addition of a source indicator` | The three changes are (1) log scaling + monotone constraints, (2) claim features, (3) source indicator. The actual m3 change is multi-source training plus the indicator. The Results say the seen-task gain comes from the multi-source rows, not the claim features. The abstract and introduction sell multi-source training as central. The three-change list omits the main change, even though the algorithm paragraph mentions the added rows. | A reviewer who compares §3.3 with §5.2 will conclude the method section misattributes the contribution. The paper's own conclusion says the claim block costs only 0.0010 F1. The current framing sets up an expectation the Results contradict. | **R1 + R2 + R10.** Rewrite the three changes as log scaling + monotone constraints, claim features for explanation, and multi-source training + corpus indicator. State that the first and third steps bundle two changes each and the ablation does not separate them. |
| C2 | **Critical** | Line 405: `Consider an answer composed of K claims, with C(k) the contradiction and E(k) the entailment of claim k, and let v_k be the verdict of claim k.` Algorithm 2, line 421: `\Require labeled HaluEval QA rows, RAGTruth non-QA rows, seeds ...` | $C(k)$ and $E(k)$ are not defined as per-claim maxima over the selected context sentences. $v_k$ depends on the support-first rule, which is defined only in §3.5. Algorithm 2 omits the 5,034 QA calibration rows that step 6 uses, and its output does not say which seed is deployed. "Zero-shot" is used for both the classifier and the pipeline. | A reader cannot reproduce the eight claim features or the algorithm from §3.3. The 35-feature count is correct in the artifact, but the method text is not. The zero-shot ambiguity matters because the display calibrator is target-fitted. | **R1** (define $C(k)$, $E(k)$, and the verdict rule before the equation) and **R3** (add the calibration rows to `\Require`, specify seed 42, say "classifier training"). |
| M1 | **Major** | Line 358: `a binary source indicator marks whether the answer came from a natural response corpus`; line 444: `the source indicator is set to one for natural responses` | The feature is `source_ragtruth`: 0 for HaluEval and 1 for RAGTruth. At serving, `EC_SOURCE_VALUE = 1.0` for every request (`src/api/main.py:89,613`). It is a corpus flag, not an answer property. The text implies it distinguishes natural from synthetic answers at inference. | The deployed model cannot vary the indicator, so it cannot separate a natural answer from a synthetic one. A reviewer who reads the API will see the mismatch, and the "marks whether" phrasing overstates what the feature does. | **R4.** Call it a "corpus indicator," state the training assignment, and say the API fixes it to 1 for every request. |
| M2 | **Major** | Line 412: `There are 35 input features in total (24 base + 2 log-scaled length + 8 claim + 1 source).` | The 26 base features are called "24 base + 2 log-scaled length", which conflicts with "26 base features" everywhere else. Only `n_chars` and `n_words` are log-scaled; `n_sentences` and `avg_word_len` are not. Table 1 does not mark the two transformed columns. | A reader cannot reconcile the counts with Tables 1 and 2. The "24 base" label makes it look like the model has 24 base features, not 26. The log transform is a key modification, so its scope should be exact. | **R5.** Use "the 26 base features (two of them log-scaled), the eight claim features, and one corpus indicator." Add a dagger to `n_chars` and `n_words` in Table 1. |
| M3 | **Major** | Line 405: `monotonicity constraints ensure that risk cannot decrease as contradiction or novelty rises and cannot increase as entailment, overlap, or semantic similarity rises.` | 13 features are constrained, not all contradiction, novelty, entailment, overlap, and semantic features. `overlap_answer_question`, `cosine_q_ans`, and `nli_ans_entails_ctx` are unconstrained. The exact list is missing. | A reader cannot reproduce the constraints, and the sentence implies a broader constraint set than the code uses. The constraints are a core part of the EC-XGB claim. | **R1.** List the constrained features and say the rest are unconstrained, or add a "Constraint" column to Table 1. |
| M4 | **Major** | Line 365: `The eight claim-level aggregates in Table 2 repeat the NLI and overlap signals at the level of atomic claims`; line 436: `the claim-level features support per-claim explanations` | Table 2 has no overlap features; the eight features are seven NLI aggregates plus a claim count. The three verdict ratios sum to one. The eight aggregates are answer-level summaries, not per-claim explanations; the per-claim verdicts come from the verification layer. | A reviewer will see the table and the text disagree. The "per-claim explanation" claim is not supported by the aggregates. The collinear ratios also weaken the "eight features" framing. | **R1 + R10.** Say "seven NLI aggregates and a claim count," rename Table 2's caption to "Claim-level features (8 features)," note the ratios sum to one, and separate the aggregates from the verification layer's per-claim verdicts. |
| M5 | **Major** | Line 444: `The calibrator can be re-fit on a small labeled sample of the target corpus` … `its calibrator is fitted on the 5,034-row RAGTruth QA calibration split, and the source indicator is set to one for natural responses.` | 5,034 rows is not a small sample; the sentence conflates the calibrator input with the model input (source indicator); the deployed Platt-vs-isotonic choice was made on the test split (`fit_ec_xgb_calibrator.py:165`, disclosed in the limitations paragraph of Section 5), but §3.4 presents the fit as clean; P0 is undefined. | The calibration subsection is the paper's headline contribution. A methods reviewer will ask how the calibrator was selected. The current text hides the test-based choice and misstates the sample size. | **R6.** Split source and target calibration, say 5,034 rows, state the deployment convention, cite Table 2 for P0, and disclose the earlier test-based Platt/isotonic choice. |
| M6 | **Major** | Line 461: `The verdict card sets high risk for a contradicted claim, medium for an unsupported claim, and low for a fully supported claim.` … `The number claim is marked unsupported rather than contradicted, so the card reports medium risk.` | The figure is the analyze card (`/predict`), whose label is score-based at 0.35/0.60. The chat card (`/verify`) uses the claim-based headline and adjusts the percentage. The text mixes the two. The 35.4% score is just above the 0.35 boundary, so the figure's medium label is score-based, not claim-based. The screenshot shows 35%, not 35.4%. | The flagship example does not demonstrate what the text says. A reviewer who opens the app will see two different label rules. The mismatch undermines the system description. | **R7.** Distinguish the cards, give the 0.35/0.60 bands, say the figure rounds 35.4% to 35%, and do not attribute the medium label to the claim verdict. |
| M7 | **Major** | Line 461: `This is a limitation of the support-first rule on numeric claims, not a contradicted-claim example; claim-verdict accuracy is not evaluated here. A 100-claim package covering 50 answers was reviewed by the three authors with AI pre-annotation` | The 100-claim audit is a result in the Method section, and it directly contradicts "claim-verdict accuracy is not evaluated here." The audit counts match the P0 artifact (63/14/5/18), but the paper never reports them in Results. | A result in the Method section is misplaced, and the contradiction suggests the audit was forgotten or the sentence is stale. A reviewer will ask where the audit is analyzed. | **R7** (replace the paragraph with the one-sentence pointer) and move the audit paragraph to §5.6 (Error Analysis and Discussion) or a new Results subsection, with the counts and agreement reported there. |
| M8 | **Major** | Line 436: `The contribution is the multi-source training protocol and the evaluation, not a new learner.` … `inference needs no LLM judge calls` | The contribution paragraph is placed before calibration/explainability and omits them. The "no LLM judge calls" claim is not true by default for `/verify`, where `judge_uncertain=True` (`src/api/main.py:257`) and an API key enables the judge. The positioning sentences at the end belong in the discussion in Section 5. | The contribution statement is incomplete and contains a system claim the API does not support by default. A reviewer will test the "no LLM judge" claim against the API. The paragraph also competes with the Results rather than setting up the method. | **R10.** Move the paragraph to the end of §3.5, include calibration and explanation, qualify the judge claim, and cut the defensive comparisons. |
| M9 | **Major** | Line 450: `B6 m0 is a separate 26-feature tuning run under the EC-XGB protocol` | m0 has no log scaling, no monotone constraints, no claim features, no source indicator, and no multi-source rows. It is the 26-feature reference, not an EC-XGB variant. The checkpoint paragraph also routes the reader to several Results tables before they exist. | The phrase "under the EC-XGB protocol" mislabels the baseline and confuses the ablation. The table routing is Results content in the Method. | **R9.** Describe m0 as the 26-feature reference from the same tuning protocol, and move the checkpoint map to §4.3 or a small table. |
| M10 | **Major** | Figure 1: `Platt Calibration — RAGTruth QA` with `5% false-positive budget`; `Per-claim verdicts — three verdicts per claim`; Figure 3 warning: `The model itself is trained on HaluEval synthetic data` | The 5% FPR budget belongs to the target threshold, not the Platt fit. Each claim gets one of three verdicts, not three verdicts. The screenshot warning is misleading for m3, which also trains on RAGTruth summarization and data-to-text rows. | The figure is the section's main visual. A reader will infer a constrained Platt fit and a three-label-per-claim design, neither of which is true. The screenshot warning contradicts §3.5's multi-source description. | Relabel the calibration box "Platt calibration (RAGTruth QA)" and put "5% FPR threshold" on the target-threshold step. Change "three verdicts per claim" to "one of three verdicts per claim." Update the warning string and recapture the screenshot. |
| m1 | Minor | Lines 405, 444, 446: `three key aspects`, `not robust`, `represents`, `highlighting`; four semicolons in §3.3-§3.5 | Four banned or discouraged patterns from `docs/prompt.md`, four prose semicolons, and four negative parallelisms (`It is not aimed ... but at`, `The output is a probability, not a binary label`, `not a new learner`, `not a contradicted-claim example`). The section also repeats "The first stage ... The second stage ..." five times. | The prose is readable, but a detector or a copy editor will flag the same patterns the guide bans. The repetition makes the pipeline paragraph sound mechanical. | Replace "three key aspects" with "three changes," "not robust" with "degrades," "represents" with "captures," and "highlighting" with "show." Split the semicolon sentences. Rewrite three of the four "not X, but Y" contrasts as direct statements. Group the stages into two or three sentences. |
| m2 | Minor | Line 365: `A changed number is a common hallucination and is easy to detect.` | The paper's own example in §3.5 shows a numeric claim marked unsupported rather than contradicted, so numeric detection is not easy. | The sentence overclaims and is contradicted later in the same section. | `A changed number is a common hallucination and a strong signal when the context states the value.` |
| m3 | Minor | Line 364: `a DeBERTa-v3 cross-encoder`, `MiniLM sentence embeddings`, `spaCy named entity recognition` | The exact checkpoints are missing: `cross-encoder/nli-deberta-v3-base`, `sentence-transformers/all-MiniLM-L6-v2`, `en_core_web_sm`. | Reproducibility. A reader cannot rebuild the feature extractor from the section, and the fallback NLI model could produce different features. | Add the checkpoint names in §3.2, or a footnote with the exact model IDs. |
| m4 | Minor | Line 450: `These methods are used in a two-tiered web application`; line 450: `an indexed collection`; line 459: `A tool call goes to the judge.` | "Two-tiered" is undefined, "indexed collection" is vague, and the tool call goes to the chat model (GPT 5.6 Luna), not directly to the judge. | A reader cannot map these terms to the system figure. The judge description is inaccurate. | `The method is deployed in a web application with a Next.js front end and a FastAPI backend`; `a local passage index`; `A chat tool call goes to GPT 5.6 Luna, which can call the judge.` |
| m5 | Minor | Table 1, lines 367-384 | The table lists feature names but no per-row counts, so the reader cannot verify that the rows sum to 26. "Entity counts" and "number counts" each hide two features. | The 26-feature total is a key number, and the table should let the reader check it without reading the code. | Add a Count column: 4, 4, 4, 6, 4, 2, 2. Mark `n_chars` and `n_words` with a dagger for log scaling. |
| m6 | Minor | Line 446: `The global SHAP explanations from a TreeExplainer applied to the raw XGBoost model were used to create Mean Absolute Importance summaries.` | Passive voice, and "Mean Absolute Importance" is not a standard term. The Results use "mean absolute SHAP". | The sentence is harder to read than the rest of the section and uses a term the reader will not find elsewhere. | **R8.** `A TreeExplainer computes exact SHAP values. Global importance is the mean absolute SHAP value over a stratified test sample.` |
| m7 | Minor | Line 462: `The audits run after every build and fail on any mismatch.` | `review_number_audit.py` and `rebuild_number_audit.py` print mismatches but exit 0; only `p1_number_audit.py` raises `SystemExit(1)`. | The sentence is a process claim, not a method claim, and it is not true for two of the three scripts. A reviewer will not check it, but it weakens the reproducibility paragraph. | Cut the sentence, or say `the audit scripts report any mismatch against the frozen artifacts.` |
| m8 | Minor | Lines 362, 364, 465: `It takes a question`, `It is not aimed`, `This is a limitation` | The first two "It" antecedents are the figure/pipeline and the task. The third "This" has no clear antecedent: the example, the support-first rule, or the card. | Vague pronouns slow the reader and are a common AI-writing tell. | `The pipeline takes...`, `The task is not aimed...`, `The numeric example shows a limitation...` |

## Exact replacements for Critical/Major issues

### R1. Replace the "three changes" paragraph and the claim-aggregation definition (lines 405-412)

```latex
EC-XGB changes the conventional setup in three places. First, the character and word counts are log-scaled, and monotone constraints are placed on the evidence features. The constraints are negative on context overlap, context Jaccard, context entailment, context cosine, claim minimum entailment, claim mean entailment, and the supported-claim ratio. They are positive on context contradiction, novel-entity ratio, novel numbers, claim maximum contradiction, claim mean contradiction, and the contradicted-claim ratio. All other features are unconstrained. Second, the eight claim features in Table~\ref{tab:claim} are added. They are seven NLI aggregates and a claim count. They use the same NLI scorer and support-first rule as the verification layer, computed against the provided context. A claim is supported when any retained sentence has an entailment probability of at least 0.5 and at least as high as its contradiction probability. It is contradicted when no sentence supports it and a sentence has a contradiction probability of at least 0.5 and above its entailment probability. Otherwise it is unsupported.

Consider an answer with $K$ claims. Let $C(k)$ be the maximum contradiction probability of claim $k$ over its selected context sentences, $E(k)$ the maximum entailment probability, and $v_k$ its verdict. The aggregate vector is
\begin{equation}
  \psi(x) = [\max_k C(k), \overline{C}, r_{\text{con}}, r_{\text{sup}}, r_{\text{unsup}}, \min_k E(k), \overline{E}, K],
\end{equation}
with
\begin{equation}
  r_{\text{con}} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}[v_k = \text{contradicted}], \quad
  r_{\text{sup}} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}[v_k = \text{supported}], \quad
  r_{\text{unsup}} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}[v_k = \text{unsupported}].
\end{equation}
Here $x=(q,c,a)$ is the input triplet, and $\overline{C}$ and $\overline{E}$ are the means of $C(k)$ and $E(k)$ over the claims. For $K>0$, the three ratios sum to one. All three are kept for direct interpretation, and the ablation in Section~\ref{sec:results} shows that the block adds no aggregate F1 gain. The model input has 35 dimensions: the 26 base features (two of them log-scaled), the eight claim features, and one corpus indicator. The claim count $K$ is kept because an answer with many claims is harder to verify.
```

Why this is accurate: `claim_features.py:141-181` defines `claim_max_contradicts` as the maximum over all pairs, `claim_mean_contradicts` as the mean of the per-claim maximum, `claim_min_entails` as the minimum of the per-claim maximum entailment, and the ratios from the support-first verdicts. The constraint directions are in `run_b6_modified.py:88-102`. The 35-feature composition is in `b6_feature_names.json` m3. The "no aggregate F1 gain" matches §5.2 ("the claim block costs 0.0010 F1, about the seed standard deviation").

### R2. Replace the third-change and zero-shot paragraph (line 413)

```latex
Third, the training set adds the RAGTruth summarization and data-to-text rows. A binary corpus indicator is one for those rows and zero for HaluEval, so the classifier can separate the two sources. The classifier never sees RAGTruth QA rows, so its QA ranking is zero-shot. The display calibrator does see labeled QA rows in the calibration subsection below, so the display score is not zero-shot. The combination is called EC-XGB. Seed 42 is the deployed checkpoint. The other seeds are a stability check.
```

Why this is accurate: `run_b6_modified.py:247-249` selects only RAGTruth summarization and data-to-text train rows; lines 306-309 set the source indicator to 1 for those rows and 0 for HaluEval; QA rows are excluded from `X_train`. The deployed artifact is `xgboost_m3_seed_42.joblib`.

### R3. Replace Algorithm 2 (lines 417-434)

```latex
\begin{algorithm}[H]
  \caption{Evidence-consistent XGBoost (EC-XGB) training and deployment}
  \label{alg:ecxgb}
  \begin{algorithmic}[1]
    \Require HaluEval QA training rows, HaluEval validation rows, RAGTruth summarization and data-to-text training rows, RAGTruth QA calibration rows, seeds $\mathcal{S}=\{42,123,456\}$
    \Ensure seed-42 model and display calibrator, source and target thresholds, per-seed metrics
    \State extract the 26 base features and the eight claim features for every row
    \State log-scale the character and word counts, and attach the monotone constraints
    \For{each seed $s \in \mathcal{S}$}
    \State build the 35-feature input, with the corpus indicator at zero for HaluEval and one for RAGTruth
    \State train XGBoost on the HaluEval rows plus the RAGTruth non-QA rows, and keep every RAGTruth QA row out of classifier training
    \State fit the Platt display calibrator on the RAGTruth QA calibration rows
    \State select the source threshold on the HaluEval validation split
    \State select the target threshold on the QA calibration rows at a 5\% false-positive budget
    \EndFor
    \State \Return the seed-42 model and calibrator, both thresholds, and the per-seed metrics
  \end{algorithmic}
\end{algorithm}
```

Why this is accurate: `fit_ec_xgb_calibrator.py` uses the 5,034-row QA calibration split; `run_b6_modified.py:369` selects the source threshold on HaluEval validation; the target threshold is selected at `alpha=0.05` (`DEFAULT_ALPHA`). The algorithm now matches the sentence that follows it and removes the undefined P0/P1/P2 labels.

### R4. Replace the source-indicator sentences in §3.4 (line 444)

```latex
The deployed EC-XGB display score uses a Platt calibrator fitted on the 5{,}034 RAGTruth QA calibration rows. Those rows are scored with the corpus indicator set to one. The deployed API uses the same value for every request, so the indicator cannot separate natural from synthetic answers at serving. The source-only protocol (P0 in Table~\ref{tab:protocols}) fits both the calibrator and the threshold on the HaluEval validation split.
```

Why this is accurate: `src/api/main.py:89` sets `EC_SOURCE_VALUE = 1.0`, and line 613 assigns it to every request. `fit_ec_xgb_calibrator.py:136` sets `frame[SOURCE_COLUMN] = 1.0` for the calibration rows. The protocols table defines P0 on the HaluEval validation split.

### R5. Replace the 35-feature sentence (line 412)

```latex
The model input has 35 dimensions: the 26 base features (two of them log-scaled), the eight claim features, and one corpus indicator. The claim count $K$ is kept because an answer with many claims is harder to verify.
```

Why this is accurate: `b6_feature_names.json` m3 lists 35 names: 26 base, 8 claim, 1 `source_ragtruth`. `LOG_LENGTH_COLUMNS = ("n_chars","n_words")` is the only log transform.

### R6. Replace the calibration paragraph (lines 440-444)

```latex
Source calibration uses the HaluEval validation split. Platt scaling fits a sigmoid $p(x) = \sigma(A f(x) + B)$ to the raw score $f(x)$, where $A$ and $B$ are fit by maximum likelihood \cite{platt}. Isotonic regression fits a non-decreasing step function $g$ so that $p(x) = g(f(x))$ \cite{isotonic}. Both are compared with the raw score on the HaluEval test split. The comparison reports expected calibration error over 10 equal-width bins,
\begin{equation}
  \mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|,
\end{equation}
where $\mathrm{acc}(B_m)$ and $\mathrm{conf}(B_m)$ are the empirical positive rate and mean confidence in bin $B_m$, the Brier score $\mathrm{BS} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)^2$, and reliability diagrams \cite{guo}. Post-hoc calibration degrades when the target distribution moves \cite{gasparin2024distance}.

The second protocol fits the calibrator on labeled target rows and evaluates it on a disjoint target test split. Section~\ref{sec:results} varies the label budget. The deployed EC-XGB display score uses a Platt calibrator fitted on the 5{,}034 RAGTruth QA calibration rows. Those rows are scored with the corpus indicator set to one, and the API uses the same value for every request. This target calibration needs labeled target rows and is not zero-shot. The source-only protocol (P0 in Table~\ref{tab:protocols}) fits both the calibrator and the threshold on the HaluEval validation split. The deployed calibrator is Platt. An earlier deployment fit compared Platt and isotonic on the test split, and Section~\ref{sec:results} records that choice.
```

Why this is accurate: `train_pipeline.ece` uses 10 equal-width bins; `fit_ec_xgb_calibrator.py:145-167` fits Platt and isotonic and chooses Platt by test ECE; `b6_calibration.json` records `chosen: "platt"`; the limitations paragraph in Section 5 already discloses the test-based choice. The replacement makes §3.4 consistent with the limitations instead of hiding the choice.

### R7. Replace the card paragraph (line 461)

```latex
The card's headline follows the claim verdicts when they are present. A contradicted claim sets high risk. An unsupported claim sets medium risk, and a fully supported answer sets low risk. The percentage is the display score. In the chat card it is adjusted by the claim mix when claims are present. In the analyze card it is the unadjusted Platt score.

Figure~\ref{fig:screens} shows the hallucinated-number case. The context says 5 days per decade and the answer says 10. The raw score is 0.999 and the target-calibrated score is 35.4\%, which the card rounds to 35\% and places in the medium band (0.35--0.60). The claim view marks the number claim unsupported, so the chat card also reports medium risk. The two labels agree in this case, but they come from different rules.

The claim-verdict audit is reported in Section~\ref{sec:results}.
```

Why this is accurate: `web/components/assistant-ui/auto-risk-card.tsx:333-340` sets `high_risk` if any claim is contradicted, `medium_risk` if any is unsupported, and `low_risk` otherwise. `src/api/main.py:1037-1046` adjusts the chat card score by `0.35*c_frac + 0.15*u_frac - 0.15*s_frac`. The analyze card uses `/predict`, whose label is `_risk_label(p_disp)` at 0.35/0.60 (`main.py:624-629`). The 0.354 score is just above 0.35, and the screenshot rounds it to 35%.

### R8. Replace the explainability paragraph (line 446)

```latex
A TreeExplainer computes exact SHAP values for the XGBoost model. Global importance is the mean absolute SHAP value over a stratified test sample. The explanation endpoint returns the five features with the largest absolute SHAP value for one prediction. Post-hoc attributions can be unstable even when the model is fixed \cite{kelodjou2024shap}. A 20-case pilot therefore tests reliability with six operators: entity, number, and date replacement, support removal, context insertion, and clause shuffling. The pilot reports the mean absolute score change, the fraction of large changes, the top-1 flip rate, and the Spearman rank correlation between the original and perturbed rankings.
```

Why this is accurate: `src/api/main.py:775-786` returns the top five features by absolute SHAP; the Results describe a 20-case, 120-attempt pilot with 48 changed cases (Explanation, Efficiency, and Judges, Section 5; `b5_perturbation_aggregates.csv`). The replacement stops implying that the whole test set was perturbed.

### R9. Replace the system/checkpoint paragraph (line 450)

```latex
The method is deployed in a web application with a Next.js front end and a FastAPI backend. The API loads the EC-XGB model, the Platt calibrator, the SHAP explainer, and the NLI, spaCy, and MiniLM feature models at startup. The deployed checkpoint is the 35-feature m3 model with seed 42. Section~\ref{sec:results} compares it with the conventional B2 checkpoint and with B6 m0, a 26-feature reference from the same tuning protocol that lacks the EC-XGB changes. Claims are split from the answer and checked with sentence-level NLI against the provided context, a local passage index, or a web search API. A claim is supported when any retained sentence has an entailment probability of at least 0.5 and at least as high as its contradiction probability. A contradiction decides only when no sentence supports the claim. A relevance gate drops sentences with low claim-token coverage and no shared content word. The verdicts are shown per claim.
```

Why this is accurate: `src/api/main.py` loads the EC model, display calibrator, explainer, and feature models; `/verify` supports `context`, `index`, and `web` evidence modes; `src/claims/verify.py:99-130` implements the support-first rule. B6 m0 has 26 base features and none of the EC-XGB changes.

### R10. Replace and relocate the contribution paragraph (line 436)

Move this paragraph to the end of §3.5, after the system description:

```latex
The contribution is the training and evaluation protocol. The learner is standard XGBoost. EC-XGB combines three standard changes: log-scaled length counts with monotone constraints, claim-level NLI aggregates for explanation, and multi-source training with a corpus indicator. Section~\ref{sec:results} measures what each change contributes. The claim aggregates are kept for per-claim explanation. They do not add an aggregate F1 gain. The seen-task gain comes from the m3 stack that adds the RAGTruth rows and the corpus indicator. The two are not separated. The risk score needs no LLM judge calls. The optional claim judge is a separate verification step, and the reported EC-XGB results do not use it.
```

Why this is accurate: §5.2 reports that the claim block costs 0.0010 F1 and the seen-task gain comes from the multi-source rows; the P0/P1 reviewer report recommends saying the m3 stack bundles the rows and the indicator; `/verify` defaults to `judge_uncertain=True`, so the qualified sentence is necessary.

## Text-weight and readability assessment

The section carries **approximately 1,980 words of prose in 115 sentences**, with a mean of 17.0 words per sentence. Nine sentences exceed 30 words and two exceed 35. The longest is the algorithm walk-through at 49 words. Five paragraphs exceed the guideline's 2-7 sentence range:

| Block | Approx. words | Sentences | Main problem |
|---|---:|---:|---|
| §3.1 pipeline | 253 | 17 | Stage-by-stage figure narration. It previews §3.2, §3.4, and §3.5, then repeats the system in §3.5. |
| §3.1 task | 89 | 5 | Long first sentence; "probability of a hallucination" is redundant; the negative parallelism repeats. |
| §3.2 features | 314 | 20 | A catalog that Table 1 already carries. It repeats the 26/8 count, explains every group, and previews SHAP results. |
| §3.3 changes | 263 | 15 | The densest block. It mixes three changes, five sub-changes, the equation, dead notation, and the zero-shot caveat. |
| §3.3 algorithm | 119 | 7 | One 49-word sentence with five clauses; "QA zero-shot" is imprecise. |
| §3.3 contribution | 155 | 8 | Misplaced before calibration/explainability; defensive comparisons; overstated "no LLM judge" claim. |
| §3.4 calibration | 204 | 11 | Source/target terms undefined; "small sample" conflicts with 5,034 rows; the test-based choice is omitted. |
| §3.4 explainability | 126 | 6 | Passive voice; "Mean Absolute Importance" is nonstandard; the pilot scale is missing. |
| §3.5 system | 173 | 10 | Checkpoint/table routing; "B6 m0 under the EC-XGB protocol" is misleading. |
| §3.5 figure | 79 | 7 | Repeats the system description from the previous paragraph and from §3.1. |
| §3.5 card | 207 | 9 | Mixes UI description, example, limitation, 100-claim audit, and comparison-panel detail. |

**Flow and figures.** The subsection order is sound: working principle, features, the learner, calibration/explainability, system. Figure 1 is introduced at the right moment, Figure 2 follows the system description, and Figure 3 anchors the card example. The problem is not the order but the duplication: §3.1 narrates every stage that §3.2-§3.5 explain again, and §3.5 repeats the system description twice. The captions are too thin to carry the layout ("End-to-end pipeline of EC-XGB", "Deployment architecture of the HaluRISC system"), so the prose does the work of the caption. Put the band labels in the captions and cut the narration.

**Where the weight sits.** The section is not too long for an 18-page paper, but the allocation is wrong. The working principle and feature catalog take 567 words, while the calibration protocol (204 words) and the contribution statement (155 words) are compressed. The system subsection takes 459 words across three paragraphs, but most of it is table routing, UI description, and reproducibility meta rather than method.

**Repetition.** "35" appears six times, "26" four times, "source" ten times, "calibrat*" 25 times, and "startup" three times. The Next.js/FastAPI startup sentence appears three times: §3.1 ("A Next.js front end calls a FastAPI service ..."), §3.5 ("The system is a Next.js frontend and a FastAPI backend ..."), and §3.5 again ("The browser calls the Next.js BFF ... FastAPI loads ..."). The RAGTruth QA calibration split appears in §3.1, §3.3, and §3.4; the 5,034-row count appears in §3.3 and §3.4. The 35-feature count appears in §3.1, §3.3, the algorithm, and the pipeline figure. The feature catalog repeats the table and the EC-XGB subsection.

**What to cut.** A 20-25% cut (400-500 words) is feasible without losing method content:
- Cut the figure narration in §3.1 (~100 words): "reads from left to right", "first stage", "lower band", "rightmost block", "training band". Put the band labels in the caption.
- Cut the per-group catalog in §3.2 (~80 words). Keep one sentence per group with the rationale; move the exact feature names to Table 1.
- Cut the defensive comparisons in §3.3 (~80 words): "Their scores come from other corpora ...", "None of the three reports ...", and the "closest detector" positioning. Move them to the discussion in Section 5.
- Cut the checkpoint/table routing, the 100-claim audit, the comparison-panel sentence, and the audit-script sentence in §3.5 (~150 words). Move the audit to Results and the checkpoint map to §4.3.
- Cut the duplicate system description in §3.5 (~97 words). Keep the figure paragraph, not the prose repetition.

**Readability.** The section is formal and mostly clear, but it reads like a specification. The "first stage / second stage / ..." enumeration, the nominalizations ("the addition of", "the comparison uses"), and the passive explainability paragraph slow the reader down. The prompt asks for active voice, one idea per sentence, and varied sentence shapes. The section has a few short punches ("They are weak on their own"), but they are outnumbered by long noun phrases. The student register is intact; there is no promotional language and no first-person voice.

## Feature and notation consistency check

| Item | Verdict | Detail |
|---|---|---|
| 26 base features | **Correct count, weak table** | 4 length/style + 4 lexical + 4 entity + 6 NLI + 4 numeric + 2 hedging + 2 semantic = 26. Table 1 does not show the per-row counts, so the reader cannot verify the total. Add a Count column. |
| 8 claim features | **Correct count, wrong label** | Seven NLI aggregates plus one claim count. The paper calls them "eight claim-level NLI aggregates"; the count is not an NLI aggregate. The three verdict ratios sum to one. |
| 1 source indicator | **Correct count, wrong semantics** | `source_ragtruth` is 0 for HaluEval and 1 for RAGTruth. The API sets it to 1 for every request. It is a corpus flag, not an answer property. |
| 35 total | **Correct in artifact** | `b6_feature_names.json` m3 has 35 names. The "24 base + 2 log-scaled length" breakdown conflicts with "26 base"; use "26 base (two log-scaled) + 8 + 1." |
| Log lengths | **Only two** | `LOG_LENGTH_COLUMNS = ("n_chars","n_words")`. `n_sentences` and `avg_word_len` are not log-scaled. The text's "raw-length counts" is too broad. |
| Monotone constraints | **13, not all** | 7 negative and 6 positive. `overlap_answer_question`, `cosine_q_ans`, and `nli_ans_entails_ctx` are unconstrained. The text's "overlap/entailment/semantic" is overbroad. |
| Claim aggregation | **Correct if C/E are per-claim maxima** | Code: `claim_max_contradicts` = max over pairs; `claim_mean_contradicts` = mean of per-claim max; `claim_min_entails` = min of per-claim max. The text never defines $C(k)$/$E(k)$ as per-claim maxima. |
| Verdict ratios | **Sum to one** | `unsupported = ~contradicted & ~supported`, so `r_con + r_sup + r_unsup = 1` for $K>0$. All three are kept; one is redundant. State this or drop one. |
| 5,034 calibration rows | **Correct** | `b6_calibration.json` has `calibration_rows: 5034`, `test_rows: 900`. The dataset section's 5,934 total minus 900 test gives 5,034. |
| Display calibrator | **Platt, chosen on test** | `fit_ec_xgb_calibrator.py:145-167` fits Platt and isotonic, chooses the lower test ECE. `b6_calibration.json` records `chosen: "platt"`. §3.4 presents the fit as clean; the limitations paragraph in Section 5 discloses the test-based choice. |
| Algorithm inputs | **Incomplete** | The `\Require` omits the 5,034 calibration rows; step 6 uses them. The output does not name seed 42. |
| B2 / B6 m0 / m3 | **Consistent but opaque** | B2 = 26-feature HaluEval-only; m0 = 26-feature reference; m3 = 35-feature multi-source. "B6 m0 under the EC-XGB protocol" is misleading. |
| SHAP | **Correct** | `TreeExplainer` on the raw model; global = mean absolute SHAP; local = top five by absolute SHAP. The perturbation is a 20-case pilot, not the full test set. |
| Card labels | **Two rules** | `/predict` label = score at 0.35/0.60; `/verify` headline = claim-based, percentage adjusted. The paper conflates them. |

## Top 3 fixes

1. **Reframe the contribution around multi-source training.** The section currently says three changes, but the third is a source indicator, and the multi-source rows appear only in the algorithm paragraph and the contribution paragraph. The Results say the seen-task gain comes from those rows. Rewrite §3.3 with **R1** and **R2**, and move the contribution paragraph to the end of §3.5 with **R10**. This is the biggest credibility problem because it makes the method section disagree with the paper's own conclusion.

2. **Make the claim features and the algorithm reproducible.** Define $C(k)$ and $E(k)$ as per-claim maxima, state the support-first rule before the equation, add the 5,034 calibration rows to Algorithm 2's `\Require`, and specify seed 42 as the deployed output. Use **R1** and **R3**. A reader who follows the current text cannot rebuild the eight features or the training procedure.

3. **Make the deployed-system description match the code.** Call the source indicator a corpus flag fixed to 1 at serving (**R4**), distinguish the `/predict` card from the `/verify` card and fix the 35.4%/35% example (**R7**), and disclose the earlier test-based Platt/isotonic choice in §3.4 (**R6**). These are the claims a reviewer can check in ten minutes by opening the API and the app.

## Summary

The section is well structured and technically sensible, and the headline feature counts, calibration artifact, and 35-feature model all check out. The problems are prose-to-artifact mismatches: the "three changes" framing omits multi-source training and overstates the claim features; the claim aggregation and Algorithm 2 are not reproducible as written; the source indicator, card example, and calibration description do not match the deployed code. The fixes are textual and structural, not new experiments. With R1-R10, the section would read as an honest methods section that a reviewer can follow from the features to the algorithm to the deployed system. Without them, the section invites the exact questions the Results already answer: what did the claim features contribute, which calibrator is deployed, and what does the card actually show.
