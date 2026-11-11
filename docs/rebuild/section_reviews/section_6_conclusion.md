# Section 6 review: Conclusion, Declarations, Acknowledgements

**Manuscript:** `Journal_Paper/halurisc.tex` (current build: 28 pages, `Journal_Paper/out/halurisc.log`)
**Scope:** `\section{Conclusion}` at line 1062 through `\end{document}` at line 1095
**Review date:** 2026-11-11
**Verdict:** **Major revision** (local text and declaration fixes, no new experiments)

The conclusion is numerically accurate and structurally compliant. Every headline number traces to the Results or the frozen artifacts, the three number audits pass (116/116, 298/298, 108/108), and the section introduces no new references. The problems are framing and verification: the conclusion does not state the research objective, two sentences claim more than the Results support, one sentence mixes two checkpoints, the closing term is undefined, and the code/data availability statements cannot be checked from the public repository. These are fixable in one editing pass.

---

## 1. Strengths

- **Numbers are trustworthy.** All 34 numeric tokens in the conclusion match the abstract where the abstract reports them, and the Results tables and artifacts otherwise. The rebuild, review, and P1 audits pass with zero mismatches. No number is invented and no new citation appears.
- **The central negative claim is correct.** The conclusion explicitly rejects a general-purpose detector, matching the abstract's "domain-specific risk score, not a general-purpose detector" and the paper's main finding. The wording of the closing term is a separate issue (M5); the negation itself is right.
- **The MedHallu sentence is genuine synthesis.** "the ranking survives the domain change while the probability does not" is the best sentence in the section. It states the paper's main lesson in one line, and it is supported by pair accuracy 0.867, AUROC 0.821, 100% flag rate, and ECE 0.493.
- **Voice and register are clean.** Active voice dominates, there is no first-person, no promotional language, no contractions, and no em dashes. The section is one paragraph, as required.
- **The declarations cover all five G13 categories** in the required order (funding, conflict, data, code, ethics), CRediT is printed for all four authors, and the five dataset licenses are correct: HaluEval MIT, RAGTruth MIT, FaithBench CC BY-NC-SA 4.0, MedHallu MIT, HaluBench CC BY-NC 2.0. The licenses were checked against the repository manifest and the public dataset cards.

---

## 2. Issues

### 2.1 Major issues

| # | Severity | Location (exact quote) | Issue | Why it matters | Suggested fix (exact replacement) |
|---|---|---|---|---|---|
| M1 | Major | L1064: "A useful hallucination detector must be fast, inexpensive, and explainable." | The sentence is a normative premise, not the research objective. G11 requires the conclusion to state the objective. The paper's objective is to test whether a simple feature model can estimate risk and how calibration and ranking change under shift. The opening frames "useful" around cost and explanation, while the paper's main finding is that these properties do not make the score a good QA detector. | A reader who starts at the conclusion never learns the research question. The section begins with an unsupported "must" and then reports results, so the logic runs backwards. The phrase is also a three-item list and a generic opener. | `This study tested whether a simple feature model can estimate hallucination risk and how its calibration and ranking change under domain and task shift.` |
| M2 | Major | L1064: "Cross-seed SHAP rankings are stable, so the explanation is reproducible." | Overclaim. §5.7 says the ranking is stable across seeds (Spearman 0.972 and 0.950), but the perturbation pilot shows it is less stable to content edits, and the AI-assisted audit judged 13 of 26 explanations implausible. §5.7 explicitly says "The explanations are diagnostic aids, not faithful attributions." Stability across seeds does not make an explanation reproducible in the sense a reader will infer. | The conclusion's claim conflicts with the section it summarizes. A reviewer who reads §5.7 after the conclusion will see the paper claiming more for its explanations than its own audit supports. This is the kind of inconsistency that erodes trust in the negative results. | `Cross-seed SHAP rankings are stable, but the perturbation pilot and the audit show that stability does not make the attributions faithful.` |
| M3 | Major | L1064: "On RAGTruth, the pooled flag rate falls from 99.9\% to 61.3\%, but held-out QA remains at 96.8\% and AUROC 0.558." | Mixes two checkpoints. The 99.9\% to 61.3\% pooled drop is the tuned EC-XGB row in Table 10. The 96.8\% flag rate and AUROC 0.558 are the fixed-hp model in Table 17 (the tuned QA row is 97.4\% and AUROC 0.549). The sentence does not label either checkpoint, and "remains at 96.8\%" omits the noun "flag rate." | A reader cannot tell which model the sentence describes, or whether 96.8\% is a flag rate, recall, or accuracy. The mixed checkpoint makes the transfer result look more coherent than it is. | `On RAGTruth the tuned model cuts the pooled flag rate from 99.9\% to 61.3\%, but the fixed-hp model still flags 96.8\% of held-out QA at AUROC 0.558.` |
| M4 | Major | L1064: "GPT reaches F1 0.565 on natural QA, but EC-XGB reaches 0.907 on the length-matched set against 0.865." | Apples-to-oranges comparison. The 0.565 is the GPT judge on 900 RAGTruth QA rows at 17.8\% prevalence. The 0.907 and 0.865 are EC-XGB and GPT on 158 HaluEval rows at 50\% positive, length-matched. The sentence leads with GPT's natural-prevalence result and then contrasts it with EC-XGB's length-matched result. §5.8 says the natural row is "a domain-shift result, not a general judge ranking" and the lower panel "controls length but not domain." | The sentence can make EC-XGB look stronger than GPT by leading with GPT's natural-prevalence result and EC-XGB's length-matched result, even though the two rows come from different datasets and prevalences. It is the comparison the Results warn against. | Cut the sentence. If the judge result must stay, replace it with: `On natural RAGTruth QA the GPT judge reaches F1 0.565 while EC-XGB is at 0.310. On the 158-row length-matched set the two are close (0.907 vs 0.865), so the natural-QA gap does not generalize.` |
| M5 | Major | L1064: "EC-XGB is therefore a domain-specific risk assessment and adaptation check, not a general-purpose detector." | "Adaptation check" is undefined and appears nowhere else in the paper. "Risk assessment" is a process, not a score. The sentence is the paper's take-home claim, so every noun must be precise. | The closing claim is vague at the exact point where the reader expects the clearest statement. A reader cannot tell what an "adaptation check" is or what it checks. The abstract already has the precise phrase, so the conclusion should use it. | `EC-XGB is therefore a domain-specific risk score, not a general-purpose detector.` |
| M6 | Major | L1076-1078: "Model artifacts, split indices, and the experiment manifest are included in the release archive." / "available in the project repository." | The availability statements are unverifiable. The public repository is `https://github.com/Atik203/HaluRISC`, but the paper gives no URL. The repository has no GitHub releases, `artifacts/models/*` is gitignored, and `artifacts/results/*` is gitignored. Split indices are tracked, but model artifacts are not. "Release archive" has no DOI, URL, or file name. | A reader cannot reproduce the deployed model or check the artifacts. This is a submission-readiness issue, not a science issue: the paper claims availability but does not say where. Journals routinely query this before acceptance. | Code: `The training pipeline, FastAPI inference service, and Next.js dashboard are available at \url{https://github.com/Atik203/HaluRISC} (branch \texttt{rebuild}, commit \texttt{aaf4025}).` Data: either cite a real archive DOI, or replace the last sentence with `Split indices are in the repository. Model artifacts and the experiment manifest are available from the corresponding author on reasonable request.` |

### 2.2 Minor issues

| # | Severity | Location (exact quote) | Issue | Why it matters | Suggested fix |
|---|---|---|---|---|---|
| m1 | Minor | L1064: "and the tuned checkpoint is 0.9855 and 0.9984." | Redundant with the fixed-hp headline, and the two bare numbers omit their metric names. The tuned values are slightly worse on AUROC and slightly better on F1, so they add noise without changing the story. | The sentence makes the reader compare four numbers where two would do. It also blurs which checkpoint is the headline. | Cut the clause, or write `the tuned checkpoint is essentially unchanged (0.9855 F1, 0.9984 AUROC).` |
| m2 | Minor | L1064: "while the claim block costs only 0.0010." | The direction is ambiguous. "The claim block costs 0.0010" can read as the block itself costs performance, when the intended meaning is that removing the block costs 0.0010 F1. | The sentence is about the contribution of the claim features, so the direction matters. The number is about one seed standard deviation, so "only" is defensible. | `while removing the claim block costs only 0.0010 F1.` |
| m3 | Minor | L1064: "Split conformal covers 94.4\% at a 90\% target, but positive coverage is only 0.685; a class-conditional quantile raises it to 0.887, and FaithBench coverage falls to 0.320." | The semicolon violates the style guide, and "raises it" can be misread as raising overall coverage from 94.4\% to 88.7\%, which is a decrease. The referent is positive coverage, but the sentence makes the reader work. | The conformal result is subtle: marginal coverage is fine, positive coverage is not. A misread here reverses the meaning. | `Split conformal covers 94.4\% at the 90\% level, but positive coverage is only 0.685. A class-conditional quantile raises positive coverage to 0.887, and FaithBench coverage falls to 0.320.` |
| m4 | Minor | L1064: the MedHallu sentence, 42 words: "On MedHallu the model ranks the paired hallucinated answer above the grounded one in 86.7\% of 1{,}000 pairs (AUROC 0.821) but flags every row and is badly calibrated (ECE 0.493), so the ranking survives the domain change while the probability does not." | The longest sentence in the section, with a "but" clause and a "so" clause. The style guide asks for 22 to 25 words maximum, and this is 42. | The sentence contains the paper's best synthesis, but the length buries it. Splitting it makes the contrast land harder. | `On MedHallu the model ranks the paired hallucinated answer above the grounded one in 86.7\% of 1{,}000 pairs (AUROC 0.821), but it flags every row and is badly calibrated (ECE 0.493). The ranking survives the domain change, but the probability does not.` |
| m5 | Minor | L1064: "Dialogue and summarization both fail to transfer, and summarization ranks below a length-only model." | Collapses two different failure modes. Dialogue flags 99.3\% of rows, so its F1 equals the all-positive baseline, but its AUROC 0.763 is above the length-only 0.724. Summarization is the one that ranks below length-only (0.717 vs 0.788). "Both fail to transfer" is true at the operating point, but a reader may infer both rank below length-only. | The Results carefully distinguish the all-positive failure from the ranking failure. The conclusion should keep that distinction, because it is the difference between "the threshold is wrong" and "the ranking is wrong." | `Dialogue collapses to an all-positive flag rate, and summarization ranks below a length-only model.` |
| m6 | Minor | L1064: "One analysis takes 127.5\,ms on GPU and 2.00\,s on CPU and costs about \$0.001 per 1{,}000 predictions." | "One analysis" is ambiguous; the paper's unit is one answer or one row. The 2.00 s CPU figure is an average from a single contended run (p50 0.98 s, p95 7.56 s), and the caveat is omitted. | A reader may quote 2.00 s as a stable CPU latency when the Results describe it as one contended run. The GPU figure is a median, not a mean. | `One answer takes 127.5\,ms on GPU and about 2.00\,s on CPU in a single contended run, and costs about \$0.001 per 1{,}000 predictions.` |
| m7 | Minor | L1064: sentences 3, 6, and 9 all start with "On": "On HaluEval QA...", "On RAGTruth...", "On MedHallu...". | Mirror openers are an AI-writing tell and make the paragraph feel like a list of results. | The repetition is easy to fix and improves flow without changing any claim. | Vary the entries: `HaluEval QA is saturated...`, `RAGTruth shows...`, `MedHallu is the clearest split...` |
| m8 | Minor | L1064: "Future work should cover domain adaptation, stronger multicalibration, and threshold transfer." | The three items are vague and only partly match the stated limitations. "Domain adaptation" is already attempted through multi-source training, so the future item should be adaptation to unseen domains. "Stronger multicalibration" and "threshold transfer" do not say what would be stronger or transferred. The limitations also name retrieval quality and end-to-end verdict accuracy, which the future work omits. | The conclusion is the only place that sets the research agenda. Vague items waste that space and do not connect to the paper's own limits. | `Future work should test adaptation to unseen QA domains, extend multicalibration over length and claim-count slices, select thresholds under prevalence shift, and evaluate retrieval end to end.` |
| m9 | Minor | L1080: "The research does not involve human subjects, and hence no ethics approval is needed." | Technically true in the regulatory sense, but the paper reports a three-author blinded review with AI pre-annotation and author-verified labels (L946, L1027). "Does not involve human subjects" can read as denying that any human annotation happened. "Hence" is also informal. | The paper's provenance is sensitive: the labels began as AI pre-annotations and were author-verified. The ethics statement should not create a new ambiguity about who annotated what. | `No human-subject data were collected. The three authors annotated public benchmark rows for internal verification; this does not constitute human-subjects research, so ethics approval was not required.` |
| m10 | Minor | L1076: "Raw evaluation datasets are not redistributed." | The repository tracks `artifacts/results/b5/b5_review_cases_reviewed.csv`, which contains 26 HaluEval question/context/answer rows. The full raw datasets are not redistributed, but case-level excerpts are. | The statement is true for the full datasets but not for every row in the repository. A precise statement avoids a reviewer asking why raw benchmark text is in the audit file. | `Full raw datasets are not redistributed. The repository includes download scripts, source revisions, SHA-256 hashes, and license notes ... and a 26-case audit file with benchmark excerpts.` |
| m11 | Minor | L1085: "The authors thank the individuals responsible for releasing the datasets used in this study (HaluEval, RAGTruth, FaithBench, MedHallu, and HaluBench) under open licenses." | "Open licenses" is imprecise: HaluBench is CC BY-NC 2.0 and FaithBench is CC BY-NC-SA 4.0, both non-commercial. "Individuals responsible" is vague; datasets are released by teams and organizations. | The acknowledgement should be accurate about the terms that actually apply. A non-commercial license is not "open" under the standard Open Definition. | `The authors thank the maintainers of HaluEval, RAGTruth, FaithBench, MedHallu, and HaluBench for releasing their datasets under licenses that permit research use.` |
| m12 | Minor | L1072 and L1074: "No external funding was provided to conduct this research." / "The researchers declare no conflict of interest." | The funding sentence is grammatically awkward, and "researchers" is inconsistent with "authors" elsewhere. The conflict sentence is weaker than the standard Elsevier wording. | These are standard declarations. Using the standard wording reduces editorial queries and matches the journal's expectations. | Funding: `This research received no external funding.` Conflicts: `The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.` |

No critical issues were found. The major issues are about what the section claims and whether a reader can verify it, not about arithmetic.

---

## 3. Text weight and readability

### 3.1 Measurements

| Metric | Value |
|---|---:|
| Words in the conclusion | 278 (the brief's 285 is close; the difference is LaTeX tokens such as `1{,}000` and `\%`) |
| Sentences | 14 |
| Mean sentence length | 19.9 words |
| Longest sentence | 42 words (MedHallu) |
| Sentences containing numbers | 9 of 14 |
| Numeric tokens | 34 |
| Numeric density | 12.2 per 100 words |
| Numeric density of the abstract | 7.5 per 100 words (21 tokens in 280 words) |
| Current PDF length | 28 pages |

The conclusion is not too long for a 28-page paper. At 278 words it is about 1.7% of the approximately 16,200 PDF tokens and slightly shorter than the 280-word abstract. A 250 to 300 word conclusion is normal for an empirical paper of this length. The problem is not raw length. The problem is that 10 of 14 sentences report Results, and the numeric density is 62% higher than the abstract's. The section reads like a second abstract with extra table rows, not like a synthesis.

A better target is **230 to 250 words, 10 to 12 sentences, numeric density below 9 per 100 words**. That is enough space for the objective, the contribution, the in-domain result and its cause, the ranking-versus-probability lesson, the explanation and cost, the significance, and the future work.

### 3.2 Sentence map

| # | Words | Function | Repeats Results? | Action |
|---:|---:|---|---|---|
| 1 | 10 | Normative premise | No | Replace with the research objective (M1) |
| 2 | 27 | Contribution: EC-XGB and components | No | Keep, tighten the component list |
| 3 | 20 | HaluEval F1/AUROC, fixed and tuned | Yes, §5.1/Table 5 | Cut the tuned clause (m1) |
| 4 | 22 | Ablation: lexical, length, claims | Yes, §5.2/Table 7 | Keep, fix "removing" (m2) |
| 5 | 10 | SHAP stability | Yes, §5.7 | Keep, qualify "not faithful" (M2) |
| 6 | 20 | RAGTruth pooled and QA | Yes, §5.4/Table 10 and Table 17 | Label checkpoints, add "flag rate" (M3) |
| 7 | 27 | Target calibration, constant predictor | Yes, §5.3/Table 8 | Keep, tighten |
| 8 | 27 | Conformal coverage and FaithBench | Yes, §5.3/Table 9 | Split, clarify "positive coverage" (m3) |
| 9 | 42 | MedHallu: ranking vs probability | Yes, §5.4/Table 11, plus synthesis | Keep the synthesis, split the sentence (m4) |
| 10 | 14 | Task shift: dialogue and summarization | Yes, §5.4 | Distinguish failure modes (m5) |
| 11 | 17 | GPU/CPU latency and cost | Yes, §5.8/Table 15 | Keep, cut the CPU clause (m6) |
| 12 | 17 | GPT judge vs EC-XGB | Yes, §5.8/Table 17 | Cut or reframe (M4) |
| 13 | 14 | Significance: domain-specific, not general | No | Replace "adaptation check" with "risk score" (M5) |
| 14 | 11 | Future work | No | Align with limitations (m8) |

Only sentences 1, 2, 13, and 14 do not repeat the Results. Sentence 9 repeats the Results but adds a synthesis, so it earns its place. The rest are recitals.

### 3.3 Cuts that do not lose the contribution

- **Cut sentence 12 (17 words).** The judge comparison is a stress test, not the paper's contribution. The abstract and §5.8 already report it. Cutting it removes the most misleading comparison in the section.
- **Cut the tuned-checkpoint clause in sentence 3 (9 words).** The fixed-hp headline is the abstract's number; the tuned values add nothing.
- **Cut the class-conditional 0.887 and FaithBench 0.320 numbers in sentence 8 (about 15 words).** Keep the 94.4% versus 0.685 contrast, which is the point. Add "and it does not transfer to FaithBench" if the negative result needs a mention.
- **Cut the CPU clause in sentence 11 (5 words).** Keep 127.5 ms and $0.001 per 1,000 predictions.
- **Compress sentence 9 from 42 to 28 words (saves 14).** Split it into a result sentence and a one-line synthesis.
- **Cut the 96.8% flag rate in sentence 6 (3 words), or keep it and cut AUROC 0.558.** The flag rate is the more intuitive failure; the AUROC is the ranking summary. Do not report both without saying which checkpoint they belong to.

Those cuts save about 60 words. Adding the objective (+16) and a short in-domain synthesis (+11) gives a net of about 245 words, with a much lower numeric density. That is the right trade.

### 3.4 Flow

The paragraph is ordered by Results subsection, not by argument: HaluEval, ablation, SHAP, RAGTruth, calibration, conformal, MedHallu, task shift, latency, judge, significance, future work. That is 12 moves in 14 sentences. The SHAP sentence sits between the ablation and RAGTruth, interrupting the in-domain-to-shift arc. The judge sentence sits after latency and before significance, where it reads as an afterthought.

A stronger order is four moves:

1. **Objective and contribution** (new sentence 1 + sentence 2).
2. **In-domain result and its cause** (sentences 3 and 4).
3. **Shift: ranking survives, probability does not** (sentences 6, 7, 8, 9, 10). Put the MedHallu synthesis at the end of this block, not in the middle.
4. **Explanation, cost, significance, future work** (sentences 5, 11, 13, 14). Move the SHAP sentence here, next to the cost sentence, because both are about what the system gives the user beyond the score.

This order keeps one paragraph and removes the "results annex" feel without adding words.

### 3.5 Readability verdict

The prose is clear sentence by sentence. The problem is paragraph-level: too many numbers, too many dataset names, and no signposting. A reader who knows the paper can follow it. A reader who starts at the conclusion cannot tell which result matters most until the MedHallu sentence, which arrives nine sentences in. The fix is to lead with the question and the lesson, then use the numbers as evidence.

One note on the page budget: `docs/rebuild/SCORE_ESTIMATE.md` still says 29 pages, but the current PDF is 28 pages. The text-weight judgment here uses 28. The difference does not change the conclusion.

---

## 4. Consistency check against the abstract and Results

All numbers below were checked against the abstract (L271) and the Results (L553-1060). The three number audits also pass: `rebuild_number_audit.py` 116/116, `review_number_audit.py` 298/298, `p1_number_audit.py` 108/108.

| Claim in the conclusion | Abstract | Results | Verdict |
|---|---|---|---|
| EC-XGB: 26 base features, eight claim-level NLI aggregates, source indicator | Same | §3.2, §3.3; 35 features total | Pass |
| HaluEval fixed-hp F1 0.9854, AUROC 0.9986 | Same | Table 6: 0.9854/0.9986 | Pass |
| Tuned checkpoint 0.9855, 0.9984 | Not in abstract | Table 5: 0.9855/0.9984 | Pass, but redundant (m1) |
| Lexical −0.0171, length −0.0066, claims −0.0010 F1 | 0.0171/0.0066 only | Table 7: −0.0171/−0.0066/−0.0010 | Pass; signs are implied by "costs" |
| SHAP stable across seeds, explanation reproducible | Not in abstract | §5.7: Spearman 0.972/0.950, top-10 9/10; content edits less stable; 13/26 implausible | Stability pass; "reproducible" overclaim (M2) |
| RAGTruth pooled flag 99.9% → 61.3% | Not in abstract | Table 10 tuned: 99.9% → 61.3% | Pass |
| RAGTruth QA flag 96.8%, AUROC 0.558 | AUROC 0.558 | Table 17 fixed-hp: 96.8%/0.558; Table 10 tuned: 97.4%/0.549 | Pass on values; mixed checkpoints (M3) |
| Target calibration ECE 0.731 → 0.125 | Same | Table 8: 0.731 → 0.125 | Pass |
| Constant predictor has lower Brier | Not in abstract | Table 8: 0.146 vs 0.161 | Pass |
| Conformal 94.4% at 90%, positive 0.685, class-cond 0.887, FaithBench 0.320 | 0.69 positive, 0.32 FaithBench | Table 9: 0.944/0.685/0.887/0.320 | Pass; "raises it" ambiguous (m3) |
| MedHallu 86.7% of 1,000 pairs, AUROC 0.821, ECE 0.493 | Same | Table 11: 0.867/0.821/0.493 | Pass |
| Dialogue and summarization fail to transfer; summarization below length-only | "collapses, with summarization below a length-only model" | §5.4: dialogue 99.3% flagged, AUROC 0.763; summarization 0.717 vs length-only 0.788 | Partial: dialogue ranking is above length-only; the failure is the flag rate (m5) |
| 127.5 ms GPU, 2.00 s CPU, $0.001 per 1,000 | Not in abstract | Table 15: 127.51 ms, 2002.15 ms; Table 16: ≈$0.001 | Pass; CPU caveat omitted (m6) |
| GPT F1 0.565 natural QA; EC-XGB 0.907 vs 0.865 length-matched | Same | Table 17: 0.565; 0.907/0.865 | Numbers pass; juxtaposition misleading (M4) |
| Domain-specific risk score, not general-purpose detector | Same | §5.4, §5.10; abstract | Pass; "adaptation check" undefined (M5) |

**No new information.** The conclusion introduces no result that is not in the Results. The tuned checkpoint values, the claim-block delta, the constant-predictor Brier comparison, the class-conditional coverage, the CPU latency, and the judge numbers all appear in §5. The only new wording is "adaptation check," which is not a new result but an undefined term.

**No new references.** The conclusion contains no `\cite`, which satisfies G11.

**No general-detector claim.** The final sentence explicitly rejects it. This is a strength, not a problem.

**No human-annotation overclaim in the conclusion.** The conclusion does not mention human annotation. The ethics statement is the only place where the wording could be read as denying the author annotation; see m9.

**One framing mismatch.** §3.4 says "The contribution is the multi-source training protocol and the evaluation, not a new learner" (L436), while the conclusion foregrounds "This research has created EC-XGB" and lists model components. The model is part of the contribution, so this is not false, but the conclusion could say "The contribution is the protocol and evaluation, packaged as EC-XGB" to match §3.4 exactly. This is optional; it is not in the issue table because it does not affect any number or claim.

---

## 5. Declarations and acknowledgements

### 5.1 What is correct

- **Order and coverage.** Funding, conflicts, data availability, code availability, and ethical approval are all present, in the G13 order. The declarations and acknowledgements are unnumbered, as the CAS class expects.
- **Licenses.** HaluEval MIT, RAGTruth MIT, FaithBench CC BY-NC-SA 4.0, MedHallu MIT, HaluBench CC BY-NC 2.0. These match the repository manifest for the first three and the public dataset cards for MedHallu and HaluBench.
- **CRediT.** All four authors have roles, including the supervisor (Ohidujjaman Tuhin: Supervision, Validation, Writing - review & editing). No author is missing a role, and no author claims Funding acquisition, which is consistent with the funding statement.
- **No ethics overreach.** The paper does not claim ethics approval it does not have. The issue is precision, not fabrication.

### 5.2 What is missing or inaccurate

- **No repository URL.** Both the data and code statements say "the repository" or "the project repository" without a URL. The public repository is `https://github.com/Atik203/HaluRISC`; the current branch is `rebuild`, and the current commit is `aaf4025`. Add the URL and the final branch or commit hash after the last code change.
- **No verifiable release archive.** The data statement says model artifacts, split indices, and the experiment manifest are "included in the release archive." The public repository has no GitHub releases, `artifacts/models/*` is gitignored, and `artifacts/results/*` is gitignored. Split indices are tracked in git, but the model artifacts are not. If the archive is a submission supplement, say so and name it. If it is a public deposit, give the DOI. If neither, use "available from the corresponding author on reasonable request."
- **"Raw evaluation datasets are not redistributed" is too broad.** `artifacts/results/b5/b5_review_cases_reviewed.csv` is tracked and contains 26 HaluEval question/context/answer rows. The full datasets are not redistributed, but the audit file contains excerpts. Say "full raw datasets" and mention the audit file.
- **Ethics wording.** "The research does not involve human subjects" is true in the regulatory sense, but the paper reports a three-author blinded review and author-verified labels. A precise statement should say that no human-subject data were collected and that the authors annotated public benchmark rows for internal verification. This avoids an editorial query without overclaiming.
- **"Open licenses" is imprecise.** HaluBench (CC BY-NC 2.0) and FaithBench (CC BY-NC-SA 4.0) are non-commercial. "Public licenses that permit research use" is accurate. If the release archive contains FaithBench-derived features, state that they inherit CC BY-NC-SA 4.0 or are not included.
- **Acknowledgement is generic.** "The individuals responsible for releasing the datasets" does not name anyone. If the authors want to thank specific maintainers, name them. If not, "the maintainers of ..." is sufficient. The supervisor is an author, so no supervision acknowledgement is needed.

### 5.3 Declaration verdict

The declarations are complete in structure and correct in substance, but two statements are not verifiable and one is too broad. The fixes are short and do not require new data. The acknowledgements are acceptable but could be more precise about licenses.

---

## 6. Top 3 fixes

1. **Rewrite the opening and cut the results recital.** Replace the "A useful hallucination detector must be..." premise with the research objective. Cut the judge sentence (M4), the tuned-checkpoint clause (m1), the class-conditional and FaithBench numbers (m3), and the CPU clause (m6). Compress the MedHallu sentence. Target 230 to 250 words. This fixes the missing objective (M1) and the text-weight problem in one pass.
2. **Make the conclusion say no more than the Results.** Replace "so the explanation is reproducible" with "stable, but not faithful" (M2). Label the RAGTruth checkpoints and add "flag rate" (M3). Replace "adaptation check" with "risk score" (M5). If the judge sentence stays, reframe it as natural QA versus length-matched, not GPT-natural versus EC-XGB-length-matched (M4). These are the claims a reviewer will check first.
3. **Make the declarations verifiable.** Add the public repository URL and branch/commit to code availability. Replace the "release archive" sentence with a real DOI or "available from the corresponding author on reasonable request." Change "raw evaluation datasets" to "full raw datasets" and mention the 26-case audit file. Tighten the ethics wording to acknowledge author annotation. These are the fixes an editor will check before sending the paper out.

---

## 7. Illustrative rewrite (260 words, one paragraph)

This is not a required replacement. It shows the target weight and order. It cuts the judge sentence, the tuned clause, the class-conditional and FaithBench numbers, and the CPU clause, and it adds the objective and a short in-domain synthesis. It is still 260 words; cutting the conformal sentence would bring it to about 248.

> This study tested whether a simple feature model can estimate hallucination risk and how its calibration and ranking change under domain and task shift. EC-XGB is an evidence-consistent XGBoost risk predictor with 26 base features, eight claim-level NLI aggregates, and a source indicator, trained on multi-source rows and paired with calibration and SHAP explanations. On HaluEval QA the fixed-hp model reaches F1 0.9854 and AUROC 0.9986, but lexical overlap and length carry that score: removing them costs 0.0171 and 0.0066 F1, and the claim block adds 0.0010. The in-domain score is therefore mostly a length and overlap effect. On RAGTruth the tuned model cuts the pooled flag rate from 99.9% to 61.3%, but still flags 96.8% of held-out QA at AUROC 0.558. Target Platt calibration lowers ECE from 0.731 to 0.125, but flags no target rows at 0.5 and loses to a constant predictor on Brier score. Split conformal covers 94.4% at 90%, but positive coverage is only 0.685. On MedHallu the model ranks the hallucinated answer above the grounded one in 86.7% of 1,000 pairs (AUROC 0.821), but flags every row and is badly calibrated (ECE 0.493). The ranking survives the domain change, but the probability does not. Dialogue collapses to an all-positive flag rate, and summarization ranks below a length-only model. One answer takes 127.5 ms on GPU and costs about $0.001 per 1,000 predictions. EC-XGB is therefore a domain-specific risk score, not a general-purpose detector. Future work should test adaptation to unseen QA domains, extend multicalibration over length and claim-count slices, and select thresholds under prevalence shift.

---

## 8. Summary

Section 6 is numerically accurate and structurally compliant, but it reads as a second Results section: 278 words, 34 numbers, 10 of 14 sentences reporting Results, and a 42-word sentence that buries the paper's best synthesis. The two claims to fix first are the GPT natural-QA versus length-matched comparison (M4) and the "reproducible explanation" claim (M2), because both say more than the Results support. The closing term "adaptation check" is undefined (M5), the RAGTruth sentence mixes two checkpoints (M3), and the code/data availability statements cannot be verified from the public repository (M6). Fix the opening objective, cut 50 to 60 words of recital, qualify the SHAP claim, label the checkpoints, and add the repository URL and a real archive or a reasonable-request statement. The fixes are local; no new experiments are needed. **Verdict: major revision, confined to Section 6.**
