# Writing and Readability Review

**Manuscript:** `Journal_Paper/halurisc.tex`
**Build reviewed:** `Journal_Paper/out/halurisc.pdf` (25 pages, 18 tables, 12 figures, 2 algorithms)
**Scope:** prose quality, paragraph weight, clarity, terminology, reader flow
**Out of scope:** number provenance, method correctness, statistics, reference integrity
**Date:** 12 November 2026

---

## Verdict

**Major revision (writing and readability).** The manuscript is readable at the sentence level and the argument is followable, but it is not yet a readable journal paper. Section 5 reads like a 13-page table transcript: about 4,050 words, 47% of the body prose, with nine paragraphs of 200 words or more and a densest paragraph of 336 words. The abstract and conclusion repeat five claims almost verbatim, so the conclusion adds little interpretation. Several labels (B2, B6, m0, m3) are used before they are defined, and two names are used for the same source indicator. These are structural writing problems, not new experiments. A focused pass on Section 5, the abstract, the conclusion, and the terminology would move the paper to minor revision.

The current build is 25 pages, and Section 5 alone runs pages 10 to 23. I treat the page count as context, not as a separate finding, because the page budget belongs to another review.

I did not re-check any reported number. Quotations below are from the manuscript, not verified against the frozen artifacts.

---

## Checks completed: 12/12

| # | Check | Status | Main evidence |
| :-- | :--- | :--- | :--- |
| 1 | Paragraph weight (no substantive paragraph under ~90 words or ~6 lines) | **Fail** | L412 (conditional), L414, L416 (80, 80, 70 words) and L645 (88 words, 5 lines) |
| 2 | One clear job and a topic sentence per paragraph | **Fail** | Multi-job paragraphs at L292, L464, L513, L555, L668, L810, L854, L989; several open with a table or figure pointer |
| 3 | No ambiguous this / It / they | **Fail** | L358, L557, L710, L810, L989 |
| 4 | Terminology consistency (fixed-hp, raw score, source indicator, EC-XGB, B2, m0, m3) | **Fail** | B2/B6 undefined; m0/m3 defined 300 lines after first use; source vs corpus indicator |
| 5 | No sentence with more than two competing clauses | **Fail** | L400, L668, L810, L812 |
| 6 | Prose interprets, does not recite table numbers | **Fail** | L555, L668, L810, L812, L882, L927 |
| 7 | Transitions name real relationships, no formulaic count frames | **Fail** | L292, L400, L955, L989 |
| 8 | No AI-writing tells | **Fail** | L362 uniform rhythm; L271 generic justification; repeated negative parallelism; L955 participial tail |
| 9 | Student register, active voice, no promotion, no first person | **Partial** | One "We" at L271; passive stack at L483; mild promotion at L271 |
| 10 | Abstract and conclusion readable in one pass and not repetitive | **Fail** | Five near-verbatim repeats between L271 and L993; six-item method menu; 186-word conclusion with a number in almost every sentence |
| 11 | Section 5 not text-heavy, dense paragraphs split or trimmed | **Fail** | 4,050 words, 47% of body prose, nine paragraphs at 200+ words |
| 12 | Tables and figures carry the numbers, prose carries interpretation | **Fail** | Every results subsection narrates its table; L668 is the clearest case |

---

## Issues

Severity: **H** = high, **M** = medium, **L** = low. Line numbers refer to `Journal_Paper/halurisc.tex`. PDF pages refer to the current 25-page build.

| # | Sev | Check | Location and quote | Issue | Why it matters | Exact fix |
| :-- | :-- | :-- | :--- | :--- | :--- | :--- |
| W1 | H | 1 | L412, L414, L416, §3.3, p.5. "Here $x=(q,c,a)$..." (80 words, 4 lines). "Third, the training set adds..." (80 words, 5 lines). "Algorithm 2 lists the procedure..." (70 words, 4 lines). | Three consecutive short blocks. Two are stubs, and the third change gets less space than the first two. | The section's visual balance signals that the third change and the algorithm matter less, which is not the argument. | Move the two body claims from L412 ("The ablation..." and "The claim count $K$ is kept...") into L414. Then merge L416 into L414. The result is one ~180-word, 10-line paragraph, and L412 is left as a 50-word notation note. |
| W2 | M | 1 | L645, §5.2, p.11-12. "Figure 5 shows the seed-level spread..." (88 words, 5 printed lines). | Short figure paragraph between two long ablation paragraphs. | It reads as a caption and interrupts the Table 7 argument. | Merge L645 into L636. Split the combined 219-word block after "without moving F1 far." The first block is ~95 words, the second ~137 words. |
| W3 | H | 2, 7 | L292, Introduction P3, p.1-2. "The existing literature leaves four issues unaddressed." (288 words). | Four jobs in one paragraph: gap list, baseline limits, method description, evaluation and results preview. The first sentence is a count frame. | P3 is the argumentative pivot. The reader cannot map each gap to a design choice, and the four issues are never used as a structure again. | Split into three paragraphs of roughly 100, 100, and 88 words. Replace the four-issue frame with the two gaps that drive the method: calibration under shift and transfer of the probability. |
| W4 | H | 2 | L464, §3.5, p.7. "The claim-verdict audit is reported in Section 5. The requirements file pins every dependency..." | Two jobs in one printed paragraph. The reproducibility sentence has no blank line before it, so the UI case and the build note run together. | The topic sentence ("Figure 3 shows...") does not cover the build note, and the reader is pulled out of the example. | Move "The requirements file..." (23 words) to L453, which becomes 174 words. Move "The percentage is the display score..." from L462 into L464. L464 stays ~123 words and 7 lines, and L462 stays ~107 words. Do not simply cut the note, because L464 would drop to 93 words and about 4 lines. |
| W5 | H | 2 | L513, §4.3, p.9. "Twelve models are tested... A majority classifier... complete the set; the two judges..." (236 words). | Baseline inventory, threshold choice, calibration fitting, metric precision, and significance plan in one block. | §4.3 is the reference readers return to while reading §5, but the protocol details are buried after the model list. | Split after "so probability metrics are undefined" (~120 and ~116 words). Keep the baseline list first and the protocol and metrics second. |
| W6 | H | 2, 6, 11 | L555 (312 words), L668 (336), L810 (304), L854 (295), L989 (199), also L597, L710, L882, L927, L955, §5.1-§5.6. | Each paragraph mixes three to six jobs: ranking, calibration, controls, operating points, audit, limits. | These are the paper's main results, but they cannot be scanned. The main finding is often in the middle. | Split L555 after "the two summaries disagree" (~150/162). Split L668 after "Figure 7..." (~120), then after "P2 is chance-level" (~120/96). Move L810's test definitions to §4.4, then split McNemar (~120) and bootstrap (~100). Split L854 after "less stable to content edits" (~130/165). Split L597 after "within seed noise" (~120/96), L710 after "still slightly below nominal" (~110/138), L882 after "Feature extraction, not the model, is the cost" (~110/159), L927 after "both are near chance" (~120/122), and L955 after "only two negative rows" (~110/124). Trim L989 to ~134 words: keep the retrieval limit, the audit-sampling limit, and the display-calibrator disclosure, move the HaluEval saturation and judge-domain points to the Conclusion or delete them as duplicates, and delete the repeated 99.9/61.3 sentence. |
| W7 | H | 3 | L358, §3.1, p.3. "Figure 1 shows the pipeline that computes it." | "it" can be evidence consistency, the probability, or the flag. | This is the sentence that introduces the pipeline, so the ambiguity blocks the figure. | "Figure 1 shows the pipeline that computes this probability." |
| W8 | H | 3 | L810 "That is the primary comparison..." after $p=0.45$ and $p=0.789$. L710 "It hides the class split..." after "the marginal guarantee holds." L989 "so they are context rather than a head-to-head bar." L557 "(0.490 vs 0.549)" with no model labels. L668 "the card" assumes the reader remembers §3.5. | Pronouns and values with two possible referents. | The reader has to guess which test, which guarantee, which model, and which card. | Name the referent every time: "The fixed-hp m0-vs-m3 comparison is the primary test...", "The marginal guarantee hides the class split...", "so these published numbers are context...", "EC-XGB 0.490 vs the encoder 0.549", "the risk card in §3.5." |
| W9 | H | 4 | L453, §3.5, p.6. "the conventional B2 checkpoint and with B6 m0." | B2 is never expanded, and B6 appears in "B6 m0" and "B6 variants" without an expansion. m3 is first used at L294 and m0 at L453, but the ladder is defined only at L597. | An outside reader cannot map the labels to models, so every results table is opaque. | At first use, write "B2 (the 26-feature conventional checkpoint)" and "B6 m0 (the fixed-hp reference)." Define the m0 to m3 ladder in §3.3, before the results. |
| W10 | H | 4 | L447 "the corpus indicator set to one" and L475 "a corpus indicator" versus L271 and L414 "source indicator." "Source" also names the source domain in P0 and source calibration. | Two names for one feature, and one word with two meanings. | The reader may think the indicator encodes corpus identity, when it is a binary HaluEval/other flag. | Standardize to "source indicator." Define it as 1 for non-HaluEval rows and 0 for HaluEval. Rename P0 "in-domain (P0)" in prose and in Table 4 once, so "source" is reserved for the feature. |
| W11 | M | 4 | L649 "the uncalibrated probability $s$, called the raw score in the figures." L328 "uncalibrated probability." L340 "raw score." L854 "raw margin." L823 "fixed hp." L412, L597, L636, L854 use "block," "aggregates," and "features" for the same claim features. | The paper uses raw margin, raw score, uncalibrated probability, display score, and target-calibrated score without defining the set. Minor labels are inconsistent. | "Raw score" can mean the margin $z$, the sigmoid probability, or the pre-calibration score. The reader cannot tell which. | Define once in §2.1: raw margin $z$, uncalibrated probability $\sigma(z)$, display score after calibration. Use "raw score" for only one of them. Fix "fixed hp" in Table 12. Use "claim features" throughout and define "claim-feature block" once if needed. |
| W12 | H | 5 | L668 "A prevalence-only predictor at 0.1778... dominated by the prevalence gap." (48 words, five clauses). L810 "McNemar's test compares... the statistic uses a continuity correction." (43 words, three where-clauses). | Too many competing clauses in the sentences that carry the interpretation. | The reader loses the subject and skips the definition. | L668 into three sentences: prior at test prevalence, prior at calibration prevalence, conclusion about the gap. L810 into four sentences: the test, count $b$, count $c$, continuity correction. |
| W13 | M | 5 | L400 "The constraints are negative on... and positive on..." (40 words, two long lists). L812 "HaluEval groups are item pairs... not unseen-task transfer." (46 words, four clauses). L513 "The external ensemble uses five features... so the fixed 0.5 threshold does not change the comparison." (44 words, three clauses). L797 "All evaluation sets are balanced 50/50... fit once on HaluEval train only." (42 words, four clauses). | Lists and bookkeeping packed into one sentence. | The direction of the monotone constraints is easy to misread, and the group counts and protocol details bury the interpretation. | L400 into two sentences, one for negative constraints and one for positive constraints. L812 into group counts (one sentence) and interpretation (one sentence). L513 into the ensemble description and the threshold check. L797 into the balance caveat and the HaluBench/length-only caveat. |
| W14 | H | 6, 12 | L668 recites Table 8: 0.731, 0.689, 0.18, 0.125, 0.161, -0.10, 0.146, 0.134, 0.164, 0.1778, 0.3107, 0.313, 5.0%, 5.8%, 15.7%, 0.076, 0.130, 0.124, 0.125. L810 and L812 recite Table 13 deltas and intervals. | The prose is a transcript of the table. | The reader reads the same numbers twice and still misses the interpretation. | Keep three numbers in L668: 0.731 to 0.125, BSS -0.10, and zero flagged rows. Replace L810 and L812 with "no F1 delta excludes zero, and the AUROC deltas are positive but tiny" and let Table 13 carry the intervals. |
| W15 | M | 6, 12 | L555 "random forest is 0.0004 behind... EC-XGB is 0.0009 ahead." L927 "GPT... 0.565 and 0.910; Qwen... 0.602 and 0.778; EC-XGB... 0.310 and 0.558." L882 "127.5 ms... p95 234.6... 43.8 ms..." | The same table-transcript pattern in three subsections. | §5 reads as a data dump, and the prose adds no value over the tables. | One headline number and one interpretation per paragraph. Let Tables 5, 15, 16, and 17 carry the rest. |
| W16 | H | 7 | L400 "changes the conventional setup in three places. First... Second... Third..." L955 "Four controls test this explanation." L989 "Several limits bound these claims." | Count frames. The "Third" in L414 arrives 14 lines later, after two equations. The other two frames are vague. | They read as generated transitions and flatten specific content. | Name the changes ("length scaling and monotone constraints," "claim features," "multi-source training") and drop First/Second/Third. Introduce the controls by purpose. Replace "several limits" with the specific limits in separate sentences. |
| W17 | H | 8 | L362, §3.2, p.4. "Length is included because... Lexical overlap captures... A novel entity is... The NLI features are... A changed number is... Hedging words are... Semantic similarity catches..." | Seven consecutive sentences with the same shape and similar length. | This is the strongest AI-rhythm tell in the paper. It reads as a bulleted list flattened into prose. | Merge into three or four sentences with varied subjects. Give one concrete example (a changed number) and cut the rest. |
| W18 | H | 8, 9, 10 | L271, Abstract. "We present EC-XGB..." "A feature model is cheap and explainable... practical option for routine screening." "evaluated with fixed-hyperparameter ablations, source and target calibration, conformal prediction, SHAP stability, a length-only control, and judge stress tests." | First person, a generic justification, and a six-item evaluation menu. | The abstract breaks the formal register and is hard to read in one pass. | "EC-XGB is an evidence-consistent XGBoost model..." Cut the "practical option" sentence. Split the evaluation list into two sentences, ablations and calibration in one, conformal, SHAP, length control, and judges in the other. Keep every item so the abstract still covers the method. |
| W19 | M | 8 | L358 "not truthfulness in general but evidence consistency." L475 "not a new learner." L668 "not the ranking." L989 "not a head-to-head bar." L993 "not a general-purpose detector." L955 "giving 79 pairs..." Paragraph-closing summary sentences at L668, L765, and L854. | Negative parallelism repeated about six times, one participial tail, and a habit of ending paragraphs on a summary sentence. | The pattern is a known AI tell and loses force with repetition. The closing summaries make every paragraph feel pre-packaged. | Keep the two strongest ("not a new learner," "not a general-purpose detector"), rewrite the rest as positive statements. Rewrite L955 as "The control yields 79 pairs (158 examples), evaluated over 1,000 bootstraps." End some paragraphs on the last fact and cut the summary sentence. |
| W20 | M | 9 | L483, §4.1, p.7. "All experiments were run... A randomized search was performed... Each seed is tuned separately..." | Three passive or agentless sentences in a row. | The setup reads like a lab report and is longer than needed. | "All experiments ran on... The search covered 30 configurations... Each seed was tuned separately..." Keep passive only where the actor does not matter. |
| W21 | H | 10 | L271 versus L993. "lexical overlap and length carry that score" appears in both. "flags every row and is badly calibrated" appears in both, with ECE 0.493. "domain-specific risk score, not a general-purpose detector" appears in both. "ECE from 0.731 to 0.125" and "positive coverage is only 0.685" are also repeated. | Five near-verbatim claims, plus the same F1/AUROC and task-shift results. | The conclusion adds no interpretation. A reader who read the abstract learns nothing new. | Cut the repeated results. Make the conclusion answer two questions: what does this mean for deployment, and what should be tested next? Keep at most one repeated number. |
| W22 | H | 11 | §5, p.10-23. 4,050 words, 47% of body prose. Nine paragraphs at 200+ words: L555 312, L597 216, L668 336, L710 248, L810 304, L854 295, L882 269, L927 242, L955 234. Densest: L668, 336 words, 17 printed lines. | Section 5 is text-heavy, and the densest paragraphs are walls of numbers. | The results are the paper's contribution, but the reader cannot scan them. Tables and text compete. | Split the nine paragraphs at the seams in W6, target 150 to 180 words per paragraph, and trim the recitation as in W14. Move test definitions to §4.4 and values to the tables. |
| W23 | H | 12 | §5.1-§5.6, Tables 5-18. L555 (Table 5), L597 (Table 6), L636 (Table 7), L668 (Table 8), L710 (Table 9), L737 and L765 (Table 10), L792 and L797 (Table 11), L810 (Table 12), L810 and L812 (Table 13), L882 (Tables 15-16), L927 (Table 17), L955 (Table 18). | The prose recites table values instead of interpreting them. | Duplication lengthens the paper and hides the interpretation. It also invites inconsistencies between text and table. | One headline number plus one interpretation per paragraph. Let the table carry the rest. Use the prose to say what the pattern means, where it breaks, and why. |

---

## Complete short-paragraph list

All body paragraphs of 100 words or fewer, in manuscript order. Printed line counts are from the current PDF. Equation lead-ins, equation continuations, and acknowledgements are excluded per the brief. L406 is a one-word formula connector ("with"), not a paragraph.

| Tex line | Section | Words | Printed lines | Status | Merge target or action |
| :-- | :--- | --: | --: | :--- | :--- |
| 288 | Introduction P1 | 88 | 6 | **Pass at threshold** | No action required. Optional: move the gap sentence from L292 into this paragraph to give it more weight. |
| 306 | §2.2 | 42 | n/a | Excluded, Eq. 1 lead-in | None. |
| 310 | §2.2 | 51 | n/a | Excluded, Eq. 1 continuation | None. |
| 312 | §2.2 | 35 | n/a | Excluded, Eq. 2 lead-in | None. |
| 316 | §2.2 | 30 | n/a | Excluded, Eq. 2 continuation | None. |
| 318 | §2.2 | 12 | n/a | Excluded, Eq. 3 lead-in | None. |
| 322 | §2.2 | 21 | n/a | Excluded, Eq. 4 lead-in | None. |
| 326 | §2.2 | 45 | n/a | Excluded, Eq. 4 continuation | None. |
| 402 | §3.3 | 34 | n/a | Excluded, Eq. 5 lead-in | None. |
| 412 | §3.3 | 80 | 4 | **Violation, conditional** | Move "The ablation..." and "The claim count $K$..." to L414. Leave only notation after Eq. 6. |
| 414 | §3.3 | 80 | 5 | **Violation** | Merge with L416. Combined, ~150 words and 9 lines. |
| 416 | §3.3 | 70 | 4 | **Violation** | Merge into L414. Keep "The QA rows stay out..." as the interpretation. |
| 441 | §3.4 | 60 | n/a | Excluded, Eq. 7 lead-in | None. |
| 445 | §3.4 | 30 | n/a | Excluded, Eq. 7 continuation | None. |
| 530 | §4.3 | 99 | 7 | Pass | No action required. |
| 645 | §5.2 | 88 | 5 | **Violation** | Merge into L636. Split the combined block after "without moving F1 far" (95 and 137 words). |
| 654 | §5.3 | 49 | n/a | Excluded, Eq. 13 continuation | None. |
| 659 | §5.3 | 17 | n/a | Excluded, Eq. 14 note | None. |
| 696 | §5.3 | 35 | n/a | Excluded, Eq. 15 lead-in | None. |
| 703 | §5.3 | 12 | n/a | Excluded, Eq. 16 lead-in | None. |
| 792 | §5.3 | 71 | n/a | Excluded, Eq. 17 lead-in | The logical block continues at L797, so the real block is 272 words. See W6. |
| 1014 | Acknowledgements | 22 | n/a | Excluded, acknowledgements | None. |

**Clear violations: 3 (L414, L416, L645). Conditional: 1 (L412, because it mixes equation notation with two body claims).** L288 is 88 words but 6 printed lines, so it sits at the threshold and passes.

---

## What is working

- Sentence-level grammar and punctuation are generally clean. There are no em dashes in prose, no "not only... but also," no "serves as" puffery, and no uncited "studies show."
- The paper is honest about negative results. "EC-XGB is therefore a domain-specific risk score, not a general-purpose detector" is the right conclusion, and the prose does not oversell the method.
- Several topic sentences are strong and should survive the revision: "Source Platt cannot repair the base-rate shift" (L668), "The task-shift test is the weakest" (L765), and "The contribution is the training and evaluation protocol, not a new learner" (L475).
- The tables and figures are well formed. The problem is duplication in the prose, not missing evidence.
- The abstract answers the six guideline questions in order. The issue is readability and duplication, not coverage.

---

## Top 3 fixes

1. **Rebuild Section 5 around interpretation, not transcription.** Split the nine long paragraphs, starting with L668, L555, L810, L854, L882, L927, and L955. Keep one headline number per paragraph and move the rest to the tables. This is the largest readability gain and it addresses checks 2, 6, 11, and 12 at once.
2. **Rewrite the abstract and conclusion as two different documents.** Remove "We present," cut the six-item evaluation menu, and replace the conclusion's number list with the deployment implication and two concrete next tests. Keep at most one repeated number. This addresses checks 9 and 10 and removes the strongest repetition in the paper.
3. **Define the labels before using them and merge the short Section 3.3 blocks.** Gloss B2 and B6 and the m0 to m3 ladder at first use, standardize "source indicator" and "raw score," merge L414 with L416, and move the non-notation sentences out of L412. This addresses checks 1 and 4 and makes the results tables readable without the project's internal vocabulary.

---

*Review method: paragraph word counts were extracted from the tex, excluding equations, tables, figures, algorithms, captions, and acknowledgements. Printed line counts were read from the current PDF. The abstract and conclusion were compared sentence by sentence. The manuscript was scanned for first person, semicolons, em dashes, negative parallelism, participial tails, count frames, and repeated labels. No manuscript file was modified.*
