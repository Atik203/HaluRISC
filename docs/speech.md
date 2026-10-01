# HaluRISC — Live Presentation Speech (3 members)

The full 12-minute video (slides, live demo, and journal results) is already
recorded. This file is the script for the **live presentation**, split by member.
Each member speaks for **2 to 3 minutes**, so the slide talk runs about 7 minutes
in total.

Every sentence connects to the one before it, and each member ends on a line
that hands over to the next. Read it as continuous speech, not as bullet points.
The pace is set for a slow, careful speaker at about 1.73 words per second, so
pause for one second after every number.

---

## Before you present

- **Open the deck at your first slide.** The deck supports a direct link, for
  example `/slide?slide=6` starts Member 2, and `/slide?slide=11` starts Member 3.
- **Keyboard:** right arrow or space to advance, left arrow to go back, **F** for
  fullscreen, **Home** and **End** to jump.
- **Stay on your own slides.** Member 1 covers 1 to 5, Member 2 covers 6 to 10,
  and Member 3 covers 11 to 16.
- **Pause after every number.** The times below already include the pauses.

---

## Timing plan

| Member | Slides | Time |
|--------|--------|------|
| 1 | Title, Introduction, Motivation and research gap, Objective, Dataset | 2:41 |
| 2 | Conventional method, Proposed method, Flow diagram, Experimental setup, Results: in-domain | 2:16 |
| 3 | Results: shift and calibration, Applications, UI, Conclusion | 2:08 |
| **Total** | | **7:05** |

---

## Member 1 — slides 1 to 5 (2:41)

### Slide 1 — Title (0:25)

> Hi everyone. Our team name is Phantom Devs, and the project is HaluRISC.
> The full title is: Evidence-Consistent XGBoost for Calibrated Hallucination Risk
> Estimation in LLM Answers.
> Our recorded video shows the working demo and the key results from our journal paper.

---

### Slide 2 — Introduction (0:51)

> Let me start with one example.
> A model was asked about the Arctic melt season, and it answered 10 days per decade.
> The evidence says 5.
> The sentence is fluent, but it is wrong, and that is the problem we study.
> We see only the question, the evidence, and the answer.
> So the answer must be judged from the text alone.
> Existing detectors need the model or a large judge model, and both are slow and
> costly.
> So we need a method that is accurate, calibrated, explainable, and fast.

---

### Slide 3 — Motivation and research gap (0:29)

> Existing work leaves three gaps open.
> First, calibration. Most lightweight detectors never report how reliable their score is.
> Second, explanations. Attribution plots are shown, but their stability is never tested.
> Third, domain shift. A detector that works on one dataset often fails on another.
> Our study covers all three together.

---

### Slide 4 — Objective (0:25)

> Now the objective.
> We estimate the probability that an answer is hallucinated from the question, the
> evidence, and the answer alone.
> Our contributions are an evidence-consistent feature set, three cumulative changes
> over the baseline, a calibration study under shift, and a deployed verification
> system.

---

### Slide 5 — Dataset (0:31)

> To test that, we use three public datasets.
> HaluEval QA gives 20,000 labeled answers from 10,000 pairs, one correct and one
> hallucinated.
> RAGTruth and FaithBench stay held out, for transfer and a stress test.
> The target is binary, and we split by question 70, 15, 15, so no group crosses a
> split.

**Hand-over:** "I will now hand over to my teammate, who will explain the conventional method and our proposed model."

---

## Member 2 — slides 6 to 10 (2:16)

### Slide 6 — Conventional method and baselines (0:24)

> Before the proposed method, the conventional setup.
> We compare four standard detectors: an overlap heuristic, logistic regression,
> random forest, and tuned XGBoost.
> XGBoost is our reference, and three limits motivate the changes: a saturated
> benchmark, length shortcuts, and unreliable scores under shift.

---

### Slide 7 — Proposed method (0:28)

> The proposed method is evidence-consistent XGBoost.
> It keeps the standard learner and adds three cumulative changes.
> M1 log-scales the length features and adds monotone constraints.
> M2 adds eight claim-level NLI aggregates.
> M3 adds RAGTruth non-QA rows with a source indicator, and M3 is the deployed model.

---

### Slide 8 — Flow diagram (0:37)

> Here is the full architecture, start to end.
> Question, evidence, and answer become 35 features.
> Those feed EC-XGB, trained with grouped cross-validation and monotone constraints.
> Platt calibration on RAGTruth QA produces the deployed probability.
> The output is the risk score, the claim verdicts, and the SHAP explanation.

---

### Slide 9 — Experimental setup (0:24)

> The model trains on a Windows 11 machine with an RTX 3060 GPU, under a strict protocol.
> We use a grouped 70, 15, 15 split, five-fold cross-validation, and three seeds.
> The calibrator is fitted on validation only, never on test.
> A strict mode caps false positives at five percent.

---

### Slide 10 — Results: in-domain (0:23)

> In-domain, standard XGBoost is the best learned model, with F1 at 0.985.
> Against random forest, McNemar gives p equals 0.044, so the gap is real.
> The heuristic is far behind, so the task is not trivial.

**Hand-over:** "I will now hand over to my teammate, who will present the shift results and the final system."

---

## Member 3 — slides 11 to 16 (2:08)

### Slide 11 — Results: shift and calibration (0:28)

> Now the main result.
> In-domain, EC-XGB matches standard XGBoost, and McNemar says it is not significant.
> On RAGTruth, the standard model flags 99.9 percent of answers.
> EC-XGB flags 61.3 percent, and AUROC rises from 0.475 to 0.582.
> The display error drops from 0.73 to 0.13.

---

### Slide 12 — Applications (0:35)

> Where does this fit?
> Chat assistants can check every answer as it streams.
> Document question answering can verify answers against an indexed corpus or the web.
> Content review can flag unsupported claims in drafts and summaries.
> Every result comes with a calibrated score, per-claim verdicts, and SHAP reasons.
> Local checks cost about one thousandth of a dollar per thousand answers.

---

### Slide 13 — UI: Chat (0:20)

> Here is how a user meets it.
> In chat, the answer streams and the risk card appears on its own.
> Marker one is the risk card with the verdict, and marker two is the grounding line that
> names the evidence used.

---

### Slide 14 — UI: Analyze (0:17)

> Analyze adds full input control.
> Marker one is the SHAP chart, marker two the full feature table, and marker three the
> model comparison.

---

### Slide 15 — Conclusion (0:23)

> Our detector needs no model weights, runs in 62 milliseconds, and gives a reason per
> score.
> In-domain F1 is 0.9855, with ECE 0.007.
> On shifted data the flag rate falls from 99.9 to 61.3 percent.

---

### Slide 16 — Thank you (0:05)

> Thank you. We are happy to take your questions.

---

## Q&A preparation (only if the judges ask)

**Member 1 — problem, gaps, and data**

- The models and features are English only.
- Around 85 percent of HaluEval is generated by sampling and filtering, so the
  benchmark is synthetic by construction.
- The two answer variants of a question stay in one split, so no group crosses a
  partition.

**Member 2 — method and setup**

- Calibrators are fitted on the validation split only, never on the test split.
- The feature vector has 35 inputs: 26 base, 8 claim aggregates, and 1 source
  indicator.
- The strict mode selects the validation threshold for a five percent
  false-positive budget, and test recall stays above 99 percent.

**Member 3 — results, limits, and future work**

- On held-out RAGTruth QA, the gain is mainly in calibration, not in F1.
- On FaithBench the model becomes conservative: it trades F1 for a much lower flag rate.
- The calibrated score is style-sensitive, which is why the card leads with the
  claim verdicts.
- Future work: multicalibration under stronger shift, per-domain thresholds, and
  neural baselines beside EC-XGB.

---

## Number cheat sheet

| Item | Value |
|------|-------|
| In-domain F1 (EC-XGB) | 0.9855 |
| In-domain AUROC | 0.9984 |
| In-domain ECE | 0.0074 |
| McNemar, XGB vs random forest | p = 0.044 |
| RAGTruth flagged, standard | 99.9% |
| RAGTruth flagged, EC-XGB | 61.3% |
| RAGTruth AUROC, standard to EC-XGB | 0.475 to 0.582 |
| RAGTruth ECE, standard to EC-XGB | 0.635 to 0.278 |
| FaithBench flagged, standard to EC-XGB | 99.5% to 7.3% |
| Display ECE, raw to Platt | 0.73 to 0.13 |
| Strict-mode test FPR | 4.2% - 5.1% |
| Strict-mode recall | 99.0% - 99.5% |
| Median analysis time | 62 ms |
| Judge F1 (GPT 5.6 Luna) | 0.84 |
| XGBoost F1, same subset | 0.985 |
| Judge cost per 1,000 | about $0.105 |
| Features | 35 (26 base + 8 claim + 1 source) |

---

## Delivery tips

- Pause for one full second after every number.
- Point at the table or chart you are naming.
- If you run long, drop the last sentence of slides 5, 7, and 9. Never shorten the
  shift result on slide 11.
- Look at the camera on slide 1, and at the judges on slides 11 and 16.
- Keep the hand-over lines short. The next member should start within two seconds.
