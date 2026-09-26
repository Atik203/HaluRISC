# HaluRISC — Video Presentation Speech (8 minutes)

Total time: **8:00**. Slides: **6:00**. Live demo: **2:00**.

This is the full speaking script. Slide numbers refer to the `/slide` deck (20 slides).
The deck has no speaker notes on purpose, so this file carries everything.

---

## 1. Warm-up analysis: how to catch the judges in the first 30 seconds

Judges watch many videos in one sitting. Most open with "Hello, my name is... and my
project is...". That is forgettable. Three rules for this presentation:

1. **Open with a wrong answer, not with your name.** Show the model saying the melt
   season grew "10 days per decade" while the evidence says "5". Everyone understands
   this in two seconds. Naming the team can come right after.
2. **Give one hard number early.** "Today's detector flags 99.9% of real answers as
   risky. Ours flags 61.3%." A number the judges can repeat back is a number they
   remember.
3. **Promise the deliverable in one line, then prove it.** "No model weights, 62
   milliseconds, calibrated, and it explains itself." The rest of the talk is proof.

Two more things that build trust with an expert audience:

- **Be honest about limits.** Say the held-out F1 is low, but calibration improves. An
  honest limitation is stronger than a clean story.
- **Close on the shift result.** It is your most defensible claim, because in-domain
  differences are not statistically significant.

Keep every sentence short. Pause after each number. If you run late, drop slide 6 and
slide 9 details, never the demo.

---

## 2. Timing plan

| # | Slide | Time | Running |
|---|-------|------|---------|
| 1 | Title | 0:15 | 0:15 |
| 2 | Problem | 0:24 | 0:39 |
| 3 | Motivation and gaps | 0:18 | 0:57 |
| 4 | Dataset sources | 0:16 | 1:13 |
| 5 | Dataset in-domain | 0:16 | 1:29 |
| 6 | Dataset external and features | 0:16 | 1:45 |
| 7 | Data preprocessing | 0:18 | 2:03 |
| 8 | End-to-end pipeline | 0:18 | 2:21 |
| 9 | Feature engineering | 0:18 | 2:39 |
| 10 | Models and EC-XGB | 0:18 | 2:57 |
| 11 | Training protocol | 0:16 | 3:13 |
| 12 | Baseline comparison | 0:21 | 3:34 |
| 13 | EC-XGB results | 0:24 | 3:58 |
| 14 | LLM as judge | 0:18 | 4:16 |
| 15 | Calibration, trust, speed | 0:20 | 4:36 |
| 16 | Chat and Analyze UI | 0:14 | 4:50 |
| 17 | Explanations and usability | 0:14 | 5:04 |
| 18 | Demo hand-off | 0:09 | 5:13 |
| 19 | Conclusion | 0:20 | 5:33 |
| 20 | Thank you | 0:07 | 5:40 |
| — | **Live demo** | **2:00** | **7:40** |

Buffer: 20 seconds. Use it to breathe, not to add content.

---

## 3. Slide script

### Slide 1 — Title (0:15)

> A language model can say: "The Arctic melt season grew by 10 days per decade."
> The evidence says 5. The sentence is fluent. It is wrong.
> This is HaluRISC. We detect that kind of answer, and we explain why it is risky.
> Team Phantom Devs. Section E.

---

### Slide 2 — Problem (0:24)

> The example is the whole problem. The answer sounds correct, so a reader trusts it.
> We only have the question, the evidence, and the answer. We cannot see the model
> weights.
> Other detectors need model access, or a large language model, or a GPU.
> We wanted four things at once. Accurate. Calibrated. Explainable. Fast. With no
> access to the model.

---

### Slide 3 — Motivation and gaps (0:18)

> Three gaps in current work.
> First, calibration. Most lightweight detectors never report it.
> Second, explanation quality. Attributions are drawn, but their stability is not
> measured.
> Third, domain shift. Detectors lose accuracy on new data, and this is rarely tested.
> We test all three in one study.

---

### Slide 4 — Dataset sources (0:16)

> We use three public datasets.
> HaluEval QA for training. It has 20,000 rows in 10,000 pairs.
> RAGTruth for transfer and calibration. 17,790 rows.
> FaithBench for a summarization stress test. 750 rows.
> The first two are MIT licensed. FaithBench is non-commercial, so we ship a download
> script, not the data.

---

### Slide 5 — Dataset in-domain (0:16)

> HaluEval is the core. Each question has a correct answer and a hallucinated answer.
> This pairs the two, so the model learns evidence consistency.
> We split by question, 70, 15, 15. Both answers stay in one split. No leakage.
> The labels are balanced, 50 / 50 in every split.

---

### Slide 6 — Dataset external and features (0:16)

> The external data never enters training. RAGTruth QA is fully held out, so its score
> is a real zero-shot result.
> On the right is the feature vector. 35 features in total. 26 base features, 8
> claim-level features, and 1 source flag.
> Zero-shot transfer is weakest on QA and best on data-to-text.

---

### Slide 7 — Data preprocessing (0:18)

> Preprocessing is simple and strict.
> We clean the text, drop empty rows, and map every dataset into one schema: question,
> context, answer, label, and a group key.
> The group key is the important part. It keeps both answers of a question together.
> The result is zero groups crossing splits. We verify this and freeze the split
> indices, so every seed uses the same data.

---

### Slide 8 — End-to-end pipeline (0:18)

> Here is the full system.
> Input is a question, evidence, and an answer.
> The system extracts 35 features.
> Then EC-XGB scores the answer, and a display calibrator adjusts the score.
> The output is three things: a calibrated risk score, per-claim verdicts, and SHAP
> explanations.
> One pass. No model weights needed.

---

### Slide 9 — Feature engineering (0:18)

> The 26 base features cover seven groups: length, lexical overlap, entities, NLI,
> numbers, hedging, and semantic drift.
> The 8 claim features are new. We split the answer into small clauses, then check each
> clause against the evidence.
> For example, "the change is dominated by a later freezeup, and the region is at its
> warmest" becomes two claims, and each one gets a verdict.
> The verdicts are support-first, with a relevance gate.

---

### Slide 10 — Models and EC-XGB (0:18)

> We compare four baselines: an overlap heuristic, logistic regression, random forest,
> and standard XGBoost.
> Then EC-XGB adds three changes.
> M1 adds length features and monotone rules. Risk cannot go down when contradiction
> goes up.
> M2 adds the 8 claim features.
> M3 adds RAGTruth non-QA rows and a source flag.
> That gives 35 features.

---

### Slide 11 — Training protocol (0:16)

> The protocol is strict, because the results must be believable.
> Grouped 70, 15, 15 split. Five-fold cross-validation for tuning.
> Three seeds: 42, 123, and 456. Every number is a mean over the three.
> The calibrator is fitted on the validation split only, never on test.
> We also report a strict mode that caps false positives at 5%.

---

### Slide 12 — Baseline comparison (0:21)

> On the in-domain test set, standard XGBoost wins among the learned models.
> F1 is 0.985. AUROC is 0.998. PR-AUC is 0.998.
> The bootstrap confidence interval for F1 is 0.981 to 0.990.
> Against the best baseline, random forest, McNemar's test gives p equals 0.044. So the
> gap is small but real.
> The heuristic is far behind. That shows the task is not trivial.

---

### Slide 13 — EC-XGB results (0:24)

> This is the main result, and it is a shift result, not an in-domain result.
> In-domain, EC-XGB is almost the same as standard XGBoost. McNemar says the difference
> is not significant.
> On RAGTruth, the standard model flags 99.9% of answers as risky. That is useless,
> because it says yes to everything.
> EC-XGB flags 61.3%. And AUROC goes up, from 0.497 to 0.582.
> So we trade a little recall for much better precision, and the flag rate becomes
> usable.
> The value is robustness, not extra benchmark points.

---

### Slide 14 — LLM as judge (0:18)

> We also tested a large language model as a judge. GPT 5.6 Luna, on 200 answers.
> Its F1 is 0.84. Recall is only 74%. So it misses about one in four hallucinations.
> XGBoost on the same subset reaches F1 0.985.
> The judge costs about 100 times more and takes about 20 times longer.
> So the judge is a baseline we audited, not the model we deploy.

---

### Slide 15 — Calibration, trust, speed (0:20)

> Three points here.
> Calibration. On natural RAGTruth answers, the raw score has ECE 0.73. After Platt
> scaling, it is 0.13. So the number on screen means something.
> Explanation. The SHAP ranking is stable, and edits to entities move the score while
> irrelevant inserts do not.
> Speed. One analysis takes about 62 milliseconds at the median.
> That is why this can run inside a chat interface.

---

### Slide 16 — Chat and Analyze UI (0:14)

> The interface comes in two modes.
> Chat scores the answer automatically while it streams. The card leads with the
> verdict, then the evidence score, then the main SHAP contributors.
> Analyze gives the same pipeline with full input control and the band thresholds.
> A score of 35% is labelled medium risk, with no lookup table.

---

### Slide 17 — Explanations and usability (0:14)

> This view shows the explanation layer.
> The SHAP chart explains the raw score. The table lists all 35 features and their
> values, so nothing is hidden.
> The comparison card scores every model side by side, including EC-XGB.
> The interface is consistent across chat, analyze, and the dashboard.

---

### Slide 18 — Demo hand-off (0:09)

> Now I will show the working system.
> I will paste one hallucinated answer and one grounded answer, and I will compare the
> models.

---

## 4. Live demo script (2:00)

Prepare before recording: open the Analyze page, load the hallucinated-number example,
and make sure the backend is running. Do not type long text on camera.

**[0:00-0:15] Set the scene**

> This is the Analyze page. On the left I paste a question, the evidence, and an answer
> to check.

**[0:15-0:50] Hallucinated example**

> The question is: how many days per decade did the melt season lengthen?
> The evidence says 5 days. The answer says 10.
> I click "Run risk analysis".
> The gauge shows medium risk, 35%. The band markers are printed below it.
> Below the gauge, the claims are listed. The number claim is contradicted, and it
> shows the evidence sentence. That is the point. The user sees why.

**[0:50-1:10] Explanation**

> I open the SHAP panel.
> The top features explain this score. The contradiction feature pushes the risk up.
> The lexical overlap with the evidence pushes it down.
> So the explanation matches the story we just read.

**[1:10-1:35] Grounded example**

> Now I load the grounded example. The answer matches the evidence.
> The risk is low, 8%.
> No contradicted claims. The verdict says supported.

**[1:35-1:55] Model comparison**

> Finally, I open the model comparison.
> EC-XGB, standard XGBoost, random forest, and logistic regression are all scored here.
> On this pair, EC-XGB separates the two answers the clearest.
> The whole comparison takes about 12 milliseconds.

**[1:55-2:00] Hand back**

> That is the complete workflow, from input to explanation.

---

## 5. Slide 19 — Conclusion (0:20)

> To conclude.
> We built a detector that needs no model weights, runs in milliseconds, and explains
> every score.
> In-domain F1 is 0.986, with ECE 0.007.
> On shifted data, the flag rate drops from 99.9% to 61.3%, and AUROC improves.
> Every number ships with tests, ablations, and a frozen manifest.

---

### Slide 20 — Thank you (0:07)

> Thank you. We are happy to take your questions.

---

## 6. Limitations and future work (say only if the judges ask)

Keep these ready. Do not volunteer all of them in the talk.

- The models and features are English only.
- On held-out RAGTruth QA, the gain is mainly in calibration, not F1.
- On FaithBench the model becomes conservative. It trades F1 for a much lower flag rate.
- Some multi-source gains come from task types seen during training.
- Future work: multicalibration under stronger shift, per-domain thresholds, and
  neural baselines beside EC-XGB.

---

## 7. Number cheat sheet

| Item | Value |
|------|-------|
| In-domain F1 (EC-XGB) | 0.9855 |
| In-domain AUROC | 0.9984 |
| In-domain ECE | 0.0074 |
| F1 bootstrap 95% CI | 0.9812 - 0.9895 |
| McNemar, XGB vs random forest | p = 0.044 |
| RAGTruth flagged, standard | 99.9% |
| RAGTruth flagged, EC-XGB | 61.3% |
| RAGTruth AUROC, standard to EC-XGB | 0.497 to 0.582 |
| Display ECE, raw to Platt | 0.73 to 0.13 |
| Strict-mode test FPR | 4.2% - 5.1% |
| Strict-mode recall | 99.0% - 99.5% |
| Median analysis time | 62 ms |
| Judge F1 (GPT 5.6 Luna) | 0.84 |
| XGBoost F1, same subset | 0.985 |
| Judge cost per 1,000 | about $0.105 |
| Features | 35 (26 base + 8 claim + 1 source) |

---

## 8. Delivery tips

- Speak at your own pace. The plan already assumes a slow speaker, about 2.2 words per
  second.
- Pause for one full second after every number. Numbers need air.
- Point at the screen with the cursor when you mention a table or a chart.
- If you fall behind, shorten slides 6, 9, and 10. Never cut the demo.
- Keep your hands still and look at the camera on slides 1 and 20.
- Practise the demo twice with the backend already running. The API takes about 30
  seconds to load its models, so start it before you record.
