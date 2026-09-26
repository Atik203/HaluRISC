# HaluRISC — Video Presentation Speech (8 minutes)

Total time: **8:00**. Slides: **5:30**. Live demo: **2:30**.

This is the full speaking script. Slide numbers refer to the `/slide` deck (17 slides:
title, 15 content slides, thank you). The deck has no speaker notes on purpose, so this
file carries everything.

The script is written for a slow, careful speaker at **1.73 words per second** (about 104
words per minute). That rate is measured from the first recording pass, so the timings
below should hold without rushing. Every number already appears on its slide, so you never
read a table line by line.

If you run long, use the fallback list in the delivery tips at the end. Each slide can lose
its last sentence without breaking the story.

---

## 1. Before you record

- **Open the chat page with the example already loaded.** Do not type on camera.
- **Open the Analyze page with the hallucinated-number example loaded.** The grounded
  example is one click away.
- **Start the backend early.** It takes about 30 seconds to load its models. Use
  `scripts\serve_api.cmd`, which keeps it alive automatically.
- **Open the deck at slide 1** in a second tab, and switch to it when the slides start.

**Good opening habits.** Open formally, then land the hook on slide 2. Give one hard
number early: "Today's detector flags 99.9% of answers as risky. Ours flags 61.3%." Pause
for one full second after every number. Be honest about limits, and close on the shift
result, because in-domain differences are not statistically significant.

---

## 2. Timing plan

| # | Slide | Time | Running |
|---|-------|------|---------|
| 1 | Title | 0:19 | 0:19 |
| 2 | Problem | 0:26 | 0:45 |
| 3 | Motivation and gaps | 0:15 | 1:00 |
| 4 | Datasets: sources, target, splits | 0:24 | 1:24 |
| 5 | External corpora and features | 0:19 | 1:43 |
| 6 | Preprocessing and the pipeline | 0:24 | 2:07 |
| 7 | Feature engineering | 0:21 | 2:28 |
| 8 | Models and EC-XGB | 0:20 | 2:48 |
| 9 | Training protocol | 0:17 | 3:05 |
| 10 | Baseline comparison | 0:20 | 3:25 |
| 11 | EC-XGB results | 0:27 | 3:52 |
| 12 | LLM as judge | 0:21 | 4:13 |
| 13 | Calibration, trust, speed | 0:15 | 4:28 |
| 14 | Interface: Chat and Analyze | 0:19 | 4:47 |
| 15 | Interface: explanations, comparison | 0:17 | 5:04 |
| 16 | Conclusion | 0:20 | 5:24 |
| 17 | Thank you | 0:05 | 5:29 |
| — | **Live demo** | **2:30** | **7:59** |

The per-slide times are word counts divided by the measured pace of 1.73 words per second,
so they already include your natural pauses. The deck lands at 5:29, and the whole video at
7:59, with no rushing.

The demo sits after slide 15. It is shown live, so there is no walkthrough slide. After the
demo, press the right arrow twice to reach the conclusion.

---

## 3. Slide script

### Slide 1 — Title (0:19)

> Hi everyone. Our team name is Phantom Devs, and our project is HaluRISC.
> The full title is: Evidence-Consistent XGBoost for Calibrated Hallucination Risk
> Estimation in Black-Box Language Model Answers.
> Then a short demo.

---

### Slide 2 — Problem (0:26)

> One example first. A model says the Arctic melt season grew 10 days per decade. The
> evidence says 5. Fluent, and wrong.
> We see only the question, the evidence, and the answer, not the weights.
> We wanted accurate, calibrated, explainable, fast, without the model weights.

---

### Slide 3 — Motivation and gaps (0:15)

> Current work leaves three gaps.
> Calibration is rarely reported. Explanation stability is rarely measured. Accuracy on
> new data is rarely tested.
> Our study covers all three.

---

### Slide 4 — Datasets (0:24)

> HaluEval QA for training, with 20,000 labeled answers in 10,000 pairs. RAGTruth for
> held-out transfer. FaithBench as a stress test.
> The target is binary: hallucinated or grounded.
> We split by question, 70, 15, 15, so both answers stay in one split. Leakage-free.

---

### Slide 5 — External corpora and features (0:20)

> The external data never enters training, so those scores are zero-shot.
> The feature vector has 35 features: 26 base, 8 claim-level, and 1 source flag.
> Claim features connect the model to the verdicts.

---

### Slide 6 — Preprocessing and the pipeline (0:24)

> We map every dataset into one schema, and a group key keeps both answers together, so
> no group crosses a split.
> One pass does everything: 35 features, EC-XGB, a calibrator.
> The output is a risk score, per-claim verdicts, and SHAP explanations.

---

### Slide 7 — Feature engineering (0:21)

> The 26 base features cover seven groups, from length and lexical overlap to NLI and
> semantic drift.
> The 8 claim features check each clause of the answer against the evidence.
> Verdicts are support-first, with a relevance gate.

---

### Slide 8 — Models and EC-XGB (0:20)

> We compare four baselines: an overlap heuristic, logistic regression, random forest, and
> standard XGBoost.
> EC-XGB adds three changes: M1 length features and monotone rules, M2 the claim features,
> M3 RAGTruth rows and a source flag.

---

### Slide 9 — Training protocol (0:17)

> Grouped 70, 15, 15 split, five-fold cross-validation, and three seeds: 42, 123, and 456.
> The calibrator is fitted on validation only.
> A strict mode caps false positives at five percent.

---

### Slide 10 — Baseline comparison (0:20)

> In-domain, standard XGBoost is best among the learned models, with F1 at 0.985.
> Against random forest, McNemar gives p equals 0.044. Small, but real.
> The heuristic is far behind, so the task is not trivial.

---

### Slide 11 — EC-XGB results (0:27)

> In-domain, EC-XGB matches standard XGBoost. McNemar says the difference is not
> significant.
> On RAGTruth, the standard model flags 99.9 percent of answers, which says yes to
> everything.
> EC-XGB flags 61.3 percent, and AUROC rises from 0.497 to 0.582.
> The value is robustness, not extra benchmark points.

---

### Slide 12 — LLM as judge (0:21)

> We tested GPT 5.6 Luna as a judge on 200 answers. Its F1 is 0.84, while XGBoost reaches
> 0.985 on the same subset at about 100 times lower cost.
> So it is a baseline, not the deployment.

---

### Slide 13 — Calibration, trust, speed (0:15)

> Calibration: on natural RAGTruth answers, ECE falls from 0.73 to 0.13 after Platt
> scaling.
> Explanations are stable under edits.
> One analysis takes about 62 milliseconds.

---

### Slide 14 — Interface: Chat and Analyze (0:19)

> Chat scores the answer while it streams, and the risk card appears on its own with the
> verdict, the score, and the SHAP contributors.
> Analyze adds full input control and the band thresholds.

---

### Slide 15 — Interface: explanations and comparison (0:17)

> The score is never a black box.
> The SHAP chart explains the raw score, and the table lists all 35 features.
> The comparison card scores every model side by side.

---

## 5. Slide 16 — Conclusion (0:20)

> A detector with no model weights, 62 milliseconds, and an explanation for every score.
> In-domain F1 is 0.986, and ECE is 0.007.
> On shifted data the flag rate falls from 99.9 to 61.3 percent.

---

### Slide 17 — Thank you (0:05)

> Thank you. We are happy to take your questions.

---

## 4. Live demo script (2:30)

The demo is shown live. The **chat page is the core**, because it shows the everyday
workflow: the answer streams, and the risk card appears by itself.

Prepare both pages before recording, with the examples already loaded. Most of the demo is
action on screen, so the narration is short. Let each panel land before you speak again.

Speech in the demo is about 1:45. The other 45 seconds are the streaming wait, the panel
expansions, and the page switches.

| Beat | Time | Running |
|------|------|---------|
| 1 Set the scene | 0:10 | 0:10 |
| 2 Chat · prompt and stream | 0:30 | 0:40 |
| 3 Chat · claim verdicts | 0:20 | 1:00 |
| 4 Chat · why this score | 0:20 | 1:20 |
| 5 Analyze · hallucinated | 0:25 | 1:45 |
| 6 Analyze · grounded | 0:15 | 2:00 |
| 7 Model comparison | 0:20 | 2:20 |
| 8 Hand back | 0:10 | 2:30 |

**[0:00-0:10] Set the scene**

> Two pages. Chat for everyday use, Analyze for a close look. I will start with chat.

**[0:10-0:40] Chat · prompt and stream**

> I paste a question, the evidence, and an answer. The answer streams back.
> Watch the risk card appear on its own below it. Likely hallucinated, evidence score 65
> percent.

**[0:40-1:00] Chat · claim verdicts**

> Twelve claims were checked. Each one is marked supported, contradicted, or unsupported.
> The headline follows the claim evidence.

**[1:00-1:20] Chat · why this score**

> I expand "Why this score".
> Here are the SHAP bars, and the full table of all 35 features.

**[1:20-1:45] Analyze · hallucinated**

> Now Analyze. The evidence says 5 days, the answer says 10. I run it.
> The gauge shows medium risk, 35 percent. The number claim is contradicted, and the
> evidence sentence is shown beside it.

**[1:45-2:00] Analyze · grounded**

> Now the grounded example. The answer matches the evidence, so the risk is low, 8
> percent, and the verdict is supported.

**[2:00-2:20] Model comparison**

> Finally, the model comparison. EC-XGB, XGBoost, random forest, and logistic regression
> are all scored here.
> On this pair, EC-XGB separates them most clearly.

**[2:20-2:30] Hand back**

> That is the full workflow, from a chat answer to a clear reason. Thank you.

---

## 6. Hand-over lines (only if each member presents a part)

- Into the dataset section: "I will now hand over to my teammate, who will present the
  datasets and the features."
- Into the methodology section: "Thank you. I will now explain our method and the training
  protocol."
- Into the demo: "I will now demonstrate the working system."
- Into the conclusion: "Thank you. I will now summarise our findings."

---

## 7. Limitations and future work (say only if the judges ask)

- The models and features are English only.
- On held-out RAGTruth QA, the gain is mainly in calibration, not in F1.
- On FaithBench the model becomes conservative. It trades F1 for a much lower flag rate.
- Some multi-source gains come from task types seen during training.
- Future work: multicalibration under stronger shift, per-domain thresholds, and neural
  baselines beside EC-XGB.

---

## 8. Number cheat sheet

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

## 9. Delivery tips

- The rate is already set for a slow speaker. Do not add sentences on the day.
- Pause for one full second after every number.
- Point at the screen when you mention a table or a chart.
- If you fall behind, drop the last sentence of slides 5, 7, 8, and 12. Never shorten the
  demo.
- In the demo, let the card and the gauge appear before you speak. Silence while a panel
  loads reads as confidence.
- Show the evidence quote, not only the score. Judges remember the reason.
- Look at the camera on slides 1 and 17.
