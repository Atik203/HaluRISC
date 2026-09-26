# HaluRISC — Video Presentation Speech (9 minutes)

Total time: **9:00**. Slides: **6:30**. Live demo: **2:30**.

This is the full speaking script. Slide numbers refer to the `/slide` deck (17 slides:
title, 15 content slides, thank you). The deck has no speaker notes on purpose, so this
file carries everything.

The script is written for a slow, careful speaker at **1.73 words per second** (about 104
words per minute). That rate is measured from the first recording pass, so the timings
below should hold without rushing.

Every sentence is written to connect to the one before it, and each slide ends on a line
that sets up the next slide. Read it as continuous speech, not as separate bullet points.

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
| 1 | Title | 0:21 | 0:21 |
| 2 | Problem | 0:51 | 1:12 |
| 3 | Motivation and gaps | 0:29 | 1:41 |
| 4 | Datasets: sources, target, splits | 0:30 | 2:11 |
| 5 | External corpora and features | 0:18 | 2:29 |
| 6 | Preprocessing and the pipeline | 0:27 | 2:56 |
| 7 | Feature engineering | 0:17 | 3:13 |
| 8 | Models and EC-XGB | 0:22 | 3:35 |
| 9 | Training protocol | 0:23 | 3:58 |
| 10 | Baseline comparison | 0:21 | 4:19 |
| 11 | EC-XGB results | 0:26 | 4:45 |
| 12 | LLM as judge | 0:21 | 5:06 |
| 13 | Calibration, trust, speed | 0:19 | 5:25 |
| 14 | Interface: Chat and Analyze | 0:20 | 5:45 |
| 15 | Interface: explanations, comparison | 0:17 | 6:02 |
| 16 | Conclusion | 0:20 | 6:22 |
| 17 | Thank you | 0:05 | 6:27 |
| — | **Live demo** | **2:30** | **8:57** |

The per-slide times are word counts divided by the measured pace of 1.73 words per second,
so they already include your natural pauses.

The demo sits after slide 15. It is shown live, so there is no walkthrough slide. After the
demo, press the right arrow twice to reach the conclusion.

---

## 3. Slide script

### Slide 1 — Title (0:21)

> Hi everyone. Our team name is Phantom Devs, and the project is HaluRISC.
> The full title is: Evidence-Consistent XGBoost for Calibrated Hallucination Risk
> Estimation in Black-Box Language Model Answers.
> I will finish with a short demonstration.

---

### Slide 2 — Problem (0:51)

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

### Slide 3 — Motivation and gaps (0:29)

> Existing work leaves three gaps open.
> First, calibration. Most lightweight detectors never report how reliable their score is.
> Second, explanations. Attribution plots are shown, but their stability is never tested.
> Third, domain shift. A detector that works on one dataset often fails on another.
> Our study covers all three together.

---

### Slide 4 — Datasets (0:30)

> To test that, we use three public datasets.
> HaluEval QA gives 20,000 labeled answers from 10,000 pairs, one correct and one
> hallucinated.
> RAGTruth and FaithBench stay held out, for transfer and a stress test.
> The target is binary, and we split by question 70, 15, 15, so no group crosses a
> split.

---

### Slide 5 — External corpora and features (0:18)

> The external data never enters training, so those scores are zero-shot.
> The feature vector has 35 features: 26 base, 8 claim, 1 source.
> Claim features connect the model to the verdicts.

---

### Slide 6 — Preprocessing and the pipeline (0:27)

> We clean the data and map every dataset into one schema.
> A group key keeps both answers together, so no group crosses a split.
> Then one pass extracts 35 features, scores them, and calibrates the score.
> The output is a risk score, verdicts, and SHAP explanations.

---

### Slide 7 — Feature engineering (0:17)

> Now the features.
> The 26 base features cover seven groups.
> The 8 claim features check each clause of the answer against the evidence.
> Verdicts are support-first, with a relevance gate.

---

### Slide 8 — Models and EC-XGB (0:22)

> With those features, we compared four baselines: a heuristic, logistic regression, random
> forest, and XGBoost.
> Then EC-XGB added monotone rules, the claim features, and RAGTruth rows with a source flag.
> That last version is the model we deploy.

---

### Slide 9 — Training protocol (0:23)

> That model is trained under a strict protocol.
> We use a grouped 70, 15, 15 split, five-fold cross-validation, and three seeds.
> The calibrator is fitted on validation only, never on test.
> A strict mode caps false positives at five percent.

---

### Slide 10 — Baseline comparison (0:21)

> In-domain, standard XGBoost is the best learned model, with F1 at 0.985.
> Against random forest, McNemar gives p equals 0.044, so the gap is real.
> The heuristic is far behind, so the task is not trivial.

---

### Slide 11 — EC-XGB results (0:26)

> Now the main result.
> In-domain, EC-XGB matches standard XGBoost. McNemar says it is not significant.
> On RAGTruth, the standard model flags 99.9 percent of answers, saying yes to everything.
> EC-XGB flags 61.3 percent, and AUROC rises from 0.497 to 0.582.
> So the value is robustness.

---

### Slide 12 — LLM as judge (0:21)

> We also compared against an LLM judge.
> GPT 5.6 Luna reaches F1 0.84 on 200 answers.
> XGBoost reaches 0.985 on the same subset, at about 100 times lower cost.
> So the judge is not the deployment.

---

### Slide 13 — Calibration, trust, speed (0:19)

> Three things make this usable.
> First, calibration. On RAGTruth answers, ECE falls from 0.73 to 0.13 after Platt
> scaling.
> Second, explanations are stable under edits.
> Third, one analysis takes about 62 milliseconds.

---

### Slide 14 — Interface: Chat and Analyze (0:20)

> Here is how a user meets it.
> In chat, the answer streams and the risk card appears on its own with the verdict and
> score.
> Analyze adds full input control and the band thresholds.

---

### Slide 15 — Interface: explanations and comparison (0:17)

> The score is never a black box.
> The SHAP chart explains the raw score, and the table lists all 35 features.
> The comparison card scores every model side by side.

---

## 4. Live demo script (2:30)

The demo is shown live. The **chat page is the core**, because it shows the everyday
workflow: the answer streams, and the risk card appears by itself.

Prepare both pages before recording, with the examples already loaded. Most of the demo is
action on screen, so the narration is short. Let each panel land before you speak again.

Speech in the demo is about 1:53. The other 37 seconds are the streaming wait, the panel
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

> To make this concrete, I will show it working.
> There are two pages: chat for everyday use, and Analyze for a close look at one answer.

**[0:10-0:40] Chat · prompt and stream**

> I paste a question, the evidence, and an answer, and the answer streams back.
> Watch what appears on its own below it: a risk card, marked likely hallucinated, with an
> evidence score of 65 percent.

**[0:40-1:00] Chat · claim verdicts**

> Twelve claims were checked, and each one is marked supported, contradicted, or
> unsupported.
> The headline follows the claim evidence.

**[1:00-1:20] Chat · why this score**

> I expand "Why this score".
> Here are the SHAP bars, and the full table of all 35 features.

**[1:20-1:45] Analyze · hallucinated**

> Now Analyze. The evidence says 5 days and the answer says 10, so I run it.
> The gauge shows medium risk, 35 percent, and the number claim is contradicted, with the
> evidence sentence shown beside it.

**[1:45-2:00] Analyze · grounded**

> Now the grounded example. The answer matches the evidence, so the risk is low, 8 percent,
> and the verdict is supported.

**[2:00-2:20] Model comparison**

> Finally, the model comparison, where EC-XGB, XGBoost, random forest, and logistic
> regression are all scored side by side.
> On this pair, EC-XGB separates them most clearly.

**[2:20-2:30] Hand back**

> That is the full workflow, from a chat answer to a clear reason. Thank you.

---

## 5. Slide 16 — Conclusion (0:20)

> Our detector needs no model weights, runs in 62 milliseconds, and gives a reason per
> score.
> In-domain F1 is 0.986, with ECE 0.007.
> On shifted data the flag rate falls from 99.9 to 61.3 percent.

---

### Slide 17 — Thank you (0:05)

> Thank you. We are happy to take your questions.

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
- If you fall behind, drop the last sentence of slides 5, 6, 8, and 12. Never shorten the
  demo.
- In the demo, let the card and the gauge appear before you speak. Silence while a panel
  loads reads as confidence.
- Show the evidence quote, not only the score. Judges remember the reason.
- Look at the camera on slides 1 and 17.
