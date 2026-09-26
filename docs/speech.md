# HaluRISC — Video Presentation Speech (8 minutes)

Total time: **8:00**. Slides: **5:30**. Live demo: **2:30**.

This is the full speaking script. Slide numbers refer to the `/slide` deck (17 slides:
title, 15 content slides, thank you). The deck has no speaker notes on purpose, so this
file carries everything.

The register is formal but the sentences are short. All contractions are spelled out,
because they are easier for a slow speaker to pronounce clearly.

---

## 1. Warm-up analysis: how to hold the judges

Judges watch many videos in one sitting. The opening must be formal and clear, and the
hook must land in the first half minute.

1. **Open formally first.** Greet the audience briefly, then give the team name, the
   project name, and the full title. This is expected and it builds credibility. Do not
   skip it. This is a recorded video, so do not mention a time of day, and do not thank
   anyone for attending.
2. **Deliver the hook right after, on slide 2.** Show the model saying the melt season
   grew "10 days per decade" while the evidence says "5". Everyone understands this in
   two seconds, and it makes the formal opening feel purposeful.
3. **Give one hard number early.** "Today's detector flags 99.9% of real answers as
   risky. Ours flags 61.3%." A number the judges can repeat back is a number they
   remember.
4. **State the deliverable in one line, then prove it.** "No model weights, 62
   milliseconds, calibrated, and it explains itself." The rest of the talk is proof.

Two more things that build trust with an expert audience:

- **Be honest about limits.** Say the held-out F1 is low, but calibration improves. An
  honest limitation is stronger than a perfect story.
- **Close on the shift result.** It is the most defensible claim, because in-domain
  differences are not statistically significant.

Keep every sentence short. Pause after each number. If you run late, shorten slides 5, 7,
and 8. Never shorten the demo.

---

## 2. Timing plan

| # | Slide | Time | Running |
|---|-------|------|---------|
| 1 | Title | 0:15 | 0:15 |
| 2 | Problem | 0:24 | 0:39 |
| 3 | Motivation and gaps | 0:15 | 0:54 |
| 4 | Datasets: sources, target, splits | 0:26 | 1:20 |
| 5 | External corpora and features | 0:20 | 1:40 |
| 6 | Preprocessing and the pipeline | 0:26 | 2:06 |
| 7 | Feature engineering | 0:20 | 2:26 |
| 8 | Models and EC-XGB | 0:20 | 2:46 |
| 9 | Training protocol | 0:16 | 3:02 |
| 10 | Baseline comparison | 0:22 | 3:24 |
| 11 | EC-XGB results | 0:26 | 3:50 |
| 12 | LLM as judge | 0:20 | 4:10 |
| 13 | Calibration, trust, speed | 0:22 | 4:32 |
| 14 | Interface: Chat and Analyze | 0:20 | 4:52 |
| 15 | Interface: explanations, comparison | 0:20 | 5:12 |
| 16 | Conclusion | 0:12 | 5:24 |
| 17 | Thank you | 0:06 | 5:30 |
| — | **Live demo** | **2:30** | **8:00** |

The demo sits between slide 15 and slide 16. It is shown live, so there is no walkthrough
slide in the deck. After the demo, press the right arrow twice to reach the conclusion.

---

## 3. Slide script

### Slide 1 — Title (0:15)

> Hi everyone. Our team name is Phantom Devs.
> Our project is called HaluRISC.
> The full title is: Evidence-Consistent XGBoost for Calibrated Hallucination Risk
> Estimation in Black-Box Language Model Answers.
> In this video, I will present the problem, our method, the results, and a short
> demonstration.

---

### Slide 2 — Problem (0:24)

> I would like to begin with one example.
> A language model can say: "The Arctic melt season grew by 10 days per decade."
> The evidence says 5. The sentence is fluent. It is wrong.
> This is the whole problem. The answer sounds correct, so a reader trusts it.
> We only have the question, the evidence, and the answer. We cannot see the model
> weights.
> Other detectors need model access, or a large language model, or a GPU.
> We wanted four things at once. Accurate. Calibrated. Explainable. Fast.

---

### Slide 3 — Motivation and gaps (0:15)

> Current work leaves three gaps.
> First, calibration. Most lightweight detectors never report it.
> Second, explanation quality. Attributions are drawn, but their stability is not
> measured.
> Third, domain shift. Detectors lose accuracy on new data, and this is rarely tested.
> Our study addresses all three gaps together.

---

### Slide 4 — Datasets (0:26)

> We use three public datasets.
> HaluEval QA is used for training. It has 20,000 rows in 10,000 pairs, and each question
> has one correct answer and one hallucinated answer.
> RAGTruth provides zero-shot transfer and calibration. It has 17,790 rows.
> FaithBench is a summarization stress test with 750 rows.
> The target is binary for each answer. One means hallucinated, and zero means grounded.
> The labels are balanced, 50 to 50.
> We split by question, in the ratio 70, 15, 15, so both answers stay in one split. The
> split is verified leakage-free.

---

### Slide 5 — External corpora and features (0:20)

> The external data never enters training. RAGTruth QA is fully held out, so its score is
> a true zero-shot result.
> On the right is the feature vector. There are 35 features in total: 26 base features, 8
> claim-level features, and 1 source flag.
> RAGTruth covers three tasks. Zero-shot transfer is weakest on question answering, and
> strongest on data-to-text.

---

### Slide 6 — Preprocessing and the pipeline (0:26)

> Preprocessing is simple and strict.
> We clean the text, remove empty rows, and map every dataset into one schema: question,
> context, answer, label, and a group key.
> The group key is the important part. It keeps both answers of a question together, so
> zero groups cross the splits. The split indices are frozen, so every seed uses the same
> data.
> After that, one pass does everything. The input goes in, 35 features are extracted,
> EC-XGB scores the answer, and a calibrator adjusts the score.
> The output is a calibrated risk score, per-claim verdicts, and SHAP explanations.

---

### Slide 7 — Feature engineering (0:20)

> The 26 base features cover seven groups: length, lexical overlap, entities, natural
> language inference, numbers, hedging, and semantic drift.
> The 8 claim features are new. We split the answer into small clauses, and we check each
> clause against the evidence.
> For example, "the change is dominated by a later freezeup, and the region is at its
> warmest" becomes two claims, and each claim receives a verdict.
> The verdicts are support-first, with a relevance gate.

---

### Slide 8 — Models and EC-XGB (0:20)

> We compare four baselines: an overlap heuristic, logistic regression, random forest, and
> standard XGBoost.
> EC-XGB then adds three changes.
> M1 adds length features and monotone rules. Risk cannot decrease when contradiction
> increases.
> M2 adds the 8 claim features.
> M3 adds RAGTruth non-QA rows and a source flag.
> M1, M2, and M3 are the ablation stages, and M3 is the model we deploy as EC-XGB. It has
> 35 features.

---

### Slide 9 — Training protocol (0:16)

> Our protocol is strict, because the results must be believable.
> We use a grouped 70, 15, 15 split, and five-fold cross-validation for tuning.
> We repeat every experiment with three seeds: 42, 123, and 456. Every number is the mean
> over the three seeds.
> The calibrator is fitted on the validation split only, never on the test split.
> We also report a strict mode that limits false positives to five percent.

---

### Slide 10 — Baseline comparison (0:22)

> On the in-domain test set, standard XGBoost performs best among the learned models.
> F1 is 0.985. AUROC is 0.998. PR-AUC is 0.998.
> The bootstrap confidence interval for F1 is 0.981 to 0.990.
> Against the best baseline, random forest, McNemar's test gives p equals 0.044. The gap
> is small, but it is real.
> The heuristic is far behind. This shows that the task is not trivial.

---

### Slide 11 — EC-XGB results (0:26)

> This is the main result, and it is a shift result, not an in-domain result.
> In-domain, EC-XGB is almost identical to standard XGBoost. McNemar's test says the
> difference is not significant.
> On RAGTruth, the standard model flags 99.9 percent of answers as risky. That is not
> useful, because it says yes to everything.
> EC-XGB flags 61.3 percent. At the same time, AUROC rises from 0.497 to 0.582.
> We accept a small loss in recall, and we gain much better precision. The flag rate
> becomes usable.
> The value of EC-XGB is robustness, not extra benchmark points.

---

### Slide 12 — LLM as judge (0:20)

> We also evaluated a large language model as a judge. We used GPT 5.6 Luna on 200
> answers.
> Its F1 is 0.84, and its recall is only 74 percent. It misses about one in four
> hallucinations.
> XGBoost, on the same subset, reaches F1 0.985.
> The judge costs about 100 times more per prediction, and it is about 20 times slower.
> For this reason, the judge is an audited baseline. It is not the model we deploy.

---

### Slide 13 — Calibration, trust, speed (0:22)

> There are three points on this slide.
> First, calibration. On natural RAGTruth answers, the raw score has an ECE of 0.73.
> After Platt scaling, it becomes 0.13. The displayed number now means something.
> Second, explanation quality. The SHAP ranking is stable, and edits to entities move the
> score, while irrelevant insertions do not.
> Third, speed. One analysis takes about 62 milliseconds at the median.
> This is what allows the model to run inside a chat interface.

---

### Slide 14 — Interface: Chat and Analyze (0:20)

> The interface has two modes.
> Chat scores the answer while it streams. The card shows the verdict first, then the
> evidence score, and then the SHAP contributors.
> Analyze gives the same pipeline with full control over the input. It shows the gauge and
> the band thresholds, so a score of 35 percent reads as medium risk at a glance.
> Every claim verdict also shows the evidence sentence it was checked against.

---

### Slide 15 — Interface: explanations and comparison (0:20)

> The score is never a black box.
> The SHAP chart explains the raw score. The table lists all 35 features and their values
> for this answer.
> The comparison card scores every model side by side, including EC-XGB, standard XGBoost,
> random forest, and logistic regression.
> The interface stays consistent across chat, analyze, and the dashboard.

---

## 4. Live demo script (2:30)

The demo is shown live from the application, not from a slide. Prepare before recording:
open the Analyze page, load the hallucinated-number example, and make sure the backend is
running. Do not type long text on camera.

The backend takes about 30 seconds to load its models. Start it before you record, or use
`scripts\serve_api.cmd`, which keeps it alive automatically.

**[0:00-0:20] Set the scene**

> This is the Analyze page. On the left, I enter a question, the evidence, and the answer I
> want to check.
> On the right, the result appears after each run.

**[0:20-1:00] Hallucinated example**

> The question is: how many days per decade did the melt season lengthen?
> The evidence says 5 days. The answer says 10.
> I click "Run risk analysis".
> The gauge shows medium risk, 35 percent. The band markers are printed below it.
> Below the gauge, the claims are listed. The number claim is contradicted, and the
> evidence sentence is shown next to it.
> This is the important part. The user can see why the answer is risky, not just that it
> is risky.

**[1:00-1:20] Explanation**

> I open the SHAP panel.
> The top features explain this score. The contradiction feature pushes the risk up.
> The lexical overlap with the evidence pushes the risk down.
> The explanation matches the reasoning we just read.

**[1:20-1:50] Grounded example**

> Now I load the grounded example. The answer matches the evidence.
> The risk is low, 8 percent.
> There are no contradicted claims. The verdict says "supported".
> So the same model gives a low score when the answer is faithful.

**[1:50-2:20] Model comparison**

> Finally, I open the model comparison.
> EC-XGB, standard XGBoost, random forest, and logistic regression are all scored here.
> On this pair, EC-XGB separates the two answers most clearly.
> The full comparison takes about 12 milliseconds.

**[2:20-2:30] Hand back**

> This is the complete workflow, from input to explanation. Thank you for watching.

---

## 5. Slide 16 — Conclusion (0:12)

> To conclude.
> We built a detector that needs no model weights, runs in milliseconds, and explains
> every score.
> In-domain F1 is 0.986, with an ECE of 0.007.
> On shifted data, the flag rate falls from 99.9 percent to 61.3 percent, and AUROC
> improves.

---

### Slide 17 — Thank you (0:06)

> Thank you for your attention. We are happy to take your questions.

---

## 6. Hand-over lines (only if each member presents a part)

Use these if you share the talk. They keep the register formal and the flow smooth.

- Into the dataset section: "I will now hand over to my teammate, who will present the
  datasets and the features."
- Into the methodology section: "Thank you. I will now explain our method and the training
  protocol."
- Into the demo: "I will now demonstrate the working system."
- Into the conclusion: "Thank you. I will now summarise our findings."

---

## 7. Limitations and future work (say only if the judges ask)

Keep these ready. Do not volunteer all of them in the talk.

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
| Judge recall | 74% |
| XGBoost F1, same subset | 0.985 |
| Judge cost per 1,000 | about $0.105 |
| Features | 35 (26 base + 8 claim + 1 source) |

---

## 9. Delivery tips

- Speak at your own pace. This plan already assumes a slow speaker, about 2.2 words per
  second.
- Pause for one full second after every number. Numbers need air.
- Point at the screen with the cursor when you mention a table or a chart.
- If you fall behind, shorten slides 5, 7, and 8. Never shorten the demo.
- Keep your hands still, and look at the camera on slides 1 and 17.
- Practise the demo twice, with the backend already running.
- The demo is your strongest moment. Show the evidence quote, not only the score.
