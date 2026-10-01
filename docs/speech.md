# HaluRISC — Video Presentation Speech (12 minutes)

Total time: **about 11:57**. Slides: **6:24**. Live demo: **4:00**. Journal key
results: **1:05**.

This is the full speaking script. Slide numbers refer to the `/slide` deck (16
slides: title, 13 content slides, thank you). The deck has no speaker notes on
purpose, so this file carries everything.

The script is written for a slow, careful speaker at **1.73 words per second**
(about 104 words per minute). That rate is measured from the first recording
pass, so the timings below should hold without rushing.

Read the video in this order: the slides first, then the live demo, then the
journal key results, then the two closing slides. Every sentence connects to the
one before it, and each block ends on a line that sets up the next block. Read it
as continuous speech, not as separate bullet points.

---

## 1. Before you record

- **Upload `context.txt` on the chat page.** Open the Evidence panel and drop the
  file in, so the document index has the demo evidence. Wait for the
  "passages indexed" toast. The file holds the passages for every demo case.
- **Open the chat page with the hallucinated example already loaded.** Use the
  "Hallucinated answer" starter (capital of France). The grounded example
  (penicillin) is one click away.
- **Open the Analyze page with the hallucinated-number example loaded.** The
  grounded example is one click away.
- **Start the backend early.** It takes about 30 seconds to load its models. Use
  `scripts\serve_api.cmd`, which keeps it alive automatically.
- **Open the deck at slide 1** in a second tab, and switch to it when the slides
  start.
- **Have the journal PDF open at page 1** in a third tab for the journal segment.

**Good opening habits.** Open formally, then land the hook on slide 2. Give one
hard number early: "Today's detector flags 99.9% of answers as risky. Ours flags
61.3%." Pause for one full second after every number. Be honest about limits, and
close on the shift result, because in-domain differences are not statistically
significant.

---

## 2. Timing plan

| # | Block | Time | Running |
|---|-------|------|---------|
| 1 | Title | 0:25 | 0:25 |
| 2 | Introduction | 0:51 | 1:16 |
| 3 | Motivation and research gap | 0:29 | 1:45 |
| 4 | Objective | 0:25 | 2:10 |
| 5 | Dataset | 0:31 | 2:41 |
| 6 | Conventional method and baselines | 0:24 | 3:05 |
| 7 | Proposed method | 0:28 | 3:33 |
| 8 | Flow diagram | 0:37 | 4:10 |
| 9 | Experimental setup | 0:24 | 4:34 |
| 10 | Results: in-domain | 0:23 | 4:57 |
| 11 | Results: shift and calibration | 0:28 | 5:25 |
| 12 | Applications | 0:35 | 6:00 |
| 13 | UI: Chat | 0:20 | 6:20 |
| 14 | UI: Analyze | 0:17 | 6:37 |
| — | **Live demo** | **4:00** | **10:37** |
| — | **Journal key results** | **1:05** | **11:42** |
| 15 | Conclusion | 0:23 | 12:05 |
| 16 | Thank you | 0:05 | 12:10 |

The block times are word counts divided by the measured pace of 1.73 words per
second, so they already include your natural pauses.

The demo sits after slide 14. It is shown live, so there is no walkthrough slide.
The journal segment comes right after the demo, over the paper PDF. Then press
the right arrow twice to reach the closing slides.

---

## 3. Slide script (slides 1 to 14)

### Slide 1 — Title (0:25)

> Hi everyone. Our team name is Phantom Devs, and the project is HaluRISC.
> The full title is: Evidence-Consistent XGBoost for Calibrated Hallucination Risk
> Estimation in LLM Answers.
> I will finish with a short demonstration and the key results from our journal paper.

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

---

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

---

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
> In chat, the answer streams and the risk card appears on its own with the verdict and
> score.
> The card leads with the claim verdicts and the evidence sentence behind them.

---

### Slide 14 — UI: Analyze (0:17)

> Analyze adds full input control.
> The SHAP chart explains the raw score, the table lists all 35 features, and the
> comparison card scores every model side by side.

---

## 4. Live demo script (4:00)

The demo is shown live. The **chat page is the core**, because it shows the
everyday workflow: the answer streams, and the risk card appears by itself. Then
Analyze shows the same model under full input control.

Prepare both pages before recording, with the examples already loaded and
`context.txt` already indexed. Most of the demo is action on screen, so the
narration is short. Let each panel land before you speak again.

The narration is about 3:00. The rest is the streaming wait, the panel
expansions, and the page switches. The first prediction after startup is slow
(about two seconds); later ones are about 150 milliseconds.

| Beat | Time | Running |
|------|------|---------|
| 1 Set the scene | 0:15 | 0:15 |
| 2 Chat · hallucinated | 0:50 | 1:05 |
| 3 Chat · grounded | 0:30 | 1:35 |
| 4 Chat · why this score | 0:25 | 2:00 |
| 5 Analyze · hallucinated number | 0:55 | 2:55 |
| 6 Analyze · model comparison | 0:30 | 3:25 |
| 7 Analyze · run all three | 0:25 | 3:50 |
| 8 Hand back | 0:10 | 4:00 |

**[0:00-0:15] Set the scene**

> To make this concrete, I will show the system working.
> There are two pages: chat for everyday use, and Analyze for a close look at one answer.
> A small evidence file is already loaded.

**[0:15-1:05] Chat · hallucinated**

> I paste a question about the capital of France, and the answer streams back.
> It says Lyon.
> Watch the card that appears under it.
> The headline reads likely hallucinated, and the score is about 70 percent.
> The single claim is marked contradicted, and the card shows the sentence from my
> evidence file: the capital is Paris.

**[1:05-1:35] Chat · grounded**

> Now the opposite case.
> I click the grounded starter and ask who discovered penicillin.
> The answer says Alexander Fleming.
> The headline reads grounded in the evidence, the score drops to about 6 percent, and
> the claim is supported.

**[1:35-2:00] Chat · why this score**

> I expand why this score.
> These are the SHAP bars for the raw model, and below them the full list of all 35
> features.
> The grounding line shows that my uploaded file was the source.

**[2:00-2:55] Analyze · hallucinated number**

> Now Analyze, where I control every input.
> I click the built-in example: the context says 5 days per decade, and the answer
> says 10.
> The gauge shows medium risk, 35 percent, and the claim is contradicted.
> The SHAP bars explain the raw score, led by the question-answer overlap and the
> answer length.

**[2:55-3:25] Analyze · model comparison**

> The same panel scores every model on this answer.
> EC-XGB, XGBoost, random forest, and logistic regression all sit near 0.999, while
> the overlap heuristic sits near 0.08.
> The heuristic only sees shared words, so it misses the contradiction that the
> learned models catch.

**[3:25-3:50] Analyze · run all three**

> Finally, one click scores all three built-in examples.
> The hallucinated number reads 35 percent, the grounded answer 5 percent, and the
> short one-word answer 3 percent, each with its latency.

**[3:50-4:00] Hand back**

> That is the full workflow, from a chat answer to a clear reason.
> Now the journal paper behind it.

---

## 5. Journal key results (1:05)

Scroll the journal PDF while you speak. Let the title page show, then the results
tables. Keep the pace slow and point at each table as you name it.

> This study is written up as a journal paper in Elsevier format, with six sections
> and 30 verified references.
> Three results carry it.
> In-domain, EC-XGB reaches F1 0.9855 and AUROC 0.9984, but that benchmark is
> saturated.
> The value appears under domain shift.
> On RAGTruth, the standard model flags 99.9 percent of answers and EC-XGB flags
> 61.3 percent, with AUROC rising from 0.475 to 0.582 and ECE falling from 0.635 to
> 0.278.
> The deployed calibrator cuts the display error from 0.73 to 0.13.
> The paper also reports the honest negatives: source-domain recalibration does not
> help, and FaithBench trades F1 for a much lower flag rate.

If you have extra seconds, add: "The paper reports one analysis at 62 milliseconds
and about one thousandth of a dollar per thousand predictions, roughly one hundred
times less than the GPT judge it beats."

---

## 6. Closing slides

### Slide 15 — Conclusion (0:23)

> Our detector needs no model weights, runs in 62 milliseconds, and gives a reason per
> score.
> In-domain F1 is 0.9855, with ECE 0.007.
> On shifted data the flag rate falls from 99.9 to 61.3 percent.

---

### Slide 16 — Thank you (0:05)

> Thank you. We are happy to take your questions.

---

## 7. Hand-over lines (only if each member presents a part)

- Into the objective: "I will now state our objective and contributions."
- Into the dataset section: "I will now hand over to my teammate, who will present the
  datasets and the features."
- Into the methodology section: "Thank you. I will now explain our method and the training
  protocol."
- Into the demo: "I will now demonstrate the working system."
- Into the journal results: "I will now show the key results from our journal paper."
- Into the conclusion: "Thank you. I will now summarise our findings."

---

## 8. Limitations and future work (say only if the judges ask)

- The models and features are English only.
- On held-out RAGTruth QA, the gain is mainly in calibration, not in F1.
- On FaithBench the model becomes conservative. It trades F1 for a much lower flag rate.
- Some multi-source gains come from task types seen during training.
- The calibrated score is style-sensitive. A full-sentence answer can look risky even
  when every claim is supported, which is why the card leads with the claim verdicts.
- Future work: multicalibration under stronger shift, per-domain thresholds, and neural
  baselines beside EC-XGB.

---

## 9. Number cheat sheet

| Item | Value |
|------|-------|
| In-domain F1 (EC-XGB) | 0.9855 |
| In-domain AUROC | 0.9984 |
| In-domain ECE | 0.0074 |
| F1 bootstrap 95% CI | 0.9812 - 0.9895 |
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

### Verified demo numbers (measured on the live API)

| Demo case | Result | Spoken number |
|-----------|--------|---------------|
| Chat, France (Lyon answer) | likely hallucinated, claim contradicted | about 70% |
| Chat, penicillin (Fleming answer) | grounded, claim supported | about 6% |
| Analyze, Arctic number (10 vs 5) | medium risk, claim contradicted, raw 0.999 | 35% |
| Analyze, Juarez grounded | low risk | 5% |
| Analyze, one-word answer | low risk | 3% |
| Compare, Arctic wrong answer | heuristic 0.083, learned models 0.999 | about 0.08 vs 0.999 |
| First call after startup | cold latency | about 2 s |
| Later calls | warm latency | about 150 ms |

---

## 10. Delivery tips

- The rate is already set for a slow speaker. Do not add sentences on the day.
- Pause for one full second after every number.
- Point at the screen when you mention a table or a chart.
- If you fall behind, drop the last sentence of slides 5, 7, 9, and 11. Never shorten the
  demo or the journal numbers.
- In the demo, let the card and the gauge appear before you speak. Silence while a panel
  loads reads as confidence.
- Show the evidence sentence, not only the score. Judges remember the reason.
- In the journal segment, name each number as it appears on the PDF page.
- Look at the camera on slide 1, at the end of the demo, and on slide 16.
