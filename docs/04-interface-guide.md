# 04 - Interface guide (what everything means)

This guide explains every screen of the HaluRISC web app in plain language.
It is written for a first-time viewer: a teammate, a reviewer, or a supervisor
who opens the demo and wants to know what the numbers, colors, and labels
actually mean.

For how the system was built, read
[`01-project-and-methods.md`](01-project-and-methods.md). For the numbers behind
each experiment, read [`02-experiments-and-results.md`](02-experiments-and-results.md).
For the technical architecture of the web app (routes, API contract), read
[`b7-research-ui.md`](b7-research-ui.md).

## Contents

1. [Before you start](#1-before-you-start)
2. [Chat mode](#2-chat-mode)
3. [Analyze mode](#3-analyze-mode)
4. [Dashboard](#4-dashboard)
5. [Presenter demo](#5-presenter-demo)
6. [About page](#6-about-page)
7. [Reading the numbers correctly](#7-reading-the-numbers-correctly)
8. [Troubleshooting](#8-troubleshooting)
9. [Where the numbers come from](#9-where-the-numbers-come-from)

---

## 1. Before you start

The app has two parts that can run separately.

| Part | What it does | Needed for |
|---|---|---|
| Next.js frontend (`web/`) | all pages, chat UI, dashboards | everything |
| FastAPI backend (`src/api/`) | model inference, claim checks, SHAP | Chat risk cards and Analyze mode |
| OpenAI key (`web/.env.local`) | the chat model itself | typing in Chat mode |

If the backend is offline, Chat mode still loads and the header shows an
offline badge. Analyze mode shows a friendly error with a retry button, and the
Presenter demo works fully because it only reads saved files.

The theme toggle in the top bar switches between dark and light. Both themes
carry the same information.

## 2. Chat mode

Route: `/chat`. You ask a question in natural language, the assistant answers,
and every answer is checked against your evidence after it finishes streaming.

### 2.1 The context bar

The bar above the conversation controls what the risk check treats as evidence.

- **Evidence button** opens the panel where you paste reference text. You can
  also drop PDF, DOCX, TXT, or MD files, and they get indexed in the local
  backend. The summary line next to the title always shows what grounding is
  active.
- **Auto risk check switch** turns the automatic card on or off. Leave it on for
  a demo.
- **Web search switch** lets the claim check search the web for evidence when
  your pasted text and documents do not cover a claim.
- **Grounding priority** is fixed: pasted text first, then uploaded documents,
  then web search, then the conversation itself.

### 2.2 The chat message

Assistant answers appear in a card with a header.

- The **header** shows the assistant name and a `thinking` dot while the model
  is streaming.
- The **copy** button copies the answer text.
- A **Stop** button replaces the send button while the model is generating, so
  you can cancel a long answer.

### 2.3 The risk card (the important part)

The card under each answer is the actual HaluRISC check. It reads top to bottom.

| Element | Meaning |
|---|---|
| Headline verdict | "Grounded in the evidence", "Needs a closer look", or "Likely hallucinated". When claims were checked, the claim evidence decides this headline. |
| Percentage | The evidence-calibrated model score. It is a probability, so 47% means answers like this one were hallucinated in about 47% of similar calibration cases. |
| Meter and ticks | The small vertical ticks mark the low, medium, and high cutoffs used by the API. The filled part is this answer's score. |
| Grounding line | Which evidence was used: your pasted context, your uploaded documents, web search, or the conversation only. Conversation-only means no external grounding, so treat the score as weaker. |
| Claim verdicts | Per-claim labels: **supported** (evidence backs the claim), **contradicted** (evidence says the opposite, with the quote shown), **unsupported** (no evidence either way), **not judged** (no evidence retrieved, so the system abstained). |
| Why this score | The top SHAP features for the raw model, drawn as red (raised risk) or green (lowered risk) bars, with the same scale so you can compare bar lengths. |
| Feature disclosure | The full list of measured features and their values, plus the thresholds. This is for inspection, not for a live demo. |
| Thumbs buttons | Records your agreement with a verdict into a local feedback file on this machine. It is used for the error analysis, not for training. |

The card also has a "How to read this card" section at the bottom with the same
information in short form.

## 3. Analyze mode

Route: `/analyze`. This is the direct interface to the model: no chat model in
the loop. You provide all three inputs yourself.

### 3.1 The form

- **Question**: what was asked.
- **Reference context / evidence**: the trusted text the answer should be based
  on.
- **Candidate answer A**: the text you want to score.
- **Compare two answers**: adds answer B and scores it against the same
  question and evidence.
- **Start from an example**: three ready-made cases. "Hallucinated number"
  scores medium to high. "Grounded answer" scores low. "Borderline case" sits
  near the decision boundary.

### 3.2 The result panel

- **Gauge**: the needle shows the calibrated probability. The two notches on
  the arc are the low/medium and medium/high cutoffs. The chips under the gauge
  spell out the same cutoffs as numbers.
- **Label pill**: the same decision, written as Low risk, Medium risk, or High
  risk.
- **SHAP chart**: explained in section 3.3.
- **All features**: the full feature table grouped by family (lexical overlap,
  entity coverage, NLI consistency, numeric consistency, hedging, semantic
  drift, length).

When you compare two answers, a banner above the panels states which answer is
safer and by how many points.

### 3.3 How to read the SHAP chart

SHAP is a method that splits a model score into per-feature contributions. In
this app the chart answers one question: "which measured properties of this
answer pushed the raw score up, and which pushed it down?"

- The vertical axis lists feature names in plain words, for example
  "Context Overlap" or "Novel Numbers Found".
- The horizontal axis is the contribution size. Bars to the right are red and
  raised the raw risk. Bars to the left are green and lowered it.
- The **model average** chip is the score the model would give if no feature
  spoke for or against the answer. Each bar is added on top of that average.
- Only the largest contributions are drawn, so a small effect can be missing
  from the chart but still present in the full feature list.

Worked example. Suppose the model average is 0.31, "Novel Numbers Found" shows
+0.35 in red, and "Context Overlap" shows -0.42 in green. Reading: the answer
contains numbers that do not appear in the context, which raised the raw risk,
but the answer also overlaps strongly with the context, which lowered it more.
The row values are contributions to the raw output, so they do not sum to the
calibrated percentage that the gauge shows.

## 4. Dashboard

Route: `/dashboard`. One tab per research question. Every number is read from
`artifacts/results/` at request time. A "How to read these metrics" panel at the
top of every tab defines the metric names.

| Tab | Question it answers |
|---|---|
| Overview | How good is the model on the in-domain test split, and how does it compare with baselines? Also shows leakage-removal impact and the manifest provenance. |
| Robustness | What happens on external data (RAGTruth, FaithBench) with no retraining? Includes per-subgroup results and label-mapping sensitivity. |
| Calibration | Do the probabilities mean what they say? Compares raw, Platt, and isotonic calibration per subset, and shows target calibration on RAGTruth QA. |
| Explainability | Are the explanations stable? Group importance, neutralization curve, perturbation stability, bootstrap stability, and the reviewer export. |
| Failures | Which cases failed and how? Perturbation-induced failures, zero-shot error cases, and the Version A error analysis. |
| Efficiency | What does one prediction cost? Per-module latency, cost per 1,000 predictions, and the LLM-as-judge comparison. |

Quick definitions used across tabs:

- **F1**: balance of precision and recall, the headline detection metric.
- **AUROC**: chance that a random positive case scores above a random negative.
- **PR-AUC**: precision-recall area, more informative under class imbalance.
- **MCC**: a balanced single-number quality score.
- **ECE**: expected calibration error, the gap between predicted probabilities
  and observed frequencies. Lower is better.
- **Brier**: mean squared error of the probabilities. Lower is better.
- **Platt / isotonic**: two post-hoc calibration methods. Platt is the default.
- **SHAP**: per-feature attribution of a model score.
- **McNemar**: paired significance test between two classifiers.
- **Bootstrap CI**: confidence interval from resampling.

## 5. Presenter demo

Route: `/demo`. A single offline page that needs no API key and no network. It
walks through five numbered sections: the problem, three real scored cases, one
real failure, the calibration evidence, and the transfer evidence. Use it when
the backend is not running or when a live demo would be risky.

## 6. About page

Route: `/about`. The pipeline in four steps, the headline evidence numbers, the
tech stack with versions from the manifest, and a "Reading the interface"
section that links to the screens above and to this guide.

## 7. Reading the numbers correctly

- **The calibrated score and the raw score are different things.** The gauge
  and headline use the calibrated probability. SHAP explains the raw model
  output. They can point in different directions, and the app says so on the
  card.
- **A low score is not a correctness proof.** It means the measured evidence
  did not look contradictory or unsupported. A fluent, confident answer can
  still be wrong in ways the features cannot see.
- **A high score is not a lie detector.** It means the answer behaves like the
  hallucinated examples the model was trained on.
- **Conversation-only grounding is weaker.** When no external evidence is set,
  the score compares the answer with the conversation, not with a source.
- **Transfer is harder than the in-domain number suggests.** The dashboard
  shows the drop on external data on purpose. Quote both when asked.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Header badge says "ML backend offline" | FastAPI is not running on port 8000 | start `uvicorn src.api.main:app --port 8000` from the repo root |
| Chat answers never appear | No `OPENAI_API_KEY` in `web/.env.local` | add the key, or use Analyze mode and the presenter demo |
| First prediction takes a long time | The backend preloads spaCy, the NLI model, and SBERT at startup | wait for the first request, later requests are fast |
| Risk card says "could not reach the local ML service" | Backend stopped mid-session | restart the backend and resend the message |
| Analyze mode shows "Analysis failed" | Backend offline or the request was rejected | press Retry, then check the technical details line |
| Charts look empty on a screenshot | They render instantly now, so a reload fixes a partial capture | reload the page |

## 9. Where the numbers come from

| Screen element | Source |
|---|---|
| Risk card verdicts and score | live `/api/ml/verify` call to the local backend |
| Analyze gauge and SHAP | live `/api/ml/predict` and `/api/ml/explain` calls |
| Dashboard tables and charts | `artifacts/results/**` read on the server at request time |
| Presenter demo | the same artifact files, no network calls |
| Version badge in the header | `artifacts/results/manifest.json` |
| Feedback thumbs | appended to `data/processed/feedback_log.jsonl` (local, gitignored) |

Nothing in the interface is typed by hand. If an artifact is missing, the
screen shows a dash or an empty state instead of a fake number.

## Related documents

- [`01-project-and-methods.md`](01-project-and-methods.md) - what the project is and how it works
- [`02-experiments-and-results.md`](02-experiments-and-results.md) - experiment numbers with artifact paths
- [`03-reproduction-and-defense.md`](03-reproduction-and-defense.md) - rebuild, demo script, and defense Q&A
- [`b5-explanation-reliability.md`](b5-explanation-reliability.md) - why the explanations are trusted
- [`b7-research-ui.md`](b7-research-ui.md) - routes, API contract, and deployment notes
