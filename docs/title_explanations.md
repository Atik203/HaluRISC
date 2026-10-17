# Title explanations, word by word

This page explains the journal title one word at a time. It is written for the
first question a supervisor usually asks: 'What does the title mean, and why
is each word there?'

Full title:

> **Calibrated Hallucination Risk Estimation in Large Language Models with Evidence-Consistent XGBoost**

Short title (running head):

> **Evidence-Consistent XGBoost**

The title states the task and the setting first (calibrated estimation of
hallucination risk in large language models) and the method second
(evidence-consistent XGBoost). A reviewer who knows nothing about the project
can read the first half alone and understand what the work measures, and the
second half names how it is done. The title stays one readable sentence
without a colon and without the product name. The paper text follows the same
rule: it uses 'this study' and 'the proposed model' instead of 'we' and
instead of the software name.

## Contents

1. [The title in one breath](#1-the-title-in-one-breath)
2. [Word-by-word table](#2-word-by-word-table)
3. [What each word would cost if removed](#3-what-each-word-would-cost-if-removed)
4. [Supervisor Q&A](#4-supervisor-qa)
5. [Numbers behind the title](#5-numbers-behind-the-title)
6. [Artifact map](#6-artifact-map)
7. [The 30-second spoken answer](#7-the-30-second-spoken-answer)
8. [Naming history](#8-naming-history)

---

## 1. The title in one breath

The work trains a lightweight model that reads a question, a reference
context, and a candidate answer, and it estimates the probability that the
answer is hallucinated. The detector never inspects the generating model, so
the inputs are plain text and the method runs on a CPU. The title names the
three claims that carry the work: the output is a probability whose value can
be used as a probability (**calibrated hallucination risk estimation**), the
setting is the responses of large language models (**in large language
models**), and the classifier is a tuned gradient-boosted tree model whose
features track the evidence (**evidence-consistent XGBoost**).

## 2. Word-by-word table

| Word or phrase | What it means | Why it must be in the title | Where it is proven |
|---|---|---|---|
| **Calibrated** | A score of 0.8 should mean that about 80 percent of such answers are hallucinated. Post-hoc methods (Platt, isotonic) align predicted probabilities with observed frequencies. | Detection papers report F1 and AUROC, which say nothing about whether the probability value is trustworthy. A risk score is used as a probability, so this word is the reliability promise, and leading with it puts the contribution first. | `artifacts/results/b6/b6_calibration.json`; in-domain ECE 0.0045, deployed RAGTruth QA display score 0.726 to 0.126, standard-model target recalibration 0.8185 to 0.1335 |
| **Hallucination** | Text that is fluent and confident but not supported by the available evidence. | Names the problem, not a mechanism. It connects the work to the datasets that label exactly this phenomenon (HaluEval, RAGTruth, FaithBench). | Definitions in the Conventional Method preliminaries; the three datasets |
| **Risk** | A probability of being hallucinated, not a verdict of truth or falsehood. | The framing rule of the project. 'Risk' keeps the claim honest and measurable, and it matches the deployment use, which is triage rather than proof. | The label policy in the abstract target, P(hallucinated &#124; q, c, a) |
| **Estimation** | Producing a probability for each answer, not a hard verdict. | Sets the statistical claim. The output is a number with a calibration guarantee, so 'estimation' is accurate; 'detection' would imply a binary decision, which is only the thresholded view of the same score. | P(hallucinated &#124; q, c, a) in the abstract; scores and thresholds across the B-runs |
| **in** | Grammatical link. | Joins the task to the setting. | (grammar) |
| **Large Language Models** | The generator family whose answers are scored: short QA answers, summaries, and chat replies. | Fixes the setting without an abbreviation, so a reader outside the field understands it on the first pass. The unit of analysis is a response, not a model, so answers from any generator can be scored. | QA and summarization subsets; claim-level verification in the deployed chat layer |
| **with** | Grammatical link. | Joins the setting to the method. | (grammar) |
| **Evidence-Consistent** | The features and the training keep the score tied to the evidence direction, and per-claim NLI aggregates check each atomic clause of the answer. | Names the design that separates the model from a generic XGBoost. Monotone constraints hold the sign of the response, and the claim-level signals let the classifier see the same evidence the deployed verification layer reports. | 35 inputs (26 base plus 8 claim aggregates plus a source indicator); the m1/m2/m3 ablation in `artifacts/results/b6/b6_model_comparison.csv` |
| **XGBoost** | A tuned gradient-boosted tree classifier over engineered evidence-consistency features, with calibration fitted on validation data. | Names the concrete method for discoverability and for a fair technical claim. It tells the reader this is a learned statistical model that runs on a CPU, not a prompt wrapper and not a neural probe that needs model access. | `src/models/train_pipeline.py`; `artifacts/results/b6/b6_model_comparison.csv` |

## 3. What each word would cost if removed

| Removed word | What the title would stop promising |
|---|---|
| Calibrated | The probability would be just a ranking score, and the ECE work (0.726 to 0.126) would seem unimportant. |
| Hallucination | The problem statement would be vague, and the datasets would not connect to the title. |
| Risk | The work would sound like a truth oracle, which the system is not. |
| Estimation | The output would sound like a hard label instead of a calibrated probability. |
| in, Large Language Models | The setting would be unclear, and a reader outside the field would have to guess where the method applies. |
| with | (grammar) |
| Evidence-Consistent | The design would look like a plain feature-based classifier, and the claim-level signals and monotone constraints would seem like extras. |
| XGBoost | The method would be undefined, and readers could not tell what kind of system produces the estimate. |

## 4. Supervisor Q&A

**Why does the title not mention the product name?**
The title states the task and the method so that any reader understands the
work from the title alone. The product name remains in the repository and in
the deployed application, and the paper introduces the model as 'the proposed
model'.

**Why not use a colon style such as Name: Subtitle?**
A colon title leads with the product name and pushes the problem into the
second half. The current title is one readable sentence: task and setting
first, method second.

**Why 'Estimation' and not 'Detection'?**
The model outputs a calibrated probability P(hallucinated &#124; q, c, a).
Thresholds turn that probability into low, medium, and high bands. 'Detection'
describes only the thresholded decision and hides the probability quality,
which is a core contribution.

**Why 'Large Language Models' and not 'LLM Answers' or a black-box phrase?**
The title spells the term out, so it avoids an abbreviation and reads cleanly
for reviewers from other areas. The unit of scoring is still the answer, and
the abstract, the method, and the system section all name the answer as the
object of study. Black-box is a deployment property of the detector, not the
contribution, so it stays in the abstract and the method. The detector never
inspects the generating model, the inputs are plain text, and the method runs
on a CPU. The reference context is the material a user already has, and the
system also works without it, with a weaker grounding signal.

**Does the title still promise cross-domain evaluation?**
Not in the words, but the paper keeps it as a main result. The abstract and
the experiments report zero-shot transfer on RAGTruth and FaithBench and the
recalibration that follows. The shorter title keeps the promise in the body,
where the evidence lives.

**Why 'Risk' and not 'Truth'?**
The score predicts risk, not truth. It flags answers that are unsupported by,
or contradict, the available evidence. Keeping 'risk' makes the claim
measurable and matches the triage use in the interface.

**Can every word be checked against a file?**
Yes. The artifact map below lists the file that backs each word.

## 5. Numbers behind the title

| Title word | Number | Source |
|---|---|---|
| Calibrated | In-domain ECE 0.0045; deployed RAGTruth QA display score 0.726 to 0.126; standard-model target recalibration 0.8185 to 0.1335 | `artifacts/results/b6/b6_calibration.json`; target recalibration table |
| Hallucination, Risk | Standard model flags 99.9 percent of natural RAGTruth answers, EC-XGB flags 61.3 percent; AUROC 0.475 to 0.582 | `artifacts/results/b6/b6_external_metrics.csv` |
| Estimation | One analysis takes 61.8 ms at the median and about USD 0.001 per 1,000 predictions | `artifacts/results/latency_analysis.json`; judge comparison |
| Large Language Models | HaluEval QA (20,000 rows) plus RAGTruth (17,790 rows) and FaithBench (750 rows) | dataset table in `artifacts/results/b6/`; `docs/manifest.frozen.json` |
| Evidence-Consistent | 35 inputs (26 base plus 8 claim aggregates plus source indicator); claim false-alarm count 12 to 8 on seed 42 | `artifacts/results/b6/b6_model_comparison.csv`; `b6_feature_names.json` |
| XGBoost | F1 = 0.9855 and AUROC = 0.9984 in-domain, 3-seed mean | `artifacts/results/b6/b6_model_comparison.csv` |

## 6. Artifact map

| Word | Primary artifact | Supporting doc |
|---|---|---|
| Calibrated | `artifacts/results/b6/b6_calibration.json` | `docs/02` B4 section |
| Hallucination, Risk | `artifacts/results/b6/b6_external_metrics.csv`; `docs/manifest.frozen.json` | `docs/02` B3 section |
| Estimation | `artifacts/results/latency_analysis.json`, `llm_judge_results.json` | `docs/02` efficiency section |
| Large Language Models | dataset table in `artifacts/results/b6/`; `docs/manifest.frozen.json` | `docs/01` section 3 |
| Evidence-Consistent, XGBoost | `artifacts/results/b6/` metric tables; `src/features/` | `docs/02` B6 section |

## 7. The 30-second spoken answer

'The title names the task first: calibrated estimation of hallucination risk
in large language models. Calibrated means the number it prints can be used as
a probability, in domain and after target recalibration. The method is an
evidence-consistent XGBoost, so the score follows the evidence direction and
the claims in the answer, and it runs on a CPU. The paper reports where that
works in-domain and where it breaks under domain shift.'

## 8. Naming history

- **HaluRISC** (Hallucination Risk Scoring and Calibration) is the system and
  repository name. It remains in the code, the API, the web application, and
  the notebook filename (`colab/HaluRISC_Training_Version_B.ipynb`).
- The earlier title, *HaluRISC: Cross-Domain Calibrated and Explainable
  Hallucination Risk Estimation for Black-Box LLM Responses*, used the colon
  style and led with the product name.
- The retired course-format manuscript used the title *HaluRISC: A Calibrated
  and Explainable Machine Learning Framework for Hallucination Risk Prediction
  in Black-Box LLM Outputs*. Its sources remain in git history.
- A later title, *Evidence-Consistent XGBoost for Calibrated Hallucination
  Risk Estimation in Black-Box Language Model Answers*, added the full setting
  phrase.
- A shorter title, *Evidence-Consistent XGBoost for Calibrated Hallucination
  Risk Estimation*, dropped the setting entirely.
- The previous title, *Evidence-Consistent XGBoost for Calibrated
  Hallucination Risk Estimation in Large Language Models*, put the method
  first.
- The current title leads with the task and the setting, *Calibrated
  Hallucination Risk Estimation in Large Language Models*, and closes with the
  method, *with Evidence-Consistent XGBoost*. A reviewer understands what the
  work measures from the first half, and the black-box property stays in the
  abstract and the method.

## Related documents

- [`01-project-and-methods.md`](01-project-and-methods.md) - what the system is and how it works
- [`02-experiments-and-results.md`](02-experiments-and-results.md) - the numbers cited above with artifact paths
- [`03-reproduction-and-defense.md`](03-reproduction-and-defense.md) - defense Q&A and the demo script
- [`04-interface-guide.md`](04-interface-guide.md) - what each screen of the demo means
- [`05-solving-hallucination-plan.md`](05-solving-hallucination-plan.md) - how the risk layer fits a full mitigation loop
- [`writing-guideline.md`](writing-guideline.md) - the supervisor's structural guideline for the journal paper
