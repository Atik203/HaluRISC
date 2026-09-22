# Title explanations, word by word

This page explains the journal title one word at a time. It is written for the
first question a supervisor usually asks: "What does the title mean, and why is
each word there?"

Full title:

> **Estimating Hallucination Risk in Black-Box Large Language Model Answers with
> Calibrated and Explainable Machine Learning**

Short title (running head):

> **Calibrated Hallucination Risk Estimation**

The title states the problem (hallucination risk in large language model
answers) and the method (calibrated and explainable machine learning) without a
colon and without the system name. The paper text follows the same rule: it uses
"this study" and "the proposed model" instead of "we" and instead of the
software name.

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

The work trains a lightweight model that reads a question, a reference context,
and a candidate answer, and it estimates the probability that the answer is
hallucinated. The title lists the four properties that make the estimate useful:
it works without access to the generating model (**black-box**), its numbers mean
what they say (**calibrated**), each score comes with a reason (**explainable**),
and the whole method is classical **machine learning** on engineered features, so
it runs on a CPU.

## 2. Word-by-word table

| Word or phrase | What it means | Why it must be in the title | Where it is proven |
|---|---|---|---|
| **Estimating** | Producing a probability for each answer, not a hard verdict. | Sets the statistical claim. The output is a number with a calibration guarantee, so "estimating" is accurate; "detecting" would imply a binary decision, which is only the thresholded view of the same score. | `P(hallucinated \| q, c, a)` in the abstract; scores and thresholds across B2 and B4 |
| **Hallucination** | Text that is fluent and confident but not supported by the available evidence. | Names the problem, not a mechanism. It connects the work to the datasets that label exactly this phenomenon (HaluEval, RAGTruth, FaithBench). | Definitions in the Conventional Method preliminaries; the three datasets |
| **Risk** | A probability of being hallucinated, not a verdict of truth or falsehood. | The framing rule of the project. "Risk" keeps the claim honest and measurable, and it matches the deployment use, which is triage rather than proof. | The label policy in `docs/01` §2; the abstract target |
| **in** | Grammatical link. | Joins the property to the object of study. | (grammar) |
| **Black-Box** | The detector never inspects the generating model: no weights, no logits, no attention maps. The inputs are plain text. | States the deployment constraint. It shows the method works when the LLM is closed or unknown, and it justifies the CPU cost profile. Without this word, readers may assume weight access like encoder probes. | 26 text-only features in `artifacts/models/feature_names.json`; latency and cost artifacts |
| **Large Language Model** | The class of systems that produce the answers. | Fixes the object of study. The problem only exists at this scale, where fluent unsupported text is easy to produce. | Dataset composition in `docs/01` §2 |
| **Answers** | The candidate texts being scored: short QA answers, summaries, and chat replies. | Says the unit of analysis is a response, not a model. The system scores an answer against evidence, so it can score answers from any generator, including future ones. | QA and summarization subsets in B3; claim-level verification in the deployed chat layer |
| **with** | Grammatical link. | Introduces the method that does the estimating. | (grammar) |
| **Calibrated** | A score of 0.8 should mean that about 80 percent of such answers are hallucinated. Post-hoc methods (Platt, isotonic) align predicted probabilities with observed frequencies. | Detection papers report F1 and AUROC, which say nothing about whether the probability value is trustworthy. A risk score is used as a probability, so this word is the reliability promise. | B4, `artifacts/results/b4/b4_calibration_metrics.json`, `b4_target_calibration.json`; in-domain ECE 0.0045, RAGTruth QA ECE 0.8185 before and 0.1335 after target recalibration |
| **and** | Grammatical link. | Joins the two method properties. | (grammar) |
| **Explainable** | Every prediction is broken into per-feature contributions with SHAP, and the explanations are tested for stability. In the deployed system, answers also receive claim-level checks with evidence quotes. | A flagged answer is useless to a user who cannot see why. This word separates a bare score from a tool a reviewer can audit. | B5, `b5_feature_importance.json`, `b5_stability_bootstrap.json`; top-1 feature never flips under controlled edits, bootstrap top-5 set Jaccard = 1.0 |
| **Machine Learning** | A trained XGBoost classifier over engineered evidence-consistency features, with calibration fitted on validation data. | Names the method family instead of one library. It tells the reader this is a learned statistical model, not a prompt wrapper and not a neural probe that needs model access. | B2 training protocol; `docs/01` §3 and §4 |

## 3. What each word would cost if removed

| Removed word | What the title would stop promising |
|---|---|
| Estimating | The output would sound like a hard label instead of a calibrated probability. |
| Hallucination | The problem statement would be vague, and the datasets would not connect to the title. |
| Risk | The work would sound like a truth oracle, which the system is not. |
| Black-Box | The method would look like another model-probing approach, and the CPU and cost advantages would lose their reason. |
| Large Language Model | The target of the analysis would be unclear. |
| Answers | The unit of scoring (an answer, not a model) would be missing. |
| Calibrated | The probability would be just a ranking score, and the ECE work (0.8185 to 0.1335) would seem unimportant. |
| Explainable | SHAP stability, perturbation tests, and the expert audit would look like extras instead of title-level claims. |
| Machine Learning | The method would be undefined; readers could not tell what kind of system produces the estimate. |

## 4. Supervisor Q&A

**Why does the title not mention the system name?**
The title states the problem and the method so that any reader understands the
work from the title alone. The system name remains in the repository and in the
deployed application, and the paper introduces the model as "the proposed
model".

**Why not use a colon style such as "Name: Subtitle"?**
A colon title leads with the product name and pushes the problem to the second
half. The current title is one readable sentence: problem first, method second.

**Why "Estimating" and not "Detection"?**
The model outputs a calibrated probability `P(hallucinated | q, c, a)`.
Thresholds turn that probability into low, medium, and high bands. "Detection"
describes only the thresholded decision and hides the probability quality, which
is a core contribution.

**Why is XGBoost not in the title?**
The model is the means. The title names the properties the system must have
(calibrated, explainable, black-box) and the method family (machine learning).
The classifier choice is described and ablated in the paper.

**Why "Black-Box" when the system still needs a reference context?**
Black-box refers to the generating model, not to the evidence. The LLM is never
opened. The reference context is the material a user already has, and the system
also works without it, with a weaker grounding signal.

**Does the title still promise cross-domain evaluation?**
Not in the words, but the paper keeps it as a main result. The abstract and the
experiments report zero-shot transfer on RAGTruth and FaithBench and the
recalibration that follows. The earlier title carried "cross-domain"; the
shorter title keeps the promise in the body, where the evidence lives.

**Why "Answers" and not "Outputs" or "Responses"?**
"Answers" is the most readable of the three in a title, and it covers QA
answers, summaries, and chat replies. The paper text still uses "responses" when
it refers to RAGTruth data because that is the dataset wording.

**Is "Machine Learning" too broad?**
It is the honest method family: engineered features plus a tuned gradient-boosted
tree classifier plus calibration. The specific tools (XGBoost, SHAP, Platt)
appear in the abstract, keywords, and method sections.

**Can every word be checked against a file?**
Yes. The artifact map below lists the file that backs each word.

## 5. Numbers behind the title

| Title word | Number | Source |
|---|---|---|
| Estimating | Target `P(hallucinated \| q, c, a)`; grouped 70/15/15 split, seeds 42/123/456 | abstract; `artifacts/split_indices.json` |
| Hallucination, Risk | F1 = 0.9846 and AUROC = 0.9979 in-domain; zero-shot F1 = 0.3022 on RAGTruth QA | `artifacts/results/b2/b2_model_comparison.json`, `b3/b3_dataset_metrics.json` |
| Black-Box | 26 features, 7 groups, CPU inference, 61.8 ms per analysis | `artifacts/models/feature_names.json`; efficiency artifacts |
| Answers | HaluEval QA (20,000 rows) plus RAGTruth and FaithBench subsets | `docs/01-project-and-methods.md` §2; `manifest.frozen.json` |
| Calibrated | ECE 0.0045 in-domain; 0.8185 to 0.1335 after target recalibration | `artifacts/results/b4/b4_calibration_metrics.json`, `b4_target_calibration.json` |
| Explainable | Top-1 flip rate 0.0; top-5 Jaccard 1.0; Kendall tau 0.6625 | `artifacts/results/b5/b5_stability_bootstrap.json`, `b5_feature_importance.json` |

## 6. Artifact map

| Word | Primary artifact | Supporting doc |
|---|---|---|
| Estimating, Risk | `artifacts/results/b2/` metric tables | `docs/02` B2 section |
| Calibrated | `artifacts/results/b4/` | `docs/02` B4 section |
| Explainable | `artifacts/results/b5/` | `b5-explanation-reliability.md` |
| Black-Box, Answers | `src/features/extract_features.py`, `src/api/main.py` | `docs/01` §5, `b7-research-ui.md` |
| Machine Learning | `artifacts/models/` (booster, scaler, calibrator) | `docs/01` §3, §4 |
| Everything | `docs/manifest.frozen.json` | `docs/03-reproduction-and-defense.md` |

## 7. The 30-second spoken answer

"The title has four commitments. The method is machine learning over engineered
evidence features, so it runs on a CPU and stays explainable. It works in a
black-box setting, because it never touches the language model, only the
question, the evidence, and the answer. It estimates risk instead of guessing
truth, which keeps the claim measurable. And it is calibrated, so the number it
prints can be used as a probability. The paper reports where that works
in-domain and where it breaks under domain shift."

## 8. Naming history

- **HaluRISC** (Hallucination Risk Scoring and Calibration) is the system and
  repository name. It remains in the code, the API, the web application, and the
  notebook filename (`colab/HaluRISC_Training_Version_B.ipynb`).
- The earlier title, *HaluRISC: Cross-Domain Calibrated and Explainable
  Hallucination Risk Estimation for Black-Box LLM Responses*, used the colon
  style and led with the product name.
- The retired course-format manuscript used the title *HaluRISC: A Calibrated
  and Explainable Machine Learning Framework for Hallucination Risk Prediction
  in Black-Box LLM Outputs*. Its sources remain in git history.
- The current title was adopted after supervisor feedback: no colon, no product
  name, readable problem statement, and method family stated directly.

## Related documents

- [`01-project-and-methods.md`](01-project-and-methods.md) - what the system is and how it works
- [`02-experiments-and-results.md`](02-experiments-and-results.md) - the numbers cited above with artifact paths
- [`03-reproduction-and-defense.md`](03-reproduction-and-defense.md) - defense Q&A and the demo script
- [`04-interface-guide.md`](04-interface-guide.md) - what each screen of the demo means
- [`05-solving-hallucination-plan.md`](05-solving-hallucination-plan.md) - how the risk layer fits a full mitigation loop
