# Title explanations, word by word

This page explains the journal title one word at a time. It is written for the
first question a supervisor usually asks: "What does the title mean, and why is
each word there?"

Full title:

> **HaluRISC: Cross-Domain Calibrated and Explainable Hallucination Risk
> Estimation for Black-Box LLM Responses**

Short title (running head):

> **HaluRISC: Calibrated Hallucination Risk Estimation**

Everything below can be defended with numbers from the frozen artifacts. The
artifact map at the end of this page says where each claim lives.

## Contents

1. [The title in one breath](#1-the-title-in-one-breath)
2. [Word-by-word table](#2-word-by-word-table)
3. [What each word would cost if removed](#3-what-each-word-would-cost-if-removed)
4. [Supervisor Q&A](#4-supervisor-qa)
5. [Numbers behind the title](#5-numbers-behind-the-title)
6. [Artifact map](#6-artifact-map)
7. [The 30-second spoken answer](#7-the-30-second-spoken-answer)
8. [Journal title vs report title](#8-journal-title-vs-report-title)

---

## 1. The title in one breath

HaluRISC is a lightweight system that reads a question, a reference context, and
a candidate answer, and it estimates the probability that the answer is
hallucinated. The title lists the four properties that make the estimate useful:
it is tested outside the training domain (**cross-domain**), its numbers mean
what they say (**calibrated**), each score comes with a reason (**explainable**),
and it works without access to the generating model (**black-box LLM
responses**).

## 2. Word-by-word table

| Word or phrase | What it means | Why it must be in the title | Where it is proven |
|---|---|---|---|
| **HaluRISC** | The system name. It expands to **Halu**cination **R**isk **S**coring and **C**alibration. | Names the artifact so the title is about a system, not a general essay. A short name also makes the running head possible. | `docs/01-project-and-methods.md` §1.1; the whole repository |
| **Cross-Domain** | The model is trained on one dataset (HaluEval QA) and tested on other datasets (RAGTruth QA, RAGTruth summarization, RAGTruth data-to-text, FaithBench) with no retraining. | Says the evaluation leaves the training domain. Without it, the paper looks like a single-benchmark result. It also signals honesty, because the same word forces us to report the transfer drop. | B3, `artifacts/results/b3/b3_dataset_metrics.json`, `b3_transfer_comparison.csv`; zero-shot F1 = 0.3022 on RAGTruth QA |
| **Calibrated** | A score of 0.8 should mean that about 80 percent of such answers are hallucinated. Calibration methods (Platt, isotonic) are fitted to make predicted probabilities match observed frequencies. | Detection papers report F1 and AUROC, which say nothing about whether the probability value is trustworthy. A risk score is used as a probability, so this word is the reliability promise. | B4, `artifacts/results/b4/b4_calibration_metrics.json`, `b4_target_calibration.json`; in-domain ECE 0.0045, RAGTruth QA ECE 0.8185 before and 0.1335 after target recalibration |
| **Explainable** | Every prediction can be broken into per-feature contributions with SHAP, and the explanation is tested for stability. In the deployed system, chat answers also get claim-level checks with evidence quotes. | A flagged answer is useless to a user who cannot see why. This word is the difference between a black-box score and a tool a reviewer can audit. | B5, `b5_feature_importance.json`, `b5_stability_bootstrap.json`; top-1 feature never flips under controlled edits, top-5 bootstrap set Jaccard = 1.0 |
| **Hallucination** | Text that is fluent and confident but not supported by the available evidence. | Names the problem the paper solves, not the mechanism. It connects the work to the datasets (HaluEval, RAGTruth, FaithBench) that label exactly this phenomenon. | Definitions in §Conventional Method (Preliminaries); the three datasets |
| **Risk** | A probability of being hallucinated, not a verdict of truth or falsehood. The model answers "how risky is this answer", not "is this answer wrong". | The framing rule of the whole project. "Risk" keeps the claim honest and measurable. It also fits the deployment story, where the output routes attention instead of replacing human review. | The target `P(hallucinated \| q, c, a)` in the abstract; the label policy in `docs/01` §2 |
| **Estimation** | Statistical estimation of a probability from evidence features. | The output is a number between 0 and 1 with a calibration guarantee, so "estimation" is the accurate statistical word. "Detection" would suggest a binary decision, which is only the thresholded view of the same score. | Scores, thresholds, and ECE/brier metrics across B2 and B4 |
| **for** | Grammatical link. | Joins the property list to the object of study. | (grammar) |
| **Black-Box** | The detector never inspects the generating model: no weights, no logits, no attention maps. Inputs are the question, an optional reference context, and the candidate answer as plain text. | States the deployment constraint. It shows the method works when the LLM is closed (an API) or unknown, and it keeps the cost on a CPU. Without this word, reviewers may assume weight access like encoder probes. | 26 features from text only, `feature_names.json`; latency and cost artifacts in B4/legacy efficiency files |
| **LLM** | The class of systems that produce the answers, for example the HaluEval generators and the RAGTruth models. | Fixes the object of study to language-model outputs, which is where fluent unsupported text is the problem. | Dataset composition in `docs/01` §2 |
| **Responses** | The candidate answers being scored. The plural covers any answer text: short QA answers, summaries, data-to-text rows, and chat replies. | Says the unit of analysis is a response, not the model itself. The system scores an answer against evidence, so it can score answers from any generator, including future ones. | QA plus summarization subsets in B3; claim-level verification in the deployed chat layer |

## 3. What each word would cost if removed

| Removed word | What the title would stop promising |
|---|---|
| HaluRISC | The work would be a generic method with no reproducible artifact name. |
| Cross-Domain | The transfer experiments would look optional, and the reported failure on RAGTruth would have no place in the framing. |
| Calibrated | The probability would be just a ranking score, and the ECE work (0.8185 to 0.1335) would seem unimportant. |
| Explainable | SHAP stability, perturbation tests, and the expert audit would look like extras instead of title-level claims. |
| Hallucination | The problem statement would be vague, and the datasets would not connect to the title. |
| Risk | The work would sound like a truth oracle, which the system is not. |
| Estimation | The output would sound like a hard binary label instead of a calibrated probability. |
| Black-Box | The method would look like another model-probing approach, and the CPU and cost advantages would lose their reason. |
| LLM | The target of analysis would be unclear, and the results could be mistaken for a general NLP task. |
| Responses | The unit of scoring (an answer, not a model) would be missing. |

## 4. Supervisor Q&A

**What does HaluRISC stand for?**
Hallucination Risk Scoring and Calibration.

**Why "estimation" and not "detection"?**
We output a calibrated probability `P(hallucinated | q, c, a)`. Thresholds turn
that probability into low, medium, and high bands. "Detection" describes only
the thresholded decision and hides the probability quality, which is the core
contribution.

**Cross-domain, but your zero-shot F1 is 0.3022. Is that not a failure?**
Yes, and the title commits to it. B3 reports the failure and explains it with a
length-related cause. B4 then shows that calibration can be restored on target
data (ECE 0.8185 to 0.1335) even while ranking quality stays low. The honest
split between "what transfers" and "what does not" is part of the contribution.

**What exactly proves "calibrated"?**
Three pieces. On the in-domain test split the model lands at ECE = 0.0045, and
the fitted calibrators are reported next to it. Source calibrators do not
transfer to RAGTruth QA (ECE 0.8185). Target recalibration with disjoint groups
reduces the same ECE to 0.1335. Reliability diagrams and Brier scores are
included next to every number.

**What exactly proves "explainable"?**
SHAP attributions for single predictions, plus B5 reliability tests: the top-1
feature never flips under controlled text edits, the bootstrap top-5 feature set
has Jaccard = 1.0, and SHAP rankings agree with permutation importance at
Kendall tau = 0.66.

**Why is XGBoost not in the title?**
The model is the means. The title names the properties the system must have
(calibrated, explainable, black-box, cross-domain). The classifier choice is
described and ablated in the paper.

**Why "black-box" when your system still needs a reference context?**
Black-box refers to the generating model, not to the evidence. We never open the
LLM. The reference context is simply the material a user already has, and the
system works without it too, with a weaker grounding signal.

**Is "risk" the same as "hallucination probability"?**
In this paper, yes: the model estimates the probability that the answer is
hallucinated under the given evidence. The word "risk" is kept because the
deployment use is triage, not proof.

**Why "responses" in the plural?**
The same system scores QA answers, summaries, and chat replies. The deployed
chat layer even scores individual claims inside one response. The plural marks
that scope.

**Why is the order "Cross-Domain Calibrated and Explainable"?**
The evaluation story moves in that direction. First the model is tested outside
its training domain, then the calibration problem under shift is measured and
fixed, then the explanations are validated. The title order follows the
evidence order.

**Can every word be checked against a file?**
Yes. The artifact map below lists the file that backs each word.

## 5. Numbers behind the title

| Title word | Number | Source |
|---|---|---|
| Cross-Domain | Zero-shot F1 = 0.3022 (RAGTruth QA); FaithBench F1 around 0.81 | `artifacts/results/b3/b3_dataset_metrics.json` |
| Calibrated | ECE 0.0045 in-domain; 0.8185 to 0.1335 after target recalibration | `artifacts/results/b4/b4_calibration_metrics.json`, `b4_target_calibration.json` |
| Explainable | Top-1 flip rate 0.0; top-5 Jaccard 1.0; Kendall tau 0.6625 | `artifacts/results/b5/b5_stability_bootstrap.json`, `b5_feature_importance.json` |
| Hallucination Risk | Target `P(hallucinated \| q, c, a)`; grouped 70/15/15 split, seeds 42/123/456 | abstract; `artifacts/split_indices.json` |
| Black-Box | 26 features, 7 groups, CPU inference, 61.8 ms per analysis | `artifacts/models/feature_names.json`; efficiency artifacts |
| LLM Responses | HaluEval QA (20,000 rows) plus RAGTruth and FaithBench subsets | `docs/01-project-and-methods.md` §2; `manifest.frozen.json` |

## 6. Artifact map

| Word | Primary artifact | Supporting doc |
|---|---|---|
| HaluRISC | `artifacts/models/xgboost_seed_42.joblib` and the API | `docs/01` §1, §5 |
| Cross-Domain | `artifacts/results/b3/` | `docs/02` B3 section |
| Calibrated | `artifacts/results/b4/` | `docs/02` B4 section |
| Explainable | `artifacts/results/b5/` | `b5-explanation-reliability.md` |
| Risk, Estimation | `artifacts/results/b2/` metric tables | `docs/02` B2 section |
| Black-Box, Responses | `src/features/extract_features.py`, `src/api/main.py` | `docs/01` §5, `b7-research-ui.md` |
| Everything | `docs/manifest.frozen.json` | `docs/03-reproduction-and-defense.md` |

## 7. The 30-second spoken answer

"The title has four commitments. HaluRISC is the system name, short for
Hallucination Risk Scoring and Calibration. It estimates a hallucination
probability from a question, evidence, and an answer, without touching the LLM
itself, so it is a black-box method. The evaluation leaves the training domain
on purpose, which is the cross-domain part, and it finds that calibration breaks
under shift. The system stays useful because the probabilities are recalibrated
and because every score comes with a stable SHAP explanation."

## 8. Journal title vs report title

The course report uses a second title:

> **HaluRISC: A Calibrated and Explainable Machine Learning Framework for
> Hallucination Risk Prediction in Black-Box LLM Outputs**

Both titles describe the same system. The differences are deliberate.

| Journal wording | Report wording | Why they differ |
|---|---|---|
| Cross-Domain Calibrated and Explainable | A Calibrated and Explainable ... Framework | The journal title leads with the strongest new evidence, the cross-domain test. The report title leads with the artifact, a framework, which matches a course deliverable that includes code, API, and dashboard. |
| Risk Estimation | Risk Prediction | Estimation is the stricter statistical word, since the output is a calibrated probability. Prediction is the more common word in a course setting. Both are true, and the paper text always uses the probability framing. |
| LLM Responses | LLM Outputs | Responses is the paper's word for scored answer candidates, including chat replies and summaries. Outputs is the broader term used on the report cover. |

If the supervisor asks which title is official, the answer is: the journal
manuscript title is the camera-ready one, and the report title is the course
cover. The system name, the two properties (calibrated, explainable), and the
black-box constraint are identical in both.

## Related documents

- [`01-project-and-methods.md`](01-project-and-methods.md) - what the system is and how it works
- [`02-experiments-and-results.md`](02-experiments-and-results.md) - the numbers cited above with artifact paths
- [`03-reproduction-and-defense.md`](03-reproduction-and-defense.md) - defense Q&A and the demo script
- [`04-interface-guide.md`](04-interface-guide.md) - what each screen of the demo means
