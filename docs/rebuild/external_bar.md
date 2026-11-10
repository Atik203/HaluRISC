# External bar for HaluEval QA (WP7)

Status: complete, 2026-11-11, branch `rebuild`.

Files:

- `docs/rebuild/external_bar.md` (this note)
- `artifacts/results/rebuild/external_bar.csv` (15 verified external numbers)
- `artifacts/results/rebuild/halueval_lexical_audit.csv` (length and overlap audit of the test split)
- `artifacts/results/rebuild/length_control_fixed_hp.json` (length control on the rebuilt m3)

Supporting scripts: `src/models/rebuild_lexical_audit.py` and `src/models/rebuild_length_control.py`.

## Bottom line

No published HaluEval QA number is protocol-matched to this paper. HaluEval has no official train/test split, and each of the four papers below samples its own rows. None uses our grouped 3,000-row test split. The best available numbers are partial matches. A zero-shot CPU-only ensemble reports F1 0.792 and AUROC 0.873 on 2,000 instances. A knowledge-grounded judge reports 80.30% average accuracy on 1,000 balanced pairs when it is given the gold passage. The benchmark's own zero-shot judges report 62.59% (ChatGPT) and 69.78% (Claude 2) accuracy without the gold passage.

These numbers are context, not a head-to-head bar. The manuscript should not claim an external win, and it does not need one. Its claims are about calibration, protocol, and transfer, not about beating a leaderboard.

## Comparability rubric

- **yes**: same test rows, same label definition, same task, same metric definition, and comparable model access.
- **partial**: same benchmark, task, and labels, with at least one shared metric, but at least one major protocol difference (split, supervision, evidence access, or metric and operating point).
- **no**: a different information condition or no shared metric, so the number cannot be read as an estimate on our target task.

No paper reaches **yes**.

Two properties are shared across the four studies. First, all of them use HaluEval's construction label: each candidate is either the original correct answer or the ChatGPT-generated hallucinated counterpart. The label definition is therefore the same. Second, all of them are black-box with respect to the generating model. None uses the generating model's internal states. The differences that block a head-to-head comparison are the split, the supervision, the evidence access, and the metric.

## This paper's anchor

| Item | Value | Source |
|---|---|---|
| Dataset | HaluEval QA, 20,000 rows = 10,000 answer pairs, 50/50 | `artifacts/results/dataset_mapping_report.json` |
| Split | Grouped 70/15/15 by `item_idx`; test 3,000 rows = 1,500 pairs; 0 groups span splits | `artifacts/split_integrity_report.json` |
| Detector | EC-XGB (m3), 35 features: 26 base + 8 claim-level NLI aggregates + source indicator; XGBoost; seeds 42, 123, 456 | `artifacts/results/rebuild/fixed_hp/fixed_hp_ablation.csv` |
| F1 | 0.9854 +/- 0.0002 | same |
| AUROC | 0.9986 +/- 0.0001 | same |
| MCC | 0.9713 +/- 0.0004 | same |
| ECE / Brier | 0.0078 / 0.0107 | same |
| Fixed-hp variants | m0 0.98488, m1 0.98533, m2 0.98491, m3 0.98543 (span 0.00055) | same |
| Conventional baselines | Logistic regression F1 0.9660, random forest 0.9842 | `artifacts/results/model_comparison.csv` |
| Length-only baseline | F1 0.9475, AUROC 0.9758 on a 1,482-row length-stratified subset; F1 0.4762, AUROC 0.5209 on a 158-row length-matched subset | `artifacts/results/rebuild/length_control_fixed_hp.json` |
| EC-XGB on the same subsets | F1 0.9867, AUROC 0.9988 on the stratified subset; F1 0.9071, AUROC 0.9555 on the matched subset | same |

The fixed-hp variants matter for the external bar. Under one configuration, m0 to m3 span 0.0006 F1. The conventional baselines are also close. The top of the HaluEval QA range does not discriminate between model classes. That is internal evidence for saturation, independent of the external numbers.

## Search and verification

Sources searched: OpenAlex, the arXiv API, ACL Anthology, Crossref, and publisher PDFs. Every value in `external_bar.csv` was read from the paper's own table, not from a secondary summary. DOIs and arXiv IDs are in the references.

Screened out:

- GraphEval (arXiv:2407.10793): SummEval and QAGS only, no HaluEval numbers.
- RelD, Chen et al. (CIKM 2023, DOI 10.1145/3583780.3614905): RelQA, not HaluEval.
- "Hallucination Detection and Confidence Calibration for Large Language Model Outputs: Reproducible Experiments on HaluEval" (AIMLR 2025, DOI 10.69987/AIMLR.2025.60401): the accessible abstract reports only an aggregate over four HaluEval domains (AUROC 0.835, F1 0.751, ECE 0.009), not a QA-only number, so it cannot support a QA comparison.
- G2LDetect (AAAI 2025, DOI 10.1609/aaai.v39i1.31985): the AAAI PDF was behind a bot challenge. I could not check whether it reports HaluEval QA numbers, so it is not cited.

Access limits: Tavily search hit its plan quota, the Semantic Scholar API returned 429, and Crossref `select` queries returned 400. The values were still verified against ACL Anthology, arXiv, and publisher PDFs.

## Verified numbers: context table, not a ranking

| Paper | Setting | Test set | Metric | Value | Comparability |
|---|---|---|---|---|---|
| Li et al., EMNLP 2023 (HaluEval), no passage | Zero-shot LLM judge | 10,000 items | Accuracy | 62.59 (ChatGPT), 69.78 (Claude 2) | no |
| Li et al., EMNLP 2023 (HaluEval), retrieved knowledge | Zero-shot LLM judge with retrieval | 10,000 items | Accuracy | 76.83 (ChatGPT) | partial |
| Zhang et al., arXiv 2024 (KnowHalu) | Zero-shot knowledge-grounded judge | 1,000 pairs (2,000 rows) | Average accuracy | 80.30 (gold passage), 72.05 (Wikipedia retrieval) | partial |
| Cheng et al., EMNLP 2024 (HaluAgent) | Fine-tuned 13B agent with tools | 100 items | Accuracy and F1 | 71.00 and 72.38 | partial |
| Faujdar and Kadvani, GroundLM 2026 | Zero-shot CPU-only features | 2,000 instances | F1 and AUROC | 0.792 and 0.873 | partial, closest |

This is a protocol summary, not a head-to-head table. The rows use different test sets and different metrics.

## Paper-by-paper comparability

### Li et al. 2023: HaluEval benchmark baselines

- **Dataset and split.** HaluEval QA has 10,000 source questions, each with a correct answer and a ChatGPT-generated hallucinated answer. The paper randomly selects one of the two answers per question for classification, giving 10,000 balanced items. There is no official train/test split. These are the benchmark's evaluation items.
- **Task and evidence.** The QA prompt contains the question and the candidate answer only. It does not contain the HotpotQA passage. The judge is instructed to use world knowledge and the benchmark's hallucination types.
- **Models and setting.** Eleven LLMs, zero-shot, no fine-tuning, temperature 0.
- **Numbers.** QA accuracy ranges from 39.66 (Falcon) to 69.78 (Claude 2). ChatGPT is 62.59, GPT-3 is 49.21, and Alpaca is 6.68, far below chance (Table 5). Adding retrieved Wikipedia knowledge raises ChatGPT from 62.59 to 76.83 (Section 3.2.2).
- **Comparability: no.** The information condition differs because there is no gold passage, the metric is accuracy only, and the sample is not ours. This is the benchmark's own baseline, not a detector comparison.
- **Use in the paper.** Cite it to establish that zero-shot recognition on HaluEval QA is far from solved and that adding evidence helps. Do not set our F1 against 62.59 as if they were the same task.

### Zhang et al. 2024: KnowHalu

- **Dataset and split.** 1,000 QA pairs sampled from HaluEval, giving 2,000 balanced rows. A separate validation subset, described as 100 cases, is used for the aggregation thresholds.
- **Task and evidence.** Binary detection with either the gold HaluEval passage or Wikipedia retrieval. The judge may return INCONCLUSIVE, and the reported average accuracy does not credit abstentions.
- **Models and setting.** GPT-3.5-turbo-1106 and Starling-7B, zero-shot, no fine-tuning, with step-wise query decomposition, retrieval, and multi-form knowledge.
- **Numbers.** With the gold passage, GPT-3.5 with the HaluEval knowledge prompt reaches 56.90 average accuracy, and KnowHalu aggregation reaches 80.30. With Wikipedia retrieval, the same pair is 53.05 and 72.05. GPT-4 zero-shot CoT is 65.05 without the gold passage (Table 2).
- **Comparability: partial.** This is the closest evidence condition (question, answer, gold passage) and the same benchmark labels. But the test rows are not ours, the metric is accuracy with abstention rather than F1 or AUROC, and the detector is a zero-shot judge. The 80.30 number is the gold-evidence condition, not a like-for-like detector score.
- **Venue caveat.** I found no peer-reviewed version. arXiv v1, submitted 3 April 2024, is the primary source. A Crossref title search returns no published version. Treat it as a preprint.

### Cheng et al. 2024: HaluAgent

- **Dataset and split.** 900 HaluEval-QA training examples and 100 test examples. 808 filtered detection trajectories from HaluEval-QA are mixed with 1,209 from other tasks to fine-tune the agent.
- **Task and evidence.** Multi-task hallucination detection with sentence segmentation, tool selection (including web search), verification, and reflection. The evidence source is the agent's tools, not necessarily the gold passage.
- **Models and setting.** Baichuan2-Chat 7B and 13B, supervised fine-tuning on 2,017 trajectories. The run count and threshold are unreported.
- **Numbers.** On HaluEval-QA, HaluAgent 13B reaches 71.00 accuracy and 72.38 F1. The 7B model reaches 67.00 and 67.33. A GPT-4 pipeline reaches 77.53 and 75.61 on the same 100 items (Table 3).
- **Comparability: partial.** Same benchmark, task, labels, and a supervised setting, which is closer to ours than the zero-shot judges. But the test is 100 items, where a 95% binomial interval on 71% is roughly +/- 9 points. The model class is a fine-tuned 7B or 13B agent rather than a 35-feature tabular model, the threshold is unreported, and the test items are not ours.
- **Use in the paper.** It shows that a supervised system trained in-domain on HaluEval-QA lands in the low 70s on a small held-out sample, far below our 0.9854. The gap is a warning about split construction and test size, not evidence that EC-XGB is better.

### Faujdar and Kadvani 2026: lightweight HaluEval benchmark

- **Dataset and split.** 3,000 of the 10,000 HaluEval QA samples in their snapshot, drawn with seed 42, with 1,000 for validation and 2,000 for test. Their unit is a labelled candidate, and the paper does not give the mapping from those samples to the 20,000-row official release. The test prevalence is therefore not stated.
- **Task and evidence.** Binary detection with the source passage plus context as the NLI premise and the candidate as the hypothesis.
- **Models and setting.** ROUGE-L, semantic similarity, BERTScore, a FEVER-trained DeBERTa-v3 NLI model, and a score-level similarity-NLI ensemble. No fine-tuning on HaluEval. The threshold is chosen on validation by the Youden index, with 2,000 bootstrap resamples.
- **Numbers.** QA ensemble: accuracy 0.803, precision 0.821, recall 0.766, F1 0.792, AUROC 0.873. NLI alone: accuracy 0.768, precision 0.865, recall 0.625, F1 0.725, AUROC 0.795 (Table 2).
- **Comparability: partial, and the closest match.** Same benchmark, task, labels, and metrics (F1 and AUROC). It is context-grounded and uses CPU-only features, like ours. But the sample is different, the method is zero-shot with a validation-tuned threshold, and our model is supervised on 14,000 rows. The AUROC comparison is the cleanest because it is threshold-independent.
- **Interpretation.** The 0.792 F1 is not a ceiling for HaluEval QA and not a floor for our method. It is the best number from four fixed, unfitted detectors. Our supervised model has a much larger hypothesis class and a different split. The gap is expected and is not a methodological win.

## Why no head-to-head table is possible

1. **No official split.** HaluEval ships 10,000 QA pairs without a train/test partition. Li et al. evaluate all 10,000. KnowHalu samples 1,000 pairs. HaluAgent uses 900 and 100. Faujdar and Kadvani sample 3,000 and split 1,000 and 2,000. Our split is a grouped 14,000/3,000/3,000 row split over the 20,000 rows. The splits are independently constructed, so there is no common evaluation set. The external papers also describe their sampling unit differently, sometimes as a sample and sometimes as a row, so even the count of 10,000 is not a common unit.
2. **Different supervision.** Three of the four studies are zero-shot or judge-based. HaluAgent is supervised, but it uses a 7B or 13B agent and a 100-item test. Ours is a 35-feature XGBoost trained on 14,000 rows. A gap between these settings is not a method comparison.
3. **Different evidence access.** Li et al. give the judge no passage. KnowHalu gives the gold passage or Wikipedia retrieval. Faujdar and Kadvani give the source and context. HaluAgent uses tools and web search. These are different tasks that share a label name.
4. **Different metrics and operating points.** Accuracy, average accuracy with abstention, F1 at 0.5, F1 at a Youden threshold, and AUROC are not interchangeable. AUROC is the only threshold-independent common metric, and only Faujdar and Kadvani report it.
5. **A construction artifact.** HaluEval's QA generation instruction asks for hallucinated answers about five words longer than the correct answer. In our test split, the mean answer length is 2.28 words for faithful answers and 10.79 for hallucinated answers, and a length-only logistic regression reaches AUROC 0.974. Any detector with a length feature, or any lexical metric that is not length-normalized, will exploit this. The external numbers cannot be read as semantic-detection comparisons.

## Benchmark baselines, as the fallback

Because no comparable detector number exists, the manuscript should cite the benchmark's own baselines:

- QA accuracy across 11 LLMs: 39.66 to 69.78. ChatGPT 62.59, Claude 2 69.78, GPT-3 49.21, InstructGPT davinci-002 60.05, Llama 2 49.60, Alpaca 6.68.
- Adding retrieved Wikipedia knowledge raises ChatGPT from 62.59 to 76.83.
- These are zero-shot, no fine-tuning, and accuracy only. The vanilla prompt has no gold passage.

This is the honest external anchor. The benchmark's own evaluation shows that HaluEval QA is hard for zero-shot judges. Our in-domain supervised result is much higher on its own split, which is expected from supervision and does not settle the semantic question. The two facts are compatible and do not need a head-to-head table.

## The length and overlap audit

Test split, 3,000 rows, 1,500 per label:

| Statistic | Faithful (label 0) | Hallucinated (label 1) |
|---|---|---|
| Mean words | 2.28 | 10.79 |
| Median words | 2 | 9 |
| Mean characters | 14.0 | 65.0 |
| Mean answer-token coverage by context (`overlap_answer_context`) | 0.949 | 0.657 |
| Mean Jaccard (`jaccard_ans_ctx`) | 0.056 | 0.161 |

AUROC of single features on the test split, in the risk direction:

| Feature | AUROC | Reading |
|---|---|---|
| `overlap_answer_context` | 0.0868 | Lower coverage means hallucination |
| `1 - overlap_answer_context` | 0.9132 | The paper's heuristic direction |
| `jaccard_ans_ctx` | 0.8545 | Higher Jaccard means hallucination |
| `n_words` | 0.9740 | Longer answers mean hallucination |

The two overlap features point in opposite directions because they normalize differently. Answer-token coverage divides by answer length. Jaccard divides by the union, which is dominated by the long context. A longer hallucinated answer adds context words in absolute terms, but a smaller fraction of its own tokens comes from the context. This is why Faujdar and Kadvani can correctly report that higher overlap indicates hallucination on QA, while our heuristic correctly reports that lower answer coverage indicates hallucination. The sign is a property of the metric, not of hallucination.

The length-only baseline is strong on the full test distribution. A logistic regression on standardized log length, trained on the train split, reaches F1 0.9475 and AUROC 0.9758 on a 1,482-row length-stratified subset. On a 158-row length-matched subset with 79 positives and 79 negatives it collapses to F1 0.4762 and AUROC 0.5209, while EC-XGB retains F1 0.9071 and AUROC 0.9555. EC-XGB is therefore not merely a length detector, but length explains most of the headline in-domain number. This is the paper's claim 1, and the external bar should not obscure it.

HaluEval's own generation instruction is one cause: "#Hallucinated Answer# can only have about 5 more words than #Right Answer#." The observed gap of 8.5 words is larger than the instruction, but the direction is the same. Janiak et al. (EMNLP 2025) independently find that hallucinated responses tend to be longer and that length-based heuristics rival complex detectors. They do not report a HaluEval QA detector number, so they are supporting evidence, not a row in the CSV.

Reproduce the audit with `python src/models/rebuild_lexical_audit.py` and `python src/models/rebuild_length_control.py` from the repo root.

## Manuscript-ready paragraph

> Published HaluEval QA results do not provide a protocol-matched external bar. Zero-shot judges report 62.6% (ChatGPT) and 69.8% (Claude 2) accuracy without the gold passage, a knowledge-grounded judge reaches 80.3% with the gold passage and 72.1% with Wikipedia retrieval on 1,000 balanced pairs, and the closest CPU-only benchmark reports F1 0.792 and AUROC 0.873 on 2,000 held-out instances. Each study defines its own split, and none uses the grouped 3,000-row test split described here. The gap between those numbers and the in-domain F1 of 0.9854 is confounded by supervision, evidence access, and benchmark construction. It is not evidence of better semantic hallucination detection. HaluEval's QA generation instruction asks for hallucinated answers about five words longer than the correct answer, and a length-only baseline reaches F1 0.948 on a length-stratified subset of the test split, against 0.987 for EC-XGB on the same rows. The external numbers are reported as context, with no head-to-head claim.

That paragraph is about 150 words. It states the negative comparability finding, gives the external range, and keeps the claim to what the evidence supports.

## What could not be verified

- No peer-reviewed version of KnowHalu. The arXiv v1 is the primary source, and a Crossref title search returns no published record.
- HaluEval has no official split and no released per-example predictions, so its accuracy cannot be converted to F1 or AUROC and its rows cannot be mapped to ours.
- The AIMLR 2025 calibration paper was excluded because only an aggregate over four domains was available, not a QA-specific number.
- G2LDetect (AAAI 2025) could not be read because the AAAI PDF was behind a bot challenge.
- Tavily, Semantic Scholar, and Crossref `select` queries were limited as noted above. The numbers were verified through ACL Anthology, arXiv, and publisher PDFs instead.

## References

1. Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., and Wen, J.-R. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. EMNLP 2023, 6449-6464. DOI: 10.18653/v1/2023.emnlp-main.397. arXiv:2305.11747.
2. Zhang, J., Xu, C., Gai, Y., Lecue, F., Song, D., and Li, B. (2024). KnowHalu: Hallucination Detection via Multi-Form Knowledge Based Factual Checking. arXiv:2404.02935. DOI: 10.48550/arXiv.2404.02935. Preprint.
3. Cheng, X., Li, J., Zhao, W. X., Zhang, H., Zhang, F., Zhang, D., Gai, K., and Wen, J.-R. (2024). Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector. EMNLP 2024, 14600-14615. DOI: 10.18653/v1/2024.emnlp-main.809.
4. Faujdar, K., and Kadvani, S. (2026). How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation. GroundLM 2026, 367-377. DOI: 10.18653/2026.groundlm-1.29.
5. Janiak, D., Binkowski, J., Sawczyn, A., Gabrys, B., Shwartz-Ziv, R., and Kajdanowicz, T. (2025). The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs. EMNLP 2025, 34728-34745. DOI: 10.18653/v1/2025.emnlp-main.1761.
