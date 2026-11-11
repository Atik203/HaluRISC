# P0 review quick start (what to check, how, examples)

This is the one-page entry point for the three-author review. The full instructions,
with worked examples for every sheet, are in
[`docs/rebuild/p0/P0_HUMAN_REVIEW_GUIDE.md`](p0/P0_HUMAN_REVIEW_GUIDE.md).
The category definitions are in
[`docs/rebuild/p0/codebook_v2.md`](p0/codebook_v2.md).

**Integrity rule.** The current labels are AI pre-annotations. The paper must say
"three-author blinded review with AI pre-annotation." Never write "independent human
annotation," "two human annotators," or "human-labeled" until all three authors have
completed the blind overlap and the main sheets.

---

## 1. What is in the package

| File | Rows | What it is | What the authors do |
|---|---:|---|---|
| `docs/rebuild/p0/p0_error_taxonomy.csv` | 50 | 25 benchmark-negative rows (8 threshold false positives + 17 near-threshold negatives) + 25 false negatives from the m3 HaluEval test errors | Confirm the error category, or correct it |
| `docs/rebuild/p0/p0_explanation_audit.csv` | 100 | 50 error-enriched + 50 length-stratified rows with m3 top-5 SHAP | Judge whether the explanation is plausible |
| `docs/rebuild/p0/p0_claim_verdicts.csv` | 100 | 100 claims from 50 answers, each with the evidence sentence the NLI step selected | Judge supported / contradicted / unsupported / unsure |
| `docs/rebuild/p0/p0_blind_overlap.csv` | 60 | 20 items from each sheet, no AI labels | Each author labels all 60 independently |
| `docs/rebuild/p0/annotation_manifest.json` | --- | Counts, hashes, model, prompt version | Do not edit |
| `artifacts/results/rebuild/p0/author_review_summary.json` | --- | Generated after the review | Do not edit by hand |

Every main-sheet row starts at `review_status = needs_author_verification`.
Change it to `author_verified` or `author_corrected` only after you have read the full
case.

---

## 2. The six-step workflow

1. **Back up the package.** `Copy-Item docs/rebuild/p0 docs/rebuild/p0_backup -Recurse`.
2. **Read the codebook.** At minimum §2 (20 categories), §3 (support test), §4
   (plausibility), §5 (claim verdicts), and §6 (confidence).
3. **Label the three main sheets.** Use the full text command in §4 below. Fill
   `author_label`, `annotator_id`, `changed`, `review_status`, and a one-sentence
   `notes` for any disagreement. Leave `final_label` blank until adjudication.
4. **Run the blind overlap.** Make three copies of `p0_blind_overlap.csv`, one per
   author. Each author fills only `author_label`, `annotator_id`, and `notes`. Do not
   look at the main sheets or another author's copy until all three are locked.
5. **Merge and compute kappa.** Use the command in §5. Report Fleiss' kappa separately
   for taxonomy, explanation, and claims. Do not pool the three label sets.
6. **Adjudicate and ingest.** Resolve disagreements, fill `final_label`, set
   `review_status = adjudicated`, then run
   `python src/models/p0_ingest_author_review.py`. It writes
   `author_review_report.md` and `author_review_snippet.tex`.

---

## 3. What / how / example for each sheet

### 3.1 Error taxonomy

**What.** Is the answer supported by the context, and if not, which single category
best explains the defect? The benchmark label is a cross-check, not ground truth.

**How.** Read question, context, and answer. If the answer is grounded, use an
`artifact_*` category. If it is not grounded, use the most specific `content` category.
Do not change `label`. Put a second independent defect in `secondary_category`.

**Example (easy).** `P0E39`: the context says the episode aired on AMC; the answer says
"Syfy." Primary `entity_substitution`, secondary `context_contradiction`, confidence 5.
The benchmark label 1 is correct.

**Example (ambiguous).** `P0E41`: the answer says "Goldcrest Films also produced it."
If "it" is *You and I*, the answer is supported; if "it" is the 1984 film, the context
is silent. Primary `coreference_error`, confidence 2, note the reading.

**Example (label noise).** `P0E44`: the answer "Tyler Winklevoss founded ConnectU" is
fully supported, but the benchmark label is 1. Primary `artifact_label_noise`,
confidence 5, note the disagreement. Do not change `label`.

### 3.2 Explanation audit

**What.** Does the top-5 signed SHAP explain the score with evidence-bearing terms, or
with lexical shortcuts? Judge the explanation, not the answer.

**How.** Mark `plausible` when the dominant terms are context overlap, contradiction,
entailment, entity overlap, number novelty, or claim NLI and agree with the text. Mark
`implausible` when the dominant terms are question similarity, answer length, or
character count, or when the score and the evidence disagree. Mark `unsure` when the
score is within about 0.05 of 0.5 or the terms are mixed.

**Example (easy).** `P0X001`: the answer is verbatim in the context; the largest term is
`overlap_answer_context = 1` at -3.41 and the positive terms are small. `plausible`,
confidence 4.

**Example (ambiguous).** `P0X021`: the answer is verbatim, the score is 0.515, and
context overlap supports it while question overlap, length, cosine, and a small
contradiction push it over. `unsure`, confidence 3.

**Example (label noise).** `P0X069`: "retired athletes" is true of both former tennis
players, but zero context overlap, cosine, and contradiction flag it at 0.967. The
explanation is `implausible`, and the parent label is suspect. Record both.

### 3.3 Claim verdicts

**What.** Does the evidence sentence shown in the row entail the claim, contradict it,
or stay silent? Do not use world knowledge or the rest of the context to rescue an
off-topic evidence sentence.

**How.** `supported` = the evidence entails the claim, including faithful paraphrase and
matching numbers. `contradicted` = the evidence states the opposite or a conflicting
value. `unsupported` = the evidence is relevant but silent on the asserted relation.
`unsure` = the evidence is a fragment, off-topic, or ambiguous.

**Example (easy).** `P0C003`: "Essential amino acids ... must be obtained from the diet"
against evidence that says exactly that. `supported`, confidence 5.

**Example (ambiguous).** `P0C033`: "the normal value of BNP is not explicitly mentioned"
against "only a low amount of BNP is found in your blood." A strict reader says
`supported`; a reader who treats "low amount" as a normal value says `contradicted`.
Record your reading.

**Example (label noise).** `P0C037`: the claim repeats the Shugabush/Furcorn combination
in the evidence. The claim is `supported` even though the parent answer is labeled 1.
Do not import the parent label into `author_label`.

---

## 4. Print the full case

Taxonomy and explanation rows (HaluEval text):

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/qa_clean.parquet').set_index('sample_id').loc[sys.argv[1]]; print('Q:',r['question']); print(); print('C:',r['context']); print(); print('A:',r['answer'])" q_3061_correct
```

Claim rows (RAGTruth QA text):

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/unified_records.parquet').set_index('sample_id').loc[sys.argv[1]]; print('Q:',r['question']); print(); print('C:',r['context']); print(); print('A:',r['answer'])" ragtruth:12445
```

Blind item: open `p0_blind_overlap.csv`, filter to the `item_id`, and read the
`prompt`, `answer_or_claim`, and `evidence` columns. The blind sheet deliberately has
no AI label and no model score.

---

## 5. Blind overlap and Fleiss' kappa

Make three copies and label them independently:

```powershell
Copy-Item docs/rebuild/p0/p0_blind_overlap.csv docs/rebuild/p0/p0_blind_overlap_author1.csv
Copy-Item docs/rebuild/p0/p0_blind_overlap.csv docs/rebuild/p0/p0_blind_overlap_author2.csv
Copy-Item docs/rebuild/p0/p0_blind_overlap.csv docs/rebuild/p0/p0_blind_overlap_author3.csv
```

Merge the three copies into one 60-row file with `a1`, `a2`, `a3`:

```powershell
& .venv\Scripts\python.exe -c "import pandas as pd; a=pd.read_csv('docs/rebuild/p0/p0_blind_overlap_author1.csv').set_index('item_id'); b=pd.read_csv('docs/rebuild/p0/p0_blind_overlap_author2.csv').set_index('item_id'); c=pd.read_csv('docs/rebuild/p0/p0_blind_overlap_author3.csv').set_index('item_id'); m=a[['sheet','author_label']].rename(columns={'author_label':'a1'}).join(b[['author_label']].rename(columns={'author_label':'a2'})).join(c[['author_label']].rename(columns={'author_label':'a3'})); print(m.shape); print(m.isna().sum().to_dict()); m.to_csv('docs/rebuild/p0/p0_blind_overlap_merged.csv')"
```

Expected `(60, 4)` and zero missing values. Then run the kappa script in
`P0_HUMAN_REVIEW_GUIDE.md` §7.2. Report kappa separately for taxonomy, explanation, and
claims. For taxonomy, report exact 20-category agreement beside the collapsed
content-vs-artifact kappa, because 20 categories on 20 items is sparse.

**Interpretation.** Landis and Koch: below 0.00 poor, 0.00-0.20 slight, 0.21-0.40 fair,
0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect. A kappa below 0.40
on a sheet means the codebook or the items need adjudication before the paper reports
that sheet.

**Adjudication rule.** If all three agree, the label is final. If two agree and the
third dissents, the majority is final unless the dissent identifies a codebook
ambiguity; then discuss and record the rule. If all three differ, discuss the item and
record the majority or `unsure`. Never average labels.

---

## 6. Ingest and update the paper

After the sheets are complete and the blind copies are merged:

```powershell
& .venv\Scripts\python.exe src\models\p0_ingest_author_review.py
```

The script reads `p0_blind_overlap_merged.csv` when present; run the merge command in §5 first. It exits non-zero until all three raters and a non-null Fleiss kappa are present.

This writes:

- `artifacts/results/rebuild/p0/author_review_summary.json`
- `artifacts/results/rebuild/p0/author_review_report.md`
- `artifacts/results/rebuild/p0/author_review_snippet.tex`

It compares the AI pre-label with the authors' label, reports the correction rate and
Cohen's kappa per sheet, and computes Fleiss' kappa from the merged 60-item overlap.
It exits non-zero if any row is still `needs_author_verification`; use
`--allow-incomplete` only for a draft.

Copy the final agreement and kappa values into the manuscript's audit paragraph. The
exact wording is:

> Three authors independently reviewed the 60-item overlap and the 250-row
> pre-annotation package. Agreement was ... (Fleiss' kappa ...). The AI pre-labels
> agreed with the final labels on ... of ... rows (...%), and the authors corrected
> ... rows.

Do not report the AI pre-label distribution as a human result.

---

## 7. Checklist and common mistakes

- [ ] Every main-sheet row has `review_status` changed from `needs_author_verification`.
- [ ] Every main-sheet row has `annotator_id`.
- [ ] `final_label` is filled only after adjudication.
- [ ] The blind sheet has no AI labels or model scores.
- [ ] The three blind copies were locked before the main sheets were revealed.
- [ ] Kappa is computed separately per sheet, not pooled.
- [ ] The taxonomy kappa is reported with the exact-agreement percentage.
- [ ] The paper says "three-author blinded review with AI pre-annotation."
- [ ] The paper does not say "independent human annotation."
- [ ] The package hash in `annotation_manifest.json` still matches the files.

Common mistakes:

1. **Labeling the explanation by the model's decision.** A correct decision for a
   lexical reason is not a plausible explanation.
2. **Using world knowledge on the claim sheet.** Judge the evidence sentence, not the
   truth of the claim in the world.
3. **Changing `label` on the taxonomy sheet.** The benchmark label is fixed; record
   disagreements in `notes` and use `artifact_label_noise`.
4. **Pooling the three kappas.** The label sets are different; pool nothing.
5. **Reporting the AI pre-label counts as human counts.** They are pre-labels.
