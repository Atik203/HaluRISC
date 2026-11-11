# P0 Author Review Guide

**File name:** `P0_HUMAN_REVIEW_GUIDE.md` (the name is fixed by the package layout; the
document itself is an author review guide).

**Purpose:** turn the AI pre-annotation in `docs/rebuild/p0/` into verified labels
without losing the audit trail. Every row starts at
`review_status = needs_author_verification`. The package is not a completed annotation
study until the three authors finish the steps below.

**Read first:** `codebook_v2.md` Sections 3, 4, 5, and 6. The codebook is the source of
truth for every decision rule. This guide is the practical procedure.

---

## 0. What is in the package

| File | Rows | What it holds | AI label set |
|---|---:|---|---|
| `p0_error_taxonomy.csv` | 50 | 25 benchmark-negative and 25 benchmark-positive HaluEval cases scored by m3 seed 42 | 20 category codes |
| `p0_explanation_audit.csv` | 100 | 50 error-enriched cases (26 from the B5 audit set plus 24 additional m3 errors) and 50 length-stratified random HaluEval test cases | `plausible`, `implausible`, `unsure` |
| `p0_claim_verdicts.csv` | 100 | 100 atomic claims from 50 RAGTruth QA test answers (25 grounded, 25 hallucinated parent answers) | `supported`, `contradicted`, `unsupported`, `unsure` |
| `p0_blind_overlap.csv` | 60 | 20 taxonomy, 20 explanation, and 20 claim items with no AI labels | blank until each author fills a copy |
| `annotation_manifest.json` | — | counts, seed, model, prompt version, codebook hash, status counts | — |

The three annotation sheets hold 250 rows in total, all at
`needs_author_verification`. The blind sheet is separate and does not carry a review
status.

### 0.1 What the current AI labels mean

- Taxonomy: `ai_suggested` is the AI category. `label` is the benchmark 0/1.
- Explanation: `ai_suggested` is the AI plausibility judgement. `raw_score` is the m3
  probability. `top_features` and `top_shap` are exact TreeSHAP values in raw margin
  (log-odds), top five by absolute value.
- Claims: `ai_suggested` is the AI claim verdict. `evidence_text` is the sentence the
  deterministic NLI step selected.

The AI labels are a starting point, not a prior to defend. Change them when the text
disagrees. Record why in `notes`.

---

## 1. Before you start

- [ ] Work on branch `rebuild`. Do not edit anything under `artifacts/`,
      `Journal_Paper/`, or `data/processed/`.
- [ ] Read `codebook_v2.md` Sections 3-6.
- [ ] Make a backup of the whole folder:
      `Copy-Item -Recurse docs/rebuild/p0 docs/rebuild/p0_backup_$(Get-Date -Format yyyyMMdd)`
- [ ] Confirm the row counts and statuses:

```powershell
& .venv\Scripts\python.exe -c "import pandas as pd; e=pd.read_csv('docs/rebuild/p0/p0_error_taxonomy.csv'); x=pd.read_csv('docs/rebuild/p0/p0_explanation_audit.csv'); c=pd.read_csv('docs/rebuild/p0/p0_claim_verdicts.csv'); b=pd.read_csv('docs/rebuild/p0/p0_blind_overlap.csv'); print(e.shape, x.shape, c.shape, b.shape); print(e['review_status'].value_counts().to_dict()); print(x['review_status'].value_counts().to_dict()); print(c['review_status'].value_counts().to_dict())"
```

Expected: `(50, 17) (100, 16) (100, 13) (60, 9)` and only
`{'needs_author_verification': ...}` for the three sheets.

- [ ] Open the CSVs in an editor that preserves UTF-8. In VS Code the default is fine.
      In Excel use **Data > From Text/CSV**, choose UTF-8 and comma, then load. Do not
      double-click the file if Excel is configured for a different locale.
- [ ] Do not add, remove, or reorder columns. Do not sort the sheet; filter by
      `case_id` or `claim_id` instead.
- [ ] Leave `label`, `ai_suggested`, `ai_rationale`, and `ai_confidence_1to5`
      untouched. They are the pre-annotation record.

### 1.1 How to edit a row

1. Find the row by `case_id` / `claim_id`.
2. Read the full case with the command in Section 2.
3. Fill `author_label` with the label for that sheet (Section 0).
4. If you disagree with the AI category or plausibility, change the relevant AI field
   in place (`primary_category`, `secondary_category`) and explain in `notes`. The
   `ai_suggested` column is the pre-annotation and stays as written.
5. Set `annotator_id` to your initials.
6. Set `changed` to `yes` if your label differs from `ai_suggested`, otherwise `no`.
7. Set `review_status` to `author_verified` if you accepted the row, or
   `author_corrected` if you changed it.
8. Save with UTF-8 encoding.

Fields that contain commas must stay quoted. Most editors do this automatically. If you
edit in a plain text editor, keep the quotes around `ai_rationale`, `notes`, and any
snippet that contains a comma.

Blank means "not filled yet". Do not write `NA`, `?`, or `-` in a categorical field.

---

## 2. Print the full case

Snippets are capped at 160 characters. Never label from a snippet alone.

### 2.1 Taxonomy and explanation rows (HaluEval text)

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/qa_clean.parquet').set_index('sample_id').loc[sys.argv[1]]; print('Q:',r['question']); print(); print('C:',r['context']); print(); print('A:',r['answer'])" q_3027_correct
```

Replace `q_3027_correct` with the row's `sample_id`.

### 2.2 Claim rows (RAGTruth QA text)

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/unified_records.parquet').set_index('sample_id').loc[sys.argv[1]]; c=pd.read_csv('docs/rebuild/p0/p0_claim_verdicts.csv').set_index('claim_id').loc[sys.argv[2]]; print('Q:',r['question']); print(); print('A:',r['answer']); print(); print('CLAIM:',c['claim_text']); print(); print('EVID:',c['evidence_text'])" ragtruth:11908 P0C001
```

Replace both arguments with the row's `sample_id` and `claim_id`.

### 2.3 A blind item

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; b=pd.read_csv('docs/rebuild/p0/p0_blind_overlap.csv').set_index('item_id').loc[sys.argv[1]]; print('SHEET:',b['sheet']); print('PROMPT:',b['prompt']); print('ANSWER/CLAIM:',b['answer_or_claim']); print('EVIDENCE:',b['evidence'])" P0E39
```

For the full text behind a blind item, use the command in 2.1 or 2.2 with the
`sample_id` shown in the blind row. For a blind claim, the `item_id` is the
`claim_id`, so the claim command works directly.

---

## 3. Sheet 1: `p0_error_taxonomy.csv`

### 3.1 What to check

For each row, read the full question, context, and answer, then answer three questions
in order:

1. Is the answer supported by the context and does it resolve the asked slot?
2. If it is not supported, which single category best explains the defect?
3. Does the benchmark `label` agree with your reading? If not, that is a finding.

The benchmark label is a cross-check. Do not change `label`. If the answer is grounded
and `label = 1`, the primary category is `artifact_label_noise`. If the answer is not
grounded and `label = 0`, the primary category is the content defect and the benchmark
label is the thing to flag in `notes`.

### 3.2 How to decide

- Grounded and flagged by the model -> artifact primary. Use
  `artifact_short_answer_fp` only when the answer is at most four words and the top
  features are length or overlap. Otherwise use `artifact_grounded_overflag`.
- Grounded and the benchmark says hallucinated -> `artifact_label_noise` primary.
- Not grounded -> content primary. Use the most specific observable defect.
- Two independent defects -> cause primary, consequence secondary. Do not use the
  secondary field for confidence.
- Cannot judge because the context is malformed -> `artifact_truncated_context`.
- Multiple defensible answer forms and no convention resolves them ->
  `artifact_question_ambiguity`.

### 3.3 Worked example: easy

`P0E39` / `q_2910_hallucinated`. Question: which cable network showed the episode?
Context: the episode "aired on AMC". Answer: `"Syfy"`. The answer names another cable
network and contradicts an explicit sentence. Primary `entity_substitution`, secondary
`context_contradiction`, confidence 5. The benchmark label 1 is correct. There is
nothing ambiguous here.

### 3.4 Worked example: ambiguous

`P0E41` / `q_5861_hallucinated`. The question asks who produced the 1984 film directed
by the same person who directed *You and I*. The context says *You and I* was "produced
by David Puttnam for his company Goldcrest Films". The answer is `"Goldcrest Films also
produced it."` Two readings are possible. If "it" is *You and I*, the answer is
supported. If "it" is the 1984 film, the context never states the producer. The question
also asks "who" while the answer names a company. Primary `coreference_error`, secondary
`artifact_question_ambiguity`, confidence 2. If you resolve "it" to *You and I*, change
the primary to `artifact_grounded_overflag` and say so in `notes`.

### 3.5 Worked example: label noise

`P0E44` / `q_2251_hallucinated`. The question asks for the third founder of ConnectU
besides Cameron Winklevoss and Divya Narendra. The context lists "Cameron Winklevoss,
Tyler Winklevoss, and Divya Narendra" and describes Tyler as "an American rower". The
answer `"Tyler Winklevoss founded ConnectU."` is fully supported. The benchmark label is
1. Primary `artifact_label_noise`, confidence 5. Do not change `label`; record the
disagreement in `notes`.

### 3.6 Columns to edit

| Column | Action |
|---|---|
| `primary_category` | Correct it for the final record if the AI category is wrong; `ai_suggested` keeps the original. |
| `secondary_category` | Add a second independent defect, or leave blank. |
| `author_label` | Your category code for the case. |
| `final_label` | Leave blank until adjudication; it is the binary label, not the category. |
| `annotator_id` | Your initials. |
| `changed` | `yes` if your category differs from `ai_suggested`, else `no`. |
| `review_status` | `author_verified` or `author_corrected`. |
| `notes` | One sentence for any disagreement or caveat. |

**Caveat for this sheet.** The 25 benchmark-negative rows are the 25 highest-scoring
`label = 0` rows. Only 8 cross the 0.5 decision threshold; the other 17 are
near-threshold cases with scores 0.272-0.493. Do not call those 17 threshold false
positives. The rationale on each row states which group it belongs to.

---

## 4. Sheet 2: `p0_explanation_audit.csv`

### 4.1 What to check

You are judging the **explanation**, not the answer. For each row:

1. Decide from the context whether the answer is grounded. Use the support test.
2. Note whether m3 flagged it (`raw_score >= 0.5`).
3. Read the top five signed terms. A positive value pushes toward hallucinated. A
   negative value pushes toward grounded.
4. Ask whether the dominant terms are evidence-bearing (context overlap, contradiction,
   entailment, entity overlap, number novelty, claim NLI) or lexical shortcuts (question
   similarity, answer length, character count).

### 4.2 How to decide

- `plausible`: the top signed terms are evidence-bearing and consistent with the score
  and the text.
- `implausible`: the top terms are lexical shortcuts, or the score and the evidence
  disagree, so the explanation misdescribes the case.
- `unsure`: the score is within about 0.05 of the threshold, the top terms are mixed, or
  the answer form is contestable.

Do not mark an explanation `plausible` merely because the model's decision is correct.
A right decision for the wrong reason is `unsure` or `implausible`.

### 4.3 Worked example: easy

`P0X001` / `q_1402_correct`. Answer: `"Walt Disney World Conference Center"`, which is
verbatim in the context. Score 0.401. The largest term is `overlap_answer_context = 1`
at `-3.41`; contradiction `+0.85` and question similarity `+0.65` are too small to flip
it. The top term is a genuine support signal. `plausible`, confidence 4.

### 4.4 Worked example: ambiguous

`P0X021` / `q_7742_correct`. Answer: `"The Patient Protection and Affordable Care Act"`,
verbatim in the context. Score 0.515, barely above threshold. Context overlap `-2.96`
supports the answer, but question overlap `+0.77`, length `+0.63`, cosine `+0.52`, and a
small contradiction `+0.47` push it over. The evidence term supports the answer while
the non-evidence terms decide the flag. `unsure`, confidence 3.

### 4.5 Worked example: label noise

`P0X069` / `q_9018_hallucinated`. Question: Jo Durie and Kimberly Po are both former
what? Answer: `"retired athletes"`. Both are former professional tennis players in the
context, so the answer is a true description. The score is 0.967. Zero context overlap
`+2.91`, cosine `+0.82`, and contradiction `+0.54` flag a grounded answer. The
explanation is `implausible`, and the parent benchmark label 1 is also suspect. Record
both points in `notes`.

### 4.6 Columns to edit

| Column | Action |
|---|---|
| `ai_suggested` | Leave as the AI plausibility label. |
| `author_label` | Your `plausible` / `implausible` / `unsure`. |
| `final_label` | Leave blank until adjudication. |
| `annotator_id` | Your initials. |
| `changed` | `yes` if your label differs from `ai_suggested`, else `no`. |
| `review_status` | `author_verified` or `author_corrected`. |
| `notes` | One sentence, especially when the parent label looks wrong. |

Do not edit `raw_score`, `top_features`, or `top_shap` unless the model is re-exported.
If you re-export, state the new model and seed in `notes`.

---

## 5. Sheet 3: `p0_claim_verdicts.csv`

### 5.1 What to check

Judge one atomic claim against the **evidence sentence shown in the row**. Do not use
world knowledge. Do not use the rest of the context to rescue a bad evidence sentence.
The evidence sentence is the one the deterministic NLI step selected; if it is
off-topic, that is itself a finding about the claim-verification pipeline.

### 5.2 How to decide

- `supported`: the evidence entails the claim, including faithful paraphrase and
  matching number or date.
- `contradicted`: the evidence states the opposite or a conflicting value for the same
  slot.
- `unsupported`: the evidence is relevant but silent on the asserted relation.
- `unsure`: the evidence sentence is a fragment, off-topic, or ambiguous, so the claim
  cannot be judged.

A claim can be true in the world and still be `unsupported`. A claim that restates the
question is not supported. A claim with a wrong number is `contradicted` when the
evidence gives the right number, and `unsupported` when the evidence is silent.

### 5.3 Worked example: easy

`P0C003` / `ragtruth:12445`. Claim: "Essential amino acids are those that cannot be made
by the body and must be obtained from the diet." Evidence: "Nonessential amino acids can
be made by the body, while essential amino acids cannot be made by the body so you must
get them from your diet." The evidence states the same relation. `supported`,
confidence 5.

### 5.4 Worked example: ambiguous

`P0C033` / `ragtruth:14764`. Claim: "the normal value of B-type natriuretic peptide
(BNP) in the blood is not explicitly mentioned." Evidence: "Normally, only a low amount
of BNP is found in your blood." The evidence gives a qualitative normal statement but no
numeric value. A strict reader calls the claim `supported`; a reader who treats "low
amount" as a normal value calls it `contradicted`. The AI chose `supported` at
confidence 4. If you prefer the other reading, change it and say why.

### 5.5 Worked example: label noise

`P0C037` / `ragtruth:14860`. Claim: "you need to use the Shugabush and the Furcorn in
the Shugabush Island breeding structure." Evidence repeats the same combination. The
claim is `supported`. The parent answer is labelled 1 (hallucinated), but that parent
label does not decide this claim. This is the claim-level version of label noise: a
supported claim can sit inside an answer that the benchmark rejects. Do not import the
parent label into `author_label`.

### 5.6 Columns to edit

| Column | Action |
|---|---|
| `ai_suggested` | Leave as the AI verdict. |
| `author_label` | Your verdict. |
| `final_label` | Leave blank until adjudication. |
| `annotator_id` | Your initials. |
| `changed` | `yes` if your verdict differs from `ai_suggested`, else `no`. |
| `review_status` | `author_verified` or `author_corrected`. |
| `notes` | One sentence, especially when the evidence sentence is off-topic. |

---

## 6. Blind overlap procedure

The blind sheet is the only agreement set. It has 60 items, 20 from each sheet, with no
AI labels.

### 6.1 Prepare three copies

```powershell
Copy-Item docs/rebuild/p0/p0_blind_overlap.csv docs/rebuild/p0/p0_blind_overlap_initials1.csv
Copy-Item docs/rebuild/p0/p0_blind_overlap.csv docs/rebuild/p0/p0_blind_overlap_initials2.csv
Copy-Item docs/rebuild/p0/p0_blind_overlap.csv docs/rebuild/p0/p0_blind_overlap_initials3.csv
```

Replace `initials1`-`initials3` with the three authors' initials.

### 6.2 Label independently

Each author fills only `author_label`, `annotator_id`, and `notes` in their own copy.

| `sheet` value | What to write in `author_label` |
|---|---|
| `taxonomy` | One 20-category code from the codebook. |
| `explanation` | `plausible`, `implausible`, or `unsure`. |
| `claims` | `supported`, `contradicted`, `unsupported`, or `unsure`. |

Rules:

- Do not open the three main sheets, the AI rationales, or another author's copy until
  all three blind copies are locked.
- Do not discuss items during labelling.
- Use the codebook, not memory.
- For taxonomy items, if you think the answer is grounded but the benchmark label is 1,
  use `artifact_label_noise`.
- Leave `notes` blank unless the item is genuinely contested. A short reason is enough.

### 6.3 Lock and hand off

When finished, save the file and record its hash:

```powershell
Get-FileHash docs/rebuild/p0/p0_blind_overlap_initials1.csv -Algorithm SHA256
```

Send the file and the hash to the coordinator. The coordinator checks that each file has
60 rows, 20 per sheet, no blank `author_label`, and the correct `annotator_id`. Only then
are the AI sheets revealed for adjudication.

### 6.4 Merge

```powershell
& .venv\Scripts\python.exe -c "import pandas as pd; a=pd.read_csv('docs/rebuild/p0/p0_blind_overlap_initials1.csv').set_index('item_id'); b=pd.read_csv('docs/rebuild/p0/p0_blind_overlap_initials2.csv').set_index('item_id'); c=pd.read_csv('docs/rebuild/p0/p0_blind_overlap_initials3.csv').set_index('item_id'); m=a[['sheet','author_label']].rename(columns={'author_label':'a1'}).join(b[['author_label']].rename(columns={'author_label':'a2'})).join(c[['author_label']].rename(columns={'author_label':'a3'})); print(m.shape); print(m.isna().sum().to_dict()); m.to_csv('docs/rebuild/p0/p0_blind_overlap_merged.csv')"
```

Expected `(60, 4)` and zero missing values. Keep the three raw copies; never overwrite
them after the lock.

---

## 7. Fleiss' kappa

Fleiss' kappa generalises Cohen's kappa to more than two raters and a nominal label set.
Use it **separately for each sheet**, because the three label sets are different. Do not
pool the 60 items into one kappa.

### 7.1 Formula

For `N` items, `n` raters, and `k` categories:

- `n_ij` = number of raters who assigned item `i` to category `j`.
- `P_i = [sum_j n_ij (n_ij - 1)] / [n (n - 1)]` is the agreement on item `i`.
- `P_bar = (1 / N) * sum_i P_i` is the observed agreement.
- `p_j = (1 / (N n)) * sum_i n_ij` is the marginal proportion of category `j`.
- `P_e = sum_j p_j^2` is the agreement expected by chance.
- `kappa = (P_bar - P_e) / (1 - P_e)`.

For three raters, `n = 3`. A worked check: if all three agree on every item, `P_i = 1`,
`P_bar = 1`, and `kappa = 1`. If agreement equals chance, `P_bar = P_e` and `kappa = 0`.

### 7.2 Python

```python
import numpy as np
import pandas as pd

def fleiss_kappa(label_rows):
    """label_rows: list of lists, one label list per item, same length per item."""
    cats = sorted({x for row in label_rows for x in row})
    index = {c: i for i, c in enumerate(cats)}
    n_items, n_raters = len(label_rows), len(label_rows[0])
    counts = np.zeros((n_items, len(cats)), dtype=int)
    for i, row in enumerate(label_rows):
        for label in row:
            counts[i, index[label]] += 1
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_j ** 2).sum()
    return (p_bar - p_e) / (1 - p_e), p_bar, p_e

merged = pd.read_csv("docs/rebuild/p0/p0_blind_overlap_merged.csv").set_index("item_id")
for sheet in ["taxonomy", "explanation", "claims"]:
    frame = merged[merged["sheet"] == sheet]
    rows = frame[["a1", "a2", "a3"]].values.tolist()
    kappa, observed, expected = fleiss_kappa(rows)
    unanimous = sum(len(set(row)) == 1 for row in rows) / len(rows)
    print(sheet, "N=", len(rows), "kappa=", round(kappa, 3),
          "observed=", round(observed, 3), "expected=", round(expected, 3),
          "unanimous=", round(unanimous, 3))
```

For the taxonomy sheet, also collapse to two families before the kappa:

```python
def collapse_taxonomy(label):
    return "artifact" if label.startswith("artifact_") else "content"

frame = merged[merged["sheet"] == "taxonomy"].copy()
rows = [[collapse_taxonomy(x) for x in row] for row in frame[["a1", "a2", "a3"]].values]
print("taxonomy family kappa:", fleiss_kappa(rows))
```

Report the 20-category exact agreement as a percentage beside the collapsed kappa. A
20-category kappa on 20 items is too sparse to interpret.

### 7.3 Interpretation and reporting

Use Landis and Koch's descriptive bands: below 0 poor, 0.00-0.20 slight, 0.21-0.40 fair,
0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect. Treat the bands as
descriptions, not pass/fail gates.

With 20 items per sheet and three authors, the interval is wide. Report:

- the number of items and raters,
- the label set and whether it was collapsed,
- kappa to two decimals,
- raw unanimous agreement,
- a bootstrap 95% interval over items (1,000 resamples) if the paper needs one,
- the fact that the blind set is a 60-item subset, not the full 250 rows.

If kappa is negative, report it as near zero and explain that the sample is small. Do
not describe a negative kappa as "disagreement beyond chance" without that caveat.

---

## 8. Adjudication rule

1. Lock the three blind copies before revealing the AI labels.
2. Merge on `item_id`. Confirm 60 rows and no missing labels.
3. For each item:
   - **3 of 3 agree**: `final_label` is that label.
   - **2 of 3 agree**: `final_label` is the majority label. Record the dissent in
     `notes`.
   - **All three differ**: discuss the item against the codebook. If two authors can
     converge, use the converged label. If not, set `final_label` to `unsure` for
     explanation or claim items. For taxonomy items, set it to
     `artifact_question_ambiguity` only if the question truly admits multiple readings;
     otherwise set it to the family-level label (`content` or `artifact`) and exclude it
     from category counts. State the unresolved count in the manifest.
4. Copy `final_label` into the matching row of the main sheet by `item_id`
   (`case_id` for taxonomy and explanation, `claim_id` for claims).
5. If adjudication changes a category, update `primary_category` in the main sheet and
   note the change. Do not edit `label`.
6. Never average confidence scores. Confidence is per annotator and per row.

The AI suggestion is revealed only after the blind labels are locked. It is one more
piece of evidence for discussion, not a tie-breaker.

---

## 9. Exact wording for the paper

### 9.1 While verification is pending

> The P0 package is an AI pre-annotation prepared for three-author blinded review. All
> 250 annotation rows carry `review_status = needs_author_verification`. No count from
> the package is reported as a verified result.

Use this wording until every row has an author label and the blind set is complete.

### 9.2 After all three authors complete the blind set

> All 250 rows were pre-annotated by an AI assistant and then reviewed by the three
> authors. The benchmark label was treated as a cross-check, not as ground truth. Each
> author labelled the 60-item blind subset independently before seeing the AI
> suggestions. Agreement was summarised with Fleiss' kappa separately for the taxonomy
> family, explanation plausibility, and claim verdicts (kappa = X.XX, X.XX, and X.XX;
> N = 20 each). Disagreements were resolved by majority, and unresolved items were
> reported as unsure. The audit samples are error-enriched and length-stratified, so the
> category counts describe the audited cases, not the prevalence of error types in the
> HaluEval test set.

Replace the three `X.XX` values with the computed numbers. If a sheet did not reach
three completed copies, delete that clause and report the sheet as descriptive only.

### 9.3 Table caption

> Error taxonomy for the audited cases. Counts are from the P0 error-enriched sample
> (25 benchmark-negative and 25 benchmark-positive HaluEval test cases), not from a
> random sample of the test set. Artifact categories describe the detector, the
> benchmark label, or the data export and are not hallucination prevalence.

### 9.4 Limitations sentence

> The taxonomy audit over-samples model errors and near-threshold cases, and the
> explanation audit combines an error-enriched half with a length-stratified random
> half. Neither half supports a prevalence estimate.

### 9.5 Wording to avoid

Do not describe the package as completed annotation, as a second review by a person, as
a two-rater study, or as already verified. Do not call the benchmark labels ground
truth. Do not report a category percentage as a test-set rate. Do not claim a detection
accuracy from this package; it is not an evaluation split. Do not report kappa until all
three blind copies are locked and complete.

---

## 10. Master checklist

### Before review

- [ ] Branch `rebuild`, no edits under `artifacts/`, `Journal_Paper/`, or
      `data/processed/`.
- [ ] Backup of `docs/rebuild/p0/`.
- [ ] Codebook Sections 3-6 read.
- [ ] Row counts and statuses match Section 1.

### Taxonomy (50 rows)

- [ ] Every row read in full.
- [ ] Exactly one primary category per row.
- [ ] Secondary only for an independent second defect.
- [ ] `author_label` filled with a valid category code.
- [ ] `changed` and `review_status` set.
- [ ] Near-threshold rows are not described as threshold false positives.

### Explanation (100 rows)

- [ ] Answer correctness judged separately from explanation plausibility.
- [ ] Top signed terms read, not just the top feature name.
- [ ] `author_label` is one of `plausible`, `implausible`, `unsure`.
- [ ] Parent-label disagreements recorded in `notes`.
- [ ] `changed` and `review_status` set.

### Claims (100 rows)

- [ ] Claim judged against the evidence sentence shown.
- [ ] No world knowledge used to rescue an off-topic sentence.
- [ ] `author_label` is one of the four verdicts.
- [ ] Off-topic evidence sentences recorded in `notes`.
- [ ] `changed` and `review_status` set.

### Blind set (60 rows)

- [ ] Three separate copies, 20 items per sheet each.
- [ ] No AI sheets opened before the lock.
- [ ] No discussion during labelling.
- [ ] All 60 `author_label` values filled in each copy.
- [ ] Hashes recorded.
- [ ] Merge has 60 rows and no missing values.
- [ ] Kappa computed per sheet, not pooled.
- [ ] Taxonomy kappa reported on the collapsed family and exact agreement reported
      separately.

### After adjudication

- [ ] `final_label` filled for every blind item.
- [ ] Unresolved items counted and excluded from category counts.
- [ ] Main sheets updated with adjudicated labels.
- [ ] Manifest counts and statuses refreshed.
- [ ] Paper wording uses Section 9 and the error-enriched caveat.
- [ ] No banned wording from Section 9.5.

---

## 11. Common mistakes

1. **Labelling from the snippet.** The snippets are 160 characters. The decisive
   sentence is often later. Print the full case.
2. **Treating the benchmark label as truth.** A grounded answer with `label = 1` is
   `artifact_label_noise`, not a model success.
3. **Editing `label`.** Never. Record disagreement in `author_label`, `final_label`, and
   `notes`.
4. **Calling every `label = 0` row a false positive.** Only 8 of the 25 taxonomy
   negatives cross 0.5. The rest are near-threshold candidates.
5. **Using `artifact_*` counts as hallucination prevalence.** Artifact categories are
   detector or benchmark artifacts.
6. **Marking a grounded answer `answer_span_copy` only.** A grounded span gets
   `artifact_grounded_overflag` primary and `answer_span_copy` secondary.
7. **Confusing answer correctness with explanation plausibility.** A correct answer can
   have an implausible explanation, and a wrong answer can have a plausible one.
8. **Judging a claim from memory.** Judge only the evidence sentence in the row.
9. **Forcing a binary claim verdict.** `unsure` is correct when the evidence sentence is
   a fragment or off-topic.
10. **Pooling the three label sets into one kappa.** Compute one kappa per sheet.
11. **Computing kappa before all three blind copies are locked.** The statistic is
    meaningless if authors saw each other's labels.
12. **Using Excel's default CSV open.** It can change UTF-8 characters and strip leading
    zeros. Use Data > From Text/CSV or VS Code.
13. **Writing `NA` in a categorical field.** Leave it blank.
14. **Reporting a category percentage as a test-set rate.** The sample is
    error-enriched.
15. **Writing any banned phrase from Section 9.5.** The package is an AI pre-annotation
    with author verification, not a completed annotation study.

---

## 12. Quick reference

**Content categories (15):** `entity_misattribution`, `entity_conflation`,
`entity_substitution`, `entity_name_distortion`, `fabricated_entity`,
`generic_or_descriptive_answer`, `relation_inversion`, `attribute_error`,
`numeric_or_temporal_error`, `wrong_granularity`, `coreference_error`,
`non_responsive_or_partial_answer`, `unsupported_specificity`, `context_contradiction`,
`answer_span_copy`.

**Artifact categories (5):** `artifact_grounded_overflag`, `artifact_short_answer_fp`,
`artifact_label_noise`, `artifact_truncated_context`, `artifact_question_ambiguity`.

**Explanation labels:** `plausible`, `implausible`, `unsure`.

**Claim labels:** `supported`, `contradicted`, `unsupported`, `unsure`.

**Review statuses:** `needs_author_verification`, `author_verified`,
`author_corrected`.

**Model:** EC-XGB m3 fixed hyperparameters, seed 42
(`artifacts/models/rebuild/fixed_hp/xgboost_m3_seed_42.joblib`), 35 features, raw
probability threshold 0.5 for the 0/1 decision. The tuned strict threshold on this split
is 0.095; the package reports raw scores and states the threshold context in the
rationales.

**SHAP:** exact TreeSHAP via `pred_contribs=True`, raw margin (log-odds), top five by
absolute value.
