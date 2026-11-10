# WP5 Human Verification Checklist

**Purpose:** verify or correct every AI-drafted label before any WP5 number enters the
manuscript.

**Integrity rule:** the current sheets are an AI-assisted draft. Do not describe them
as human annotation, independent review, or a two-annotator study. Every row starts at
`review_status = needs_author_verification`. Change that value only after you have
personally checked the row.

**Files:**

- `docs/rebuild/wp5_codebook.md` — category definitions and decision rules.
- `docs/rebuild/wp5_error_labels.csv` — 20 m3 error cases.
- `docs/rebuild/wp5_explanation_labels.csv` — 26 m3 explanation cases.
- `docs/rebuild/wp5_summary.md` — counts, caveats, blank kappa worksheets.
- This checklist.

---

## 0. Before you start

- [ ] Read `wp5_codebook.md` §3 (support test), §4 (primary/secondary rule), and §5
      (confidence scale).
- [ ] Confirm you are on branch `rebuild` and that you will not edit anything under
      `artifacts/`. The source tables are frozen; the WP5 sheets are new files under
      `docs/rebuild/`.
- [ ] Open the two CSVs in a CSV editor that preserves UTF-8 (Excel, LibreOffice, or
      VS Code). Do not use a tool that changes commas inside quoted fields.
- [ ] Keep a backup copy before editing:
      `Copy-Item docs/rebuild/wp5_error_labels.csv docs/rebuild/wp5_error_labels.backup.csv`
      and the same for the explanation sheet.

### 0.1 See the full text for a case

For the error sheet (HaluEval text from `qa_clean.parquet`):

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/qa_clean.parquet').set_index('sample_id').loc[sys.argv[1]]; print('Q:',r['question']); print(); print('C:',r['context']); print(); print('A:',r['answer'])" q_4514_hallucinated
```

For the explanation sheet, the m3 audit CSV has the snippet and top SHAP; full text is
in `unified_records.parquet` with the `halueval:` prefix:

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/unified_records.parquet').set_index('sample_id').loc['halueval:'+sys.argv[1]]; print('Q:',r['question']); print(); print('C:',r['context']); print(); print('A:',r['answer'])" q_8518_correct
```

Replace the `sample_id` at the end with the row you are checking.

### 0.2 Check the structure before labelling

- [ ] `wp5_error_labels.csv` has 20 data rows and these 9 columns, in this order:
      `case_id,sample_id,label,answer_snippet,primary_category,secondary_category,rationale,confidence_1to5,review_status`.
- [ ] `wp5_explanation_labels.csv` has 26 data rows and these 8 columns, in this order:
      `case_id,sample_id,top_feature_1,top_feature_2,explanation_plausible,rationale,confidence_1to5,review_status`.
- [ ] Every `case_id` is unique (`E01`-`E20`, `X01`-`X26`).
- [ ] Every `sample_id` is unique within its sheet and matches the source table.
- [ ] Every `answer_snippet` is at most 160 characters and is a prefix of the source
      answer.
- [ ] Every row has `review_status = needs_author_verification`.

Quick check:

```powershell
& .venv\Scripts\python.exe -c "import pandas as pd; e=pd.read_csv('docs/rebuild/wp5_error_labels.csv'); x=pd.read_csv('docs/rebuild/wp5_explanation_labels.csv'); print(e.shape, x.shape); print(e['case_id'].is_unique, x['case_id'].is_unique); print(e['review_status'].value_counts().to_dict()); print(x['review_status'].value_counts().to_dict()); print(e['answer_snippet'].str.len().max())"
```

Expected: `(20, 9) (26, 8)`, both `True`, both statuses only
`needs_author_verification`, max snippet length <= 160.

---

## 1. Exact columns to edit

### 1.1 Error sheet (`wp5_error_labels.csv`)

| Column | Edit? | Rule |
|---|---|---|
| `case_id` | No | Join key; leave `E01`-`E20`. |
| `sample_id` | No | Join key; must match `qa_clean.parquet`. |
| `label` | Only if the benchmark label is wrong | 0 = grounded, 1 = hallucinated. If you change it, record the old value in the checklist notes. |
| `answer_snippet` | Only if it does not match the source answer | Keep <= 160 characters; do not paraphrase. |
| `primary_category` | **Yes** | Exactly one code from the 20-category list in §8. |
| `secondary_category` | **Yes** | A second code, or leave blank. Do not restate the primary. |
| `rationale` | **Yes** | Quote or paraphrase the decisive context span. State FP/FN if relevant. |
| `confidence_1to5` | **Yes** | Integer 1-5. |
| `review_status` | **Yes** | `needs_author_verification` -> `author_verified` or `author_corrected`. |

### 1.2 Explanation sheet (`wp5_explanation_labels.csv`)

| Column | Edit? | Rule |
|---|---|---|
| `case_id` | No | Join key; leave `X01`-`X26`. |
| `sample_id` | No | Join key. |
| `top_feature_1` | No, unless re-exporting on a new model | Must match the m3 `top_features` order. |
| `top_feature_2` | No, unless re-exporting on a new model | Same. |
| `explanation_plausible` | **Yes** | `plausible`, `implausible`, or `unsure`. |
| `rationale` | **Yes** | Explain why the top signed SHAP terms do or do not fit the score and evidence. |
| `confidence_1to5` | **Yes** | Integer 1-5. |
| `review_status` | **Yes** | `needs_author_verification` -> `author_verified` or `author_corrected`. |

**Do not edit** `docs/rebuild/wp5_summary.md` counts by hand until all rows are
verified. Recompute them with the command in §6.

---

## 2. Error sheet: per-case checklist (20 rows)

For each row: open the full question/context/answer, read the current rationale, and
decide whether the primary category is right. If it is wrong, change
`primary_category`. If a second independent defect is present, add
`secondary_category`; otherwise leave it blank. Update `rationale` and
`confidence_1to5` whenever you change a label. Set `review_status` at the end.

### 2.1 False positives (label 0, m3 score >= 0.5)

- [ ] **E01 / `q_3027_correct`** — m3 0.526; current primary `answer_span_copy`,
      secondary `artifact_grounded_overflag`, confidence 2. Check: does the 80-word
      span actually name the company that acquired the railroad, or only the operator
      and Conrail? This is the most contestable row. Edit: `primary_category`,
      `secondary_category`, `rationale`, `confidence_1to5`, `review_status`.
- [ ] **E02 / `q_3061_correct`** — m3 0.864; current primary
      `artifact_short_answer_fp`. Check: is "portrait" supported, and is the answer one
      word with zero overlap? Edit: same columns.
- [ ] **E03 / `q_4329_correct`** — m3 0.778; current primary
      `artifact_grounded_overflag`. Check: is "2007 UEFA Champions League Final" the
      exact entity in the context? Edit: same columns.
- [ ] **E04 / `q_5922_correct`** — m3 0.602; current primary
      `artifact_grounded_overflag`. Check: does the context name UTEP as the
      university? Edit: same columns.
- [ ] **E05 / `q_6775_correct`** — m3 0.799; current primary
      `generic_or_descriptive_answer`, secondary `artifact_grounded_overflag`,
      confidence 3. Check: the answer describes Benito Juárez but does not name him.
      Resolve ambiguity E05 in §3 before finalizing. Edit: same columns.
- [ ] **E06 / `q_7742_correct`** — m3 0.515; current primary
      `artifact_grounded_overflag`, confidence 4. Check: is the answer verbatim in the
      context? Edit: same columns.
- [ ] **E07 / `q_8518_correct`** — m3 0.803; current primary
      `artifact_short_answer_fp`. Check: is "Australia" supported, and is the answer
      one word with zero overlap? Edit: same columns.
- [ ] **E08 / `q_8650_correct`** — m3 0.958; current primary
      `artifact_grounded_overflag`. Check: does the context name the Atlantic Islands
      of Galicia National Park as the only park in Galicia? Edit: same columns.

### 2.2 False negatives (label 1, m3 score < 0.5)

- [ ] **E09 / `q_3058_hallucinated`** — m3 0.298; current primary
      `entity_conflation`, secondary `context_contradiction`. Check: does the context
      name Deborah Kaplan and Harry Elfont, and does "Harry Kaplan" match either? Edit:
      same columns.
- [ ] **E10 / `q_4896_hallucinated`** — m3 0.233; current primary
      `non_responsive_or_partial_answer`, secondary `artifact_truncated_context`,
      confidence 3. Check: does the answer address both works and the yes/no question?
      Is the context malformed at 138 characters? Edit: same columns.
- [ ] **E11 / `q_8501_hallucinated`** — m3 0.335; current primary
      `entity_name_distortion`, secondary `unsupported_specificity`. Check: does
      "The G7 Summit Research Group" occur in the context, or is it a blend of "G7
      Research Group" and "G7 Summit"? Edit: same columns.
- [ ] **E12 / `q_1702_hallucinated`** — m3 0.048; current primary
      `generic_or_descriptive_answer`, secondary `fabricated_entity`. Check: does the
      context name the Dieppe Raid? Resolve ambiguity E05/E12 in §3. Edit: same columns.
- [ ] **E13 / `q_3297_hallucinated`** — m3 0.312; current primary
      `artifact_label_noise`. Check: do the context dates support "died in 2014"?
      Edit: same columns.
- [ ] **E14 / `q_4514_hallucinated`** — m3 0.339; current primary
      `entity_misattribution`, secondary `entity_substitution`. Check: is Little
      Creatures the company established in 2000 by the original Matilda Bay brewers?
      Edit: same columns.
- [ ] **E15 / `q_5861_hallucinated`** — m3 0.226; current primary
      `coreference_error`, secondary `artifact_question_ambiguity`, confidence 2.
      Check: does "it" refer to *You and I* or the 1984 film? Is Goldcrest a valid
      co-producer? Resolve ambiguity E15 in §3. Edit: same columns.
- [ ] **E16 / `q_7223_hallucinated`** — m3 0.405; current primary
      `numeric_or_temporal_error`. Check: does the context say Danica McKellar wrote
      five non-fiction books? Edit: same columns.
- [ ] **E17 / `q_7398_hallucinated`** — m3 0.224; current primary
      `relation_inversion`, secondary `entity_misattribution`. Check: did Anthony
      Hickox direct Knife Edge, with Natalie Press starring? Edit: same columns.
- [ ] **E18 / `q_1739_hallucinated`** — m3 0.143; current primary
      `wrong_granularity`, secondary `entity_substitution`. Check: is "Creep" the single
      and "Blood Money" the album? Edit: same columns.
- [ ] **E19 / `q_8342_hallucinated`** — m3 0.403; current primary `attribute_error`,
      secondary `entity_substitution`. Check: is Dante the Brazilian who plays for
      Nice, and is Pizarro Peruvian? Edit: same columns.
- [ ] **E20 / `q_2184_hallucinated`** — m3 0.076; current primary
      `fabricated_entity`. Check: does the context name "Act Like a Lady, Think Like a
      Man", and is "The Lady and the Man" absent? Edit: same columns.

---

## 3. Resolve these ambiguities before finalizing dependent rows

### 3.1 E01 / `q_3027_correct`: copied span versus answer

- [ ] Read the full context: the JKWI is described as the operator, and the span says
      it "had acquired from Conrail" the line segments.
- [ ] Decide whether the gold label 0 is defensible (the span contains the acquisition
      clause) or whether the answer never names the acquirer.
- [ ] If the span is accepted, keep `answer_span_copy` as primary and
      `artifact_grounded_overflag` as secondary. If it is rejected, change the primary
      to `relation_inversion` or `non_responsive_or_partial_answer` and record the
      reason.
- [ ] Update `rationale`, `confidence_1to5`, and `review_status`.

### 3.2 E15 / `q_5861_hallucinated`: Goldcrest versus Puttnam

- [ ] Read the full context: *You and I* was "produced by David Puttnam for his company
      Goldcrest Films."
- [ ] Decide whether the answer's "it" refers to *You and I* (supported), the 1984
      film (not stated in the context), or is unresolved.
- [ ] Decide whether "Goldcrest Films" is a valid answer to "who produced" or whether
      the expected answer is David Puttnam.
- [ ] Set one primary category and (optionally) one secondary. Record the decision in
      the rationale and in the notes column of §5.

### 3.3 E05 / `q_6775_correct` and E12 / `q_1702_hallucinated`: description versus name

- [ ] Choose one rule:
      - **Name-only:** a name-seeking question requires the named entity.
      - **Unique-description:** a description that uniquely identifies the entity
        counts as correct.
      - **Case-by-case:** document why a description is specific enough.
- [ ] Apply the rule to E05 (`"liberal politician of Zapotec origin from Oaxaca."`) and
      E12 (`"Northern France Battle"` vs the Dieppe Raid).
- [ ] If the rule changes a label, update `primary_category`, `secondary_category`,
      `rationale`, `confidence_1to5`, and `review_status`.

### 3.4 Also check E13 / `q_3297_hallucinated`

- [ ] Decide whether the supported answer (Fred Phelps Sr. died in 2014) is benchmark
      label noise.
- [ ] Decide whether to exclude it from the false-negative count, keep it as
      `artifact_label_noise`, or report it separately. State the decision in
      `wp5_summary.md`.

---

## 4. Explanation sheet: per-case checklist (26 rows)

For each row: read the question, context, answer, m3 raw score, and the top-2 signed
SHAP terms. Decide whether the feature story is a plausible account of the score given
the evidence. Change `explanation_plausible` only if you disagree with the AI draft.
Update `rationale`, `confidence_1to5`, and `review_status`.

### 4.1 Currently `implausible` (13 rows)

- [ ] **X01 / `q_8518_correct`** — m3 0.803; zero overlap pushes +2.52. Grounded one-word answer.
- [ ] **X02 / `q_3061_correct`** — m3 0.864; zero overlap +2.70, length -1.65. Grounded one-word answer.
- [ ] **X05 / `q_8650_correct`** — m3 0.958; question similarity +2.16/+1.66 overwhelms context overlap -2.70.
- [ ] **X07 / `q_5922_correct`** — m3 0.602; question similarity +1.40/+0.99 pushes over.
- [ ] **X08 / `q_4329_correct`** — m3 0.778; question similarity +1.08/+0.79 pushes over.
- [ ] **X10 / `q_4539_hallucinated`** — m3 0.379; context overlap -3.21 masks the wrong athlete.
- [ ] **X11 / `q_2910_hallucinated`** — m3 0.154; length -2.04/-1.79 hides the Syfy/AMC contradiction.
- [ ] **X12 / `q_1921_hallucinated`** — m3 0.119; context overlap -2.65 masks the capacity/population error.
- [ ] **X13 / `q_5384_hallucinated`** — m3 0.004; overlap/length -2.75/-1.55/-1.18 masks the festival substitution.
- [ ] **X15 / `q_1739_hallucinated`** — m3 0.143; context overlap -2.71 masks the album/single error.
- [ ] **X16 / `q_2184_hallucinated`** — m3 0.076; context overlap -3.41 masks the fabricated title.
- [ ] **X24 / `q_8342_hallucinated`** — m3 0.403; context overlap -2.68, no contradiction term for Pizarro's nationality.
- [ ] **X25 / `q_6775_correct`** — m3 0.799; question similarity/length +0.90/+0.82 pushes a grounded description over.

### 4.2 Currently `plausible` (10 rows)

- [ ] **X04 / `q_1402_correct`** — m3 0.401; context overlap -3.41 dominates a grounded answer.
- [ ] **X06 / `q_4156_correct`** — m3 0.320; context overlap -3.01 dominates a grounded team name.
- [ ] **X09 / `q_7876_correct`** — m3 0.403; context overlap -3.27 dominates a copied state.
- [ ] **X17 / `q_2251_hallucinated`** — m3 0.270; supported Tyler Winklevoss answer, low score sensible.
- [ ] **X18 / `q_2436_hallucinated`** — m3 0.296; supported Southbank population, low score sensible.
- [ ] **X19 / `q_5173_hallucinated`** — m3 0.011; EMC prior name, low score sensible but question ambiguous.
- [ ] **X21 / `q_6297_correct`** — m3 0.282; context overlap -2.45 dominates a supported actor.
- [ ] **X22 / `q_4092_hallucinated`** — m3 0.669; neutral NLI +0.91 flags a non-responsive answer.
- [ ] **X23 / `q_9100_correct`** — m3 0.403; context overlap -2.36 and length -2.02 dominate a supported band.
- [ ] **X26 / `q_6955_correct`** — m3 0.319; context overlap -2.65 dominates a supported parish.

### 4.3 Currently `unsure` (3 rows)

- [ ] **X03 / `q_7742_correct`** — m3 0.515; context overlap -2.96 supports low risk, but question overlap/length push just over.
- [ ] **X14 / `q_1729_hallucinated`** — m3 0.490; marginal, mixed context-overlap and question-similarity terms.
- [ ] **X20 / `q_6989_hallucinated`** — m3 0.461; borderline description of CFB Trenton, not a name.

### 4.4 Note on the source passes

The `reviewer_1` / `reviewer_2` columns in `b5_review_cases_reviewed.csv` are two
AI-expert passes, not two humans. The current sheet does **not** reconcile those
passes; it is a fresh m3 SHAP reading. If you compare your label to the B5 history,
describe it as a B2-to-m3 re-audit, not as human agreement.

---

## 5. Second rater and kappa worksheet

Do this only if a second human is available. Do not compute kappa from the AI labels
alone.

### 5.1 Prepare the second-rater copies

- [ ] Create `docs/rebuild/wp5_explanation_labels_second_rater.csv` with only
      `case_id,sample_id,top_feature_1,top_feature_2` and blank
      `explanation_plausible,confidence_1to5,rationale`.
- [ ] Create `docs/rebuild/wp5_error_labels_second_rater.csv` with only
      `case_id,sample_id,label,answer_snippet` and blank
      `primary_category,secondary_category,rationale,confidence_1to5`.
- [ ] Give the second rater the codebook, the full question/context/answer, and the
      feature names. Do not give them the AI labels, rationales, or this checklist's
      current labels.

### 5.2 Explanation kappa (3x3)

Fill the blank cells in `wp5_summary.md` §5.1. Row totals are 10 plausible, 13
implausible, 3 unsure.

- [ ] Second rater has labelled all 26 cases independently.
- [ ] The 3x3 table is complete.
- [ ] `p_o`, `p_e`, and kappa are computed from the completed table.
- [ ] The 95% CI method is stated.
- [ ] The result is described as agreement between the AI-assisted draft and one human
      rater, not as inter-human agreement.

### 5.3 Error-sheet agreement (2x2 or per-case)

- [ ] Second rater has labelled all 20 cases independently.
- [ ] The collapsed content/artifact 2x2 in `wp5_summary.md` §5.2 is complete (row
      totals 13 content, 7 artifact).
- [ ] Per-case agreement is recorded in the table below.
- [ ] If the full 20-category labels are used, report percent agreement and a
      confusion table; do not report a single 20x20 kappa.

### 5.4 Per-case agreement notes

| Case | AI primary | Second-rater primary | Agree? (Y/N) | AI secondary | Second-rater secondary | Notes |
|---|---|---|---|---|---|---|
| E01 | `answer_span_copy` | | | `artifact_grounded_overflag` | | |
| E02 | `artifact_short_answer_fp` | | | | | |
| E03 | `artifact_grounded_overflag` | | | | | |
| E04 | `artifact_grounded_overflag` | | | | | |
| E05 | `generic_or_descriptive_answer` | | | `artifact_grounded_overflag` | | |
| E06 | `artifact_grounded_overflag` | | | | | |
| E07 | `artifact_short_answer_fp` | | | | | |
| E08 | `artifact_grounded_overflag` | | | | | |
| E09 | `entity_conflation` | | | `context_contradiction` | | |
| E10 | `non_responsive_or_partial_answer` | | | `artifact_truncated_context` | | |
| E11 | `entity_name_distortion` | | | `unsupported_specificity` | | |
| E12 | `generic_or_descriptive_answer` | | | `fabricated_entity` | | |
| E13 | `artifact_label_noise` | | | | | |
| E14 | `entity_misattribution` | | | `entity_substitution` | | |
| E15 | `coreference_error` | | | `artifact_question_ambiguity` | | |
| E16 | `numeric_or_temporal_error` | | | | | |
| E17 | `relation_inversion` | | | `entity_misattribution` | | |
| E18 | `wrong_granularity` | | | `entity_substitution` | | |
| E19 | `attribute_error` | | | `entity_substitution` | | |
| E20 | `fabricated_entity` | | | | | |

For the explanation sheet, use the 3x3 table in `wp5_summary.md` §5.1. If you also
want a per-case log, add a copy of the table above with `X01`-`X26`.

---

## 6. After edits: update counts and status

- [ ] Recompute the error counts:

```powershell
& .venv\Scripts\python.exe -c "import pandas as pd; e=pd.read_csv('docs/rebuild/wp5_error_labels.csv'); print(e['primary_category'].value_counts().to_string()); print(); print(e['secondary_category'].value_counts().to_string()); print(); print(e['review_status'].value_counts().to_string())"
```

- [ ] Recompute the explanation counts:

```powershell
& .venv\Scripts\python.exe -c "import pandas as pd; x=pd.read_csv('docs/rebuild/wp5_explanation_labels.csv'); print(x['explanation_plausible'].value_counts().to_string()); print(); print(x['review_status'].value_counts().to_string())"
```

- [ ] Update the tables in `wp5_summary.md` §2 and §3 to match the recomputed counts.
- [ ] If any row is still `needs_author_verification`, either finish it or exclude it
      from the paper counts and say so.
- [ ] Do not leave a row as `author_verified` if you only skimmed it. The status is a
      claim of personal review.

---

## 7. Sign-off

- [ ] Author name / initials: ______________________
- [ ] Date completed: ______________________
- [ ] Error rows verified: ____ / 20
- [ ] Error rows corrected: ____ / 20
- [ ] Explanation rows verified: ____ / 26
- [ ] Explanation rows corrected: ____ / 26
- [ ] Rows still `needs_author_verification`: ____
- [ ] Second rater completed: yes / no / not available
- [ ] Kappa computed: yes / no (if no, the paper must not report kappa)
- [ ] I confirm that the `review_status` values reflect my personal review, and that
      the paper will describe this as an AI-assisted author audit, not as independent
      human annotation.

Signature: ______________________

---

## 8. Reference: 20 category codes

Use only these exact strings in `primary_category` and `secondary_category`.

**Content (15):**

`entity_misattribution`, `entity_conflation`, `entity_substitution`,
`entity_name_distortion`, `fabricated_entity`, `generic_or_descriptive_answer`,
`relation_inversion`, `attribute_error`, `numeric_or_temporal_error`,
`wrong_granularity`, `coreference_error`, `non_responsive_or_partial_answer`,
`unsupported_specificity`, `context_contradiction`, `answer_span_copy`.

**Artifact (5):**

`artifact_grounded_overflag`, `artifact_short_answer_fp`, `artifact_label_noise`,
`artifact_truncated_context`, `artifact_question_ambiguity`.

**Explanation labels:** `plausible`, `implausible`, `unsure`.

**Review statuses:** `needs_author_verification`, `author_verified`,
`author_corrected`.
