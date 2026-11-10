# WP5 Hallucination Error Codebook (AI-assisted draft)

**Status:** draft for author verification. Every label in `wp5_error_labels.csv` and
`wp5_explanation_labels.csv` carries `review_status = needs_author_verification`.
These labels are **not human annotations**, **not an independent review**, and **not a
two-annotator study**. The author must verify, correct, or reject each row before any
number from this package enters the manuscript.

**Generated:** 2026-11-11 on branch `rebuild`.

---

## 1. Purpose and scope

This codebook defines a 20-category taxonomy for the WP5 error audit. It replaces the
two heuristic tags in `src/models/error_analysis.py` (`label_ambiguity` and
`short_answer_ambiguity`) with categories that an author can apply consistently to a
question, a retrieved context, and a short answer.

The taxonomy has two families:

- **Answer-content categories (15).** The answer itself is wrong, incomplete, or
  non-responsive relative to the question and context.
- **Audit-artifact categories (5).** The answer is grounded, or the case cannot be
  judged, and the apparent error belongs to the detector, the benchmark label, or the
  data export. Artifact categories must never be reported as hallucination prevalence.

The unit of annotation is one `(question, context, answer, sample_id)` row. The
benchmark `label` column is a cross-check, not ground truth. The audit is allowed to
find label noise.

### Source tables

| Sheet | Source | Rows | Model / SHAP provenance |
|---|---|---|---|
| `wp5_error_labels.csv` | `artifacts/results/rebuild/fixed_hp/fixed_hp_predictions.parquet` (m3, seed 42, `halueval_test`), joined to `data/processed/qa_clean.parquet` for full text | 20 (8 false positives, 12 false negatives) | EC-XGB m3 fixed-hp seed 42 (`m3-fixed-hp-seed-42`), 35 features; **m3, not B2** |
| `wp5_explanation_labels.csv` | `artifacts/results/rebuild/explanations/m3_audit_cases.csv` | 26 rows (m3 audit roles: 7 FP, 12 FN, 3 borderline, 3 other, 1 TP) | EC-XGB m3 fixed-hp seed 42; exact TreeSHAP top-5 feature names, input values, and signed contributions in raw margin (log-odds) |

The m3 model is `artifacts/models/rebuild/fixed_hp/xgboost_m3_seed_42.joblib`; SHAP is
exact TreeSHAP via `pred_contribs=True` in raw margin (log-odds). The explanation
sheet's `top_features`, `top_values`, and `top_shap` columns come directly from
`m3_audit_cases.csv`.

In `wp5_error_labels.csv`, `label` is the benchmark gold label (0 = grounded,
1 = hallucinated), not the review verdict. The false-positive/false-negative
direction is stated at the start of each rationale. Eight rows are detector false
positives (`label = 0`, model flagged) and twelve are detector false negatives
(`label = 1`, model did not flag).

The 20 error cases are **all 8 m3 false positives** plus a purposive 12-case sample of
the 35 m3 false negatives. The 12 FNs were chosen to cover distinct codebook
categories, not to estimate category prevalence. The 26 explanation cases are the
existing B5 audit set (selected under B2) **re-scored under m3**; the case selection is
not an m3 prevalence sample. State both selection rules wherever the counts appear.

The m3 audit CSV contains `top_features`, `top_values`, and `top_shap` (signed TreeSHAP
contributions in raw margin). The rationales in `wp5_explanation_labels.csv` cite the
actual m3 top features and signs. They are an AI-drafted reading of the SHAP output;
verify them against `m3_audit_case_shap.parquet` before treating an explanation
rationale as final.

---

## 2. Integrity and provenance rules

1. **No human claim.** Do not write "two human annotators", "independent human
   review", "inter-annotator agreement", or "human-verified" unless a second human
   actually labels the sheet independently.
2. **AI passes are not raters.** The B5 columns `reviewer_1` and `reviewer_2` are two
   AI-expert passes, not two people. They are historical context for the 26-case set,
   not the labels in this package. The m3 explanation labels are a fresh AI draft; they
   do not reuse the B5 judgments as human evidence.
3. **Kappa needs a second rater.** No Cohen's kappa is computed in this package. The
   worksheet in `wp5_summary.md` is blank. Fill it only after a second human labels
   all cases without seeing the AI draft.
4. **Error-enriched, purposive samples.** The 20-case error sheet uses all 8 m3 false
   positives and 12 of the 35 m3 false negatives from a 3,000-row HaluEval test set.
   The 12 FNs are a codebook-coverage sample, not a random sample. The 26-case
   explanation sheet is the B5 audit set re-scored under m3, with m3 roles 7 FP,
   12 FN, 3 borderline, 3 other, 1 TP. Neither sheet estimates natural prevalence. Do
   not report category percentages as test-set rates.
5. **Full context.** The error sheet uses full text from
   `data/processed/qa_clean.parquet`. The explanation sheet uses the text stored with
   the m3 audit export (`data/processed/unified_records.parquet`, HaluEval rows). Join
   on `sample_id` before checking a case. Do not label from a truncated snippet.

---

## 3. Support test

Before assigning a category, answer one question: **is the answer supported by the
retrieved context and does it resolve the asked slot?**

Treat the answer as **supported** when all of the following hold:

- Every content-bearing element (entity, relation, number, date) is entailed by the
  context or is a faithful paraphrase of it.
- The answer resolves the slot the question asks for (a name for a name question, a
  number for a number question, a yes/no for a yes/no question).
- The answer does not add a specific detail that the context does not support.

Treat the answer as **not supported** when any of the following hold:

- The answer names a different entity, relation, number, or date.
- The answer gives a description or category where the question asks for a specific
  name or value.
- The answer addresses only part of a multi-part question, or answers a different
  question.
- The answer directly contradicts an explicit context statement.

Special cases:

- A correct paraphrase is supported. If the benchmark labels it hallucinated, use
  `artifact_label_noise`.
- A description that uniquely identifies the entity but does not name it is a form
  problem, not full support for a name-seeking question. Use
  `generic_or_descriptive_answer`, and note in the rationale that the description is
  true.
- A context that is cut off or malformed is not evidence for or against the answer.
  Use `artifact_truncated_context` and set confidence at most 3.
- World knowledge does not override the retrieved context. If the answer is true in
  the world but unsupported by the context, code the answer-side defect and note the
  external fact in the rationale only as a caveat.

---

## 4. Primary and secondary label rule

Every row has **exactly one primary category**. `secondary_category` is optional and
blank when no second independent defect is present.

**Primary category.** Choose the defect that most directly explains why the answer is
wrong or non-responsive. When the answer is grounded and the model raised a false
alarm, the primary category is the detector artifact, because there is no answer-side
defect.

**Secondary category.** Use it only for a second, independently identifiable defect.
Do not use it for confidence, severity, or a restatement of the primary defect. A
secondary label should tell the author something the primary label does not.

**Precedence:**

1. Answer grounded, model flagged it → artifact primary
   (`artifact_grounded_overflag` or `artifact_short_answer_fp`).
2. Answer grounded, benchmark label says hallucinated → `artifact_label_noise`
   primary.
3. Answer not grounded → content primary. If a detector artifact also applies, the
   artifact goes to secondary.
4. Two content defects → the cause is primary and the consequence is secondary
   (for example `entity_conflation` → `context_contradiction`). If neither is a
   cause, choose the defect that changes the truth value most and put the other in
   secondary.
5. Context incomplete or malformed → `artifact_truncated_context` may be primary only
   when the answer cannot be judged; otherwise it is secondary.
6. Question genuinely admits more than one answer form → `artifact_question_ambiguity`
   may be primary only when no single convention resolves it; otherwise it is
   secondary. Do not use it to avoid a hard call.

**Do not** use a content category for a grounded answer merely because the answer is
verbose. Use `answer_span_copy` as a secondary form note, not as evidence of
hallucination.

---

## 5. Confidence scale

| Value | Meaning |
|---:|---|
| 5 | Clear from the text; no reasonable alternative reading. |
| 4 | Likely; a minor ambiguity or a small inferential step. |
| 3 | Two readings are possible; one is preferred. |
| 2 | Heavily ambiguous, context incomplete, or the answer form is contestable. |
| 1 | Cannot judge from the available text; use only with `artifact_truncated_context`. |

Confidence is about the label, not about how interesting the case is. A confidence of
5 on `artifact_label_noise` means the context clearly supports the answer, not that
the benchmark is certainly wrong for every purpose.

---

## 6. Labelling workflow

1. Join the error CSV row to `data/processed/qa_clean.parquet` on `sample_id`. Join
   the explanation CSV row to `artifacts/results/rebuild/explanations/m3_audit_cases.csv`
   (and `data/processed/unified_records.parquet` for full text).
2. Read the **full** question, context, and answer. Do not label from the answer
   snippet alone.
3. Apply the support test in §3.
4. If supported: choose `artifact_grounded_overflag`, `artifact_short_answer_fp`, or
   `artifact_label_noise`. If the case is unjudgeable, choose
   `artifact_truncated_context` or `artifact_question_ambiguity`.
5. If not supported: choose one content category from §8. Prefer the most specific
   observable defect over a general one.
6. Add a secondary category only if a second independent defect is visible.
7. Write a rationale that quotes or paraphrases the decisive context span. State
   whether the case is a detector false positive or false negative when that is
   relevant.
8. Set confidence using §5.
9. Leave `review_status = needs_author_verification` until the author has checked the
   row. After checking, the author may change it to `author_verified` or
   `author_corrected` and record the date in the checklist.

---

## 7. Quick reference

| # | Code | Family | One-line definition |
|---:|---|---|---|
| 1 | `entity_misattribution` | content | A real entity is placed in the wrong slot or relation. |
| 2 | `entity_conflation` | content | Two entities or name parts are merged into one name that matches neither. |
| 3 | `entity_substitution` | content | A plausible same-class entity replaces the correct one. |
| 4 | `entity_name_distortion` | content | The right entity is named with an altered, non-standard string. |
| 5 | `fabricated_entity` | content | A proper name or title is invented and has no support. |
| 6 | `generic_or_descriptive_answer` | content | A description or category is given instead of the requested name/value. |
| 7 | `relation_inversion` | content | The answer supplies the converse or a different argument role. |
| 8 | `attribute_error` | content | The right entity carries a wrong property (nationality, genre, role). |
| 9 | `numeric_or_temporal_error` | content | A wrong number, quantity, date, or ordering is given. |
| 10 | `wrong_granularity` | content | The answer is at the wrong level (album instead of single, capacity instead of population). |
| 11 | `coreference_error` | content | A pronoun or description resolves to the wrong antecedent. |
| 12 | `non_responsive_or_partial_answer` | content | The answer evades the question or addresses only part of it. |
| 13 | `unsupported_specificity` | content | The core answer is close, but an extra specific detail is unsupported. |
| 14 | `context_contradiction` | content | The answer directly contradicts an explicit context statement. |
| 15 | `answer_span_copy` | content | A long verbatim span is copied instead of extracting the asked value. |
| 16 | `artifact_grounded_overflag` | artifact | A grounded answer is scored above the threshold; detector false positive. |
| 17 | `artifact_short_answer_fp` | artifact | A correct one-word/short answer is over-flagged because of length/overlap. |
| 18 | `artifact_label_noise` | artifact | The benchmark label is opposite to clear contextual support. |
| 19 | `artifact_truncated_context` | artifact | The context is cut off or malformed, so the case cannot be judged. |
| 20 | `artifact_question_ambiguity` | artifact | The question admits more than one defensible answer form. |

---

## 8. Category definitions

### 1. `entity_misattribution`

**Definition.** A real, identifiable entity is placed in the wrong slot or relation.
The name itself is not invented or blended; the attachment is wrong. The entity may
appear in the context or be a real-world entity, but it is not the entity the question
asks for.

**Positive example.** `q_4514_hallucinated`: the question asks which brewery was
established in 2000 by the original brewers of Matilda Bay. The context says Little
Creatures is that brewery. The answer, `"Matilda Bay Brewing Company."`, names the
prior company, a real brewery but the wrong slot.

**Negative example.** `q_7876_correct`: the question asks which U.S. state the
Hanford Site is in. The answer, `"the U.S. state of Washington."`, is the correct
state. The entity and the relation both match.

**Decision rule.** Use when the answer names a real entity that belongs to a different
slot. If the entity is a blend of two names, use `entity_conflation`. If the entity is
a plausible same-class substitute with no special association, use
`entity_substitution`. If the relation itself is reversed, use `relation_inversion`.

---

### 2. `entity_conflation`

**Definition.** Two distinct entities, or parts of two names, are merged into one name
or referent that matches neither source entity.

**Positive example.** `q_3058_hallucinated`: the context says *Can't Hardly Wait*
was written and directed by Deborah Kaplan and Harry Elfont. The answer,
`"Harry Kaplan."`, combines Harry Elfont's forename with Kaplan's surname and matches
neither writer.

**Negative example.** `q_4156_correct`: the answer,
`"2010 Missouri Tigers football team"`, is an exact entity from the context. No parts
of two entities are merged.

**Decision rule.** Use when both source entities are identifiable and the answer
combines them. If the answer invents a wholly new name with no source parts, use
`fabricated_entity`. If it only adds a word to a real name, use
`entity_name_distortion`.

---

### 3. `entity_substitution`

**Definition.** The correct entity is replaced by a different, plausible member of the
same class, often a distractor from the question or a neighboring entity in the
context. The substitute is a real entity and is not a blend.

**Positive example.** `q_2910_hallucinated`: the context says the episode aired
on AMC. The answer, `"Syfy"`, is another cable network, a same-class substitute that
contradicts the context.

**Negative example.** `q_4329_correct`: the answer,
`"2007 UEFA Champions League Final"`, is the exact match named in the context. There
is no substitution.

**Decision rule.** Use when the substitute is same-class and no explicit role
inversion or association explains the choice. If the substitute is associated with
the correct entity (a former company, a teammate, a family member), use
`entity_misattribution`. If the substitute is a containing or contained entity, use
`wrong_granularity`.

---

### 4. `entity_name_distortion`

**Definition.** The correct entity is recognizably intended, but its name is altered by
adding, dropping, or swapping a word, producing a non-standard or nonexistent variant.

**Positive example.** `q_8501_hallucinated`: the context names "The G7 Research
Group" and separately mentions "the G7 Summit." The answer,
`"The G7 Summit Research Group"`, inserts "Summit" into the group's name.

**Negative example.** `q_7876_correct`: `"the U.S. state of Washington."` is the
name used in the context, with no added or dropped words.

**Decision rule.** Use when the referent is the correct entity but the string is not
the name used in the context. If the altered name refers to a different real entity,
use `entity_substitution`. If the extra word is a separate unsupported claim, use
`unsupported_specificity` as secondary.

---

### 5. `fabricated_entity`

**Definition.** The answer invents a proper-noun entity, title, or name that has no
support in the context and does not correspond to a real entity in the asked slot.

**Positive example.** X16 / `q_2184_hallucinated`: the question asks for the title of
the book written by the host of *Family Feud*. The context names Steve Harvey's book
*Act Like a Lady, Think Like a Man*. The answer, `"The Lady and the Man"`, is a
fabricated title close to the real one.

**Negative example.** `q_2251_hallucinated`: the answer,
`"Tyler Winklevoss founded ConnectU."`, names a real person who is explicitly listed
as a founder in the context. It is not fabricated.

**Decision rule.** Use for invented proper nouns. If the string is a distorted real
name, use `entity_name_distortion`. If it is a generic description, use
`generic_or_descriptive_answer`. If it is a real entity in the wrong slot, use
`entity_misattribution` or `entity_substitution`.

---

### 6. `generic_or_descriptive_answer`

**Definition.** The answer gives a generic category, description, or paraphrase
instead of the specific name or value requested. It may be true, but it does not
resolve the question's slot with the expected answer form.

**Positive example.** `q_1702_hallucinated`: the question asks for the name of
the engagement on the northern coast of France on 19 August 1942. The context names
the Dieppe Raid. The answer, `"Northern France Battle"`, is a generic label, not the
name.

**Negative example.** `q_4329_correct`: the answer,
`"2007 UEFA Champions League Final"`, is the specific name requested.

**Decision rule.** Use when the answer is a description or category rather than a
proper name or exact value. If the answer is a partial name plus a description,
consider `non_responsive_or_partial_answer`. If the description is the only correct
way to identify the entity and the benchmark accepts it, use
`artifact_question_ambiguity` instead.

---

### 7. `relation_inversion`

**Definition.** The answer supplies the converse or a different argument role of the
asked relation (for example, operator instead of acquirer, member instead of founder,
company instead of producer).

**Positive example.** `q_7398_hallucinated`: the question asks who directed the 2009
British thriller *Knife Edge*. The context says it was "directed by Anthony Hickox and
starring Natalie Press"; the answer, `"Natalie Press directed Knife Edge."`, swaps the
director and star roles.

**Negative example.** `q_7876_correct`: the answer,
`"the U.S. state of Washington."`, states the correct location relation for the
Hanford Site.

**Decision rule.** Use when the question's relation has a clear direction and the
answer swaps it. If the answer names a different entity in the same role, use
`entity_misattribution`. If the answer only gives a description, use
`generic_or_descriptive_answer`. If the answer is a verbatim span that contains both
roles, use `answer_span_copy` as primary and note the role confusion.

---

### 8. `attribute_error`

**Definition.** The correct entity is identified, but a property (nationality, genre,
role, affiliation) is wrong.

**Positive example.** X24 / `q_8342_hallucinated`: the question asks which Brazilian
Bayern signed before the 2012-13 season. The context identifies Dante as the Brazilian
who plays for Nice. The answer, `"Bayern signed Claudio Pizarro."`, names a real Bayern
signing, but Pizarro is Peruvian. The entity is right; the nationality attribute is
wrong.

**Negative example.** `q_4156_correct`: the answer,
`"2010 Missouri Tigers football team"`, matches the team and its attributes in the
context.

**Decision rule.** Use when the entity is right and a property is wrong. If the entity
itself is wrong, use `entity_misattribution` or `entity_substitution`. If the property
is a number or date, use `numeric_or_temporal_error`.

---

### 9. `numeric_or_temporal_error`

**Definition.** The answer gives a wrong number, quantity, date, year, or ordering, or
answers a numeric or temporal question with a non-numeric value.

**Positive example.** X12 / `q_1921_hallucinated`: the question asks the 2010
population of the city where the Kansas Jayhawks played in 1927. The context says the
city's population was 87,643 and the stadium capacity was 35,000. The answer,
`"Memorial Stadium capacity was 35,000."`, gives a different quantity.

**Negative example.** `q_3297_hallucinated`: the answer,
`"Fred Phelps Sr. died in 2014."`, matches the context dates (November 13, 1929 to
March 19, 2014).

**Decision rule.** Use when the asked slot is numeric or temporal and the answer
supplies a different quantity/date or a non-numeric substitute. If the number is right
but the unit or referent is wrong, code `wrong_granularity` as primary and this as
secondary.

---

### 10. `wrong_granularity`

**Definition.** The answer gives a containing or contained entity, a broader or
narrower unit, or a different referent at the wrong level of granularity. Examples:
album instead of single, island instead of city, stadium capacity instead of city
population.

**Positive example.** X15 / `q_1739_hallucinated`: the question asks which single by
Mobb Deep features 50 Cent. The context says "Creep" is the single, from the album
*Blood Money*. The answer, `"\"Blood Money\" features 50 Cent."`, gives the album.

**Negative example.** `q_4329_correct`: the answer,
`"2007 UEFA Champions League Final"`, is the match at the granularity the question
asks for.

**Decision rule.** Use when the answer is from the right domain but the wrong level or
containment. If the answer is a different same-class entity, use
`entity_substitution`. If the answer is a number for a different quantity, use
`numeric_or_temporal_error` as primary and this as secondary.

---

### 11. `coreference_error`

**Definition.** A pronoun or definite description in the answer resolves to the wrong
antecedent, or the answer leaves the antecedent unresolved between two candidates.

**Positive example.** X10 / `q_4539_hallucinated`: the context lists Tim Duckworth,
Billy Hamilton, and Jason Campbell as natives, then says "He also holds the Cincinnati
Reds record for most stolen bases by a rookie in a season." The pronoun is ambiguous;
the answer, `"Jason Campbell holds the record."`, picks the wrong athlete (the record
holder is Billy Hamilton).

**Negative example.** `q_7876_correct`: the answer,
`"the U.S. state of Washington."`, has an explicit referent and no pronoun ambiguity.

**Decision rule.** Use when the error is traceable to pronoun or description
resolution. If the answer simply names the wrong entity without a pronoun, use
`entity_misattribution`. If the question itself is ambiguous, use
`artifact_question_ambiguity` as secondary.

---

### 12. `non_responsive_or_partial_answer`

**Definition.** The answer addresses a different question, evades the asked slot, or
answers only one component of a multi-part, compound, or comparison question.

**Positive example.** `q_4896_hallucinated`: the question asks whether both
*King Roger* and *Measure for Measure* were written by William Shakespeare. The answer,
`"King Roger is a Polish play."`, gives a genre statement about one work and never
answers the yes/no comparison.

**Negative example.** `q_7742_correct`: the answer,
`"The Patient Protection and Affordable Care Act"`, directly fills the single slot the
question asks for.

**Decision rule.** Use when the answer does not engage the asked relation, or engages
only part of it. If the answer gives a description of the right entity, use
`generic_or_descriptive_answer`. If the answer gives the wrong entity for one
component, code the entity category as primary and this as secondary.

---

### 13. `unsupported_specificity`

**Definition.** The answer adds a specific detail (a word, number, qualifier, or
sub-claim) that the context does not support, even if the core answer is close.

**Positive example.** `q_8501_hallucinated`: the context names "The G7 Research
Group" and separately mentions "the G7 Summit." The answer,
`"The G7 Summit Research Group"`, adds "Summit" as if it were part of the group's
name.

**Negative example.** `q_7876_correct`: every element of
`"the U.S. state of Washington."` appears in the context.

**Decision rule.** Use when the core entity is right but an extra specific element is
unsupported. If the extra element creates a different real entity, use
`entity_name_distortion` or `entity_substitution`. If the whole answer is invented,
use `fabricated_entity`.

---

### 14. `context_contradiction`

**Definition.** The answer directly contradicts an explicit statement in the context.

**Positive example.** `q_2910_hallucinated`: the context says the episode aired
on AMC. The answer, `"Syfy"`, names a different network and directly contradicts the
context.

**Negative example.** `q_7876_correct`: the answer,
`"the U.S. state of Washington."`, is consistent with the context sentence about the
Hanford Site.

**Decision rule.** Use when the context states the opposite or an alternative and the
answer conflicts with it. If the context is merely silent, use `fabricated_entity` or
`unsupported_specificity`. If the answer is a different same-class entity without an
explicit contradiction, use `entity_substitution`.

---

### 15. `answer_span_copy`

**Definition.** The answer copies a long verbatim context span instead of extracting
the specific value the question asks for. The copied span may contain the answer, but
it does not isolate it.

**Positive example.** `q_3027_correct`: the answer is an 80-word verbatim span
that begins "The JKWI was the designated operator of the Irvine, Warren, Kane &
Johnsonburg Railroad..." and continues through the partnership and acquisition
details. It does not reduce to the company that acquired the railroad.

**Negative example.** `q_8518_correct`: the answer, `"Australia"`, is a minimal
extraction, not a copied span.

**Decision rule.** Use when the answer is a long verbatim or near-verbatim context
span. This is an answer-form defect, not by itself evidence of hallucination. If the
span is grounded, pair it with `artifact_grounded_overflag` as secondary. If the span
contains a wrong relation or entity, code that content category as primary and
`answer_span_copy` as secondary.

---

### 16. `artifact_grounded_overflag`

**Definition.** The answer is fully supported by the context, or is a correct
paraphrase, but the model assigns a risk score above the decision threshold. The error
is in the detector or calibration, not in the answer.

**Positive example.** `q_7742_correct`: the answer,
`"The Patient Protection and Affordable Care Act"`, is verbatim in the context and
`overlap_answer_context = 1.0`, yet the m3 raw score is 0.52 (just above the 0.5
threshold).

**Negative example.** `q_2910_hallucinated`: the answer, `"Syfy"`, contradicts
the context's "AMC". The low score is appropriate; this is not a grounded overflag.

**Decision rule.** Use for false positives where the answer is grounded. If the answer
is a correct one-word answer with little or no lexical overlap, use
`artifact_short_answer_fp` as the more specific category. If the answer is grounded
and the benchmark label is 1, use `artifact_label_noise`.

---

### 17. `artifact_short_answer_fp`

**Definition.** A correct short answer, often one word, with little or no lexical
overlap is over-flagged because length and overlap features dominate the score.

**Positive example.** `q_8518_correct`: the answer, `"Australia"`, is supported
by "Australian hard rock band AC/DC", but it has one word and zero lexical overlap;
the m3 raw score is 0.80.

**Negative example.** `q_7876_correct`: the answer,
`"the U.S. state of Washington."`, is longer and shares content words with the
context, so brevity is not the main artifact.

**Decision rule.** Use when the answer is correct, at most four words, and the top
features are `n_words`, `n_chars`, or `overlap_answer_context`. If the answer is longer
but still grounded, use `artifact_grounded_overflag`. If the short answer is wrong,
use the relevant content category.

---

### 18. `artifact_label_noise`

**Definition.** The benchmark label says hallucinated or correct, but the answer is
clearly supported or contradicted by the context. The error is in the benchmark label,
not in the model or the answer.

**Positive example.** `q_2251_hallucinated`: the answer,
`"Tyler Winklevoss founded ConnectU."`, is fully supported: the context lists Tyler
Winklevoss as a founder and describes him as "an American rower". The dataset label is
1, but the answer is correct.

**Negative example.** `q_2910_hallucinated`: the answer, `"Syfy"`, contradicts
the context's "AMC". The label 1 is correct; this is not noise.

**Decision rule.** Use only when the context clearly supports the answer and the
benchmark label is opposite. Quote the supporting span in the rationale. If the answer
is a correct paraphrase but the benchmark expects a different answer form, use
`artifact_question_ambiguity` instead.

---

### 19. `artifact_truncated_context`

**Definition.** The retrieved context is cut off, malformed, or missing the sentence
needed to verify the answer, so the case cannot be judged from the available text.

**Positive example.** `q_4896_hallucinated`: the context is 138 characters and
reads "King Roger (Polish: \"Król Roger\" , Op.Measure for Measure is a play by William
Shakespeare..." The sentence about King Roger is cut before its genre is stated, so the
answer's "Polish play" cannot be checked against the context.

**Negative example.** `q_7742_correct`: the context is complete and contains the
answer verbatim.

**Decision rule.** Use when the context itself is incomplete or malformed. Do not use
it merely because the answer is long. If full text is available in
`data/processed/qa_clean.parquet`, re-export and re-label before keeping this category.

---

### 20. `artifact_question_ambiguity`

**Definition.** The question has more than one defensible reading or answer form, so
the label depends on an unstated convention (for example, name versus description,
person versus company, single versus album).

**Positive example.** `q_6989_hallucinated`: the question asks which air
transport hub Russell Williams commanded. The answer,
`"The Canadian military's largest airbase."`, is a true description of CFB Trenton but
does not name it. Whether the description counts as an answer depends on the
answer-form rule.

**Negative example.** `q_4156_correct`: the question asks which team represented
the University of Missouri in the 2010 NCAA Division I FBS football season. The answer,
`"2010 Missouri Tigers football team"`, is the single clear answer.

**Decision rule.** Use when the ambiguity is in the question or the answer-form
contract, not in the evidence. If the context is incomplete, use
`artifact_truncated_context`. If the benchmark label is clearly opposite to the
evidence, use `artifact_label_noise`.

---

## 9. Relationship to the old B2 heuristic tags

`src/models/error_analysis.py` used two heuristic tags (`label_ambiguity` and
`short_answer_ambiguity`) on the old B2 error export. That export is not the source of
the m3 sheet. The m3 error sheet is a fresh 8-FP + 12-FN sample from
`fixed_hp_predictions.parquet`, and every row is re-coded with the 20-category
codebook. Do not merge the old B2 tags into the m3 counts.

If a paper table needs a historical crosswalk, describe it as a B2-to-m3 taxonomy
mapping, not as a relabel of the same cases.

---

## 10. Kappa and second rater

Do not compute Cohen's kappa from this package alone. The AI draft is not a rater.
A second human must label the 26 explanation cases independently, without seeing the
AI labels, before any agreement statistic is meaningful. The blank 3x3 worksheet and
the collapse-to-content/artifact 2x2 worksheet are in `wp5_summary.md`.

If a second rater is not available, report the audit as "AI-assisted author review"
and do not report kappa. The paper can report the label counts and the author's
verification status, but not inter-rater agreement.

---

## 11. Change log

| Date | Change |
|---|---|
| 2026-11-11 | Initial AI-assisted draft. 20 categories (15 content, 5 artifact). 20 error cases and 26 explanation cases labelled; all `needs_author_verification`. |
| 2026-11-11 | Repointed to m3 fixed-hp seed 42. Error sheet redrawn as all 8 m3 FPs + 12 m3 FNs; explanation sheet rewritten from `m3_audit_cases.csv` with signed TreeSHAP. All 20 categories now appear as a primary or secondary label. |
