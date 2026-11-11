# P0 Annotation Codebook v2

**Status:** AI pre-annotation for three-author blinded review. Every row in the P0
sheets starts at `review_status = needs_author_verification`. No row is a verified
label until an author has read the full case and changed the status. These sheets are
**not** a completed annotation study, **not** an agreement study, and **not** evidence
that the benchmark labels are correct.

**Generated:** 2026-11-11 on branch `rebuild`.

**Package:** `docs/rebuild/p0/` — this codebook, three annotation sheets, a blind
overlap sheet, a review guide, and `annotation_manifest.json`.

---

## 1. What this package annotates

The P0 package supports three separate judgements. Do not collapse them into one
score.

| Sheet | Unit | Question the annotator answers |
|---|---|---|
| `p0_error_taxonomy.csv` | one `(question, context, answer, sample_id)` case | What kind of error is this, if any? Is the benchmark label right? |
| `p0_explanation_audit.csv` | one scored case plus its m3 TreeSHAP terms | Do the top signed feature contributions give a plausible account of the score and the evidence? |
| `p0_claim_verdicts.csv` | one atomic claim from one answer | Is this claim supported, contradicted, or unsupported by the retrieved evidence? |

The three judgements use different label sets. `primary_category` is not a claim
verdict, and `plausible` is not a statement that the answer is correct.

### 1.1 Integrity rules

1. **The benchmark label is a cross-check, not ground truth.** `label` is the dataset
   label (0 = grounded, 1 = hallucinated). It may be wrong. The audit is allowed to
   find label noise, and an AI or author verdict that disagrees with `label` is a
   finding, not an error to suppress.
2. **Never edit `label` to make a case agree with the model.** Keep the benchmark
   value. Record disagreement in `author_label` and `final_label`, and explain it in
   `notes`.
3. **AI suggestions are not author labels.** `ai_suggested`, `ai_rationale`, and
   `ai_confidence_1to5` are a pre-annotation. They are not a rater, not a second
   opinion, and not evidence of agreement.
4. **One case, one primary category.** `secondary_category` is optional and must
   identify a second independent defect. It is not a severity or confidence field.
5. **Snippets are for orientation only.** The taxonomy question, context, and answer
   snippets and the explanation answer snippet are at most 160 characters. Read the
   full question, context, and answer before confirming a label. The guide gives a
   one-line command for printing the full case.
6. **No invented provenance.** Do not describe this package as completed annotation,
   as a two-rater study, or as agreement between people. The blind sheet becomes an
   agreement study only after all three authors have labelled it independently.

---

## 2. The 20 error categories

The taxonomy has 15 answer-content categories and 5 audit-artifact categories. Use
the exact codes below. Artifact categories describe the detector, the benchmark label,
or the data export. They must never be reported as hallucination prevalence.

| # | Code | Family | One-line definition |
|---:|---|---|---|
| 1 | `entity_misattribution` | content | A real entity is placed in the wrong slot or relation. |
| 2 | `entity_conflation` | content | Two entities or name parts are merged into a name that matches neither. |
| 3 | `entity_substitution` | content | A plausible same-class entity replaces the correct one. |
| 4 | `entity_name_distortion` | content | The right entity is named with an altered, non-standard string. |
| 5 | `fabricated_entity` | content | A proper name or title is invented and has no support. |
| 6 | `generic_or_descriptive_answer` | content | A description or category is given instead of the requested name or value. |
| 7 | `relation_inversion` | content | The answer supplies the converse or a different argument role. |
| 8 | `attribute_error` | content | The right entity carries a wrong property such as nationality, genre, or role. |
| 9 | `numeric_or_temporal_error` | content | A wrong number, quantity, date, or ordering is given. |
| 10 | `wrong_granularity` | content | The answer is at the wrong level, such as album instead of single. |
| 11 | `coreference_error` | content | A pronoun or description resolves to the wrong antecedent. |
| 12 | `non_responsive_or_partial_answer` | content | The answer evades the question or addresses only part of it. |
| 13 | `unsupported_specificity` | content | The core answer is close, but an extra specific detail is unsupported. |
| 14 | `context_contradiction` | content | The answer directly contradicts an explicit context statement. |
| 15 | `answer_span_copy` | content | A long verbatim span is copied instead of extracting the asked value. |
| 16 | `artifact_grounded_overflag` | artifact | A grounded answer is scored above the threshold. |
| 17 | `artifact_short_answer_fp` | artifact | A correct short answer is over-flagged because of length or overlap features. |
| 18 | `artifact_label_noise` | artifact | The benchmark label is opposite to clear contextual support. |
| 19 | `artifact_truncated_context` | artifact | The context is cut off or malformed, so the case cannot be judged. |
| 20 | `artifact_question_ambiguity` | artifact | The question admits more than one defensible answer form. |

### 2.1 Category boundaries that matter most

Use these contrasts when two codes both look plausible.

- **`entity_misattribution` vs `entity_substitution`.** Misattribution means the named
  entity is associated with the correct entity or slot in some other way (a prior
  company, a teammate, a family member). Substitution means a same-class entity with
  no special association replaces the answer.
- **`entity_conflation` vs `fabricated_entity`.** Conflation requires two identifiable
  source names whose parts are combined. A wholly new name is fabricated even when it
  sounds plausible.
- **`entity_name_distortion` vs `unsupported_specificity`.** Distortion changes the
  name of the intended entity. Unsupported specificity adds a separate detail while the
  core answer stays correct.
- **`generic_or_descriptive_answer` vs `artifact_question_ambiguity`.** A description is
  a content defect when the question clearly asks for a name. Ambiguity applies only
  when no single answer-form convention resolves the question.
- **`wrong_granularity` vs `entity_substitution`.** Granularity means containment or
  level (album/single, stadium capacity/city population). Substitution means a sibling
  entity at the same level.
- **`context_contradiction` vs `entity_substitution`.** When the answer replaces a
  same-class entity and that replacement contradicts an explicit context sentence,
  code the cause first (`entity_substitution`) and the contradiction second.
- **`answer_span_copy` is a form defect, not proof of hallucination.** A grounded span
  gets `artifact_grounded_overflag` as primary and `answer_span_copy` as secondary. A
  span that contains a wrong relation gets the content code as primary and
  `answer_span_copy` as secondary.

### 2.2 Primary and secondary rules

1. Choose exactly one primary category: the defect that most directly explains why the
   answer is wrong or non-responsive.
2. If the answer is grounded and the model flagged it, the primary category is the
   artifact, because there is no answer-side defect.
3. If the answer is grounded and the benchmark says hallucinated, `artifact_label_noise`
   is primary.
4. If two content defects are present, the cause is primary and the consequence is
   secondary. If neither is a cause, the defect that changes the truth value most is
   primary.
5. `artifact_truncated_context` may be primary only when the answer cannot be judged.
   Otherwise it is secondary.
6. `artifact_question_ambiguity` may be primary only when no convention resolves the
   question. Do not use it to avoid a hard call.
7. Never use `answer_span_copy` as the only label for a grounded answer merely because
   the answer is verbose.

### 2.3 Selection rule and the false-positive caveat

The taxonomy sheet contains 25 benchmark-negative cases (`label = 0`) and 25
benchmark-positive cases (`label = 1`) from the m3 seed-42 HaluEval test scores. The
negative half is the 25 highest-scoring `label = 0` rows. **Only 8 of those 25 cross the
0.5 decision threshold**; the other 17 are near-threshold high-risk grounded cases
(scores 0.272-0.493). The positive half is the 25 lowest-scoring `label = 1` rows
(scores 0.004-0.312). The sheet is a rank-defined error-enriched sample, not a
threshold confusion matrix. Each rationale states whether the case is a threshold false
positive, a near-threshold false-positive candidate, or a false negative. Do not compute
precision or recall from this sheet.

---

## 3. Support test (applies to every sheet)

Before assigning any category or verdict, answer one question: **is the answer
supported by the retrieved context, and does it resolve the asked slot?**

Treat the answer as **supported** only when all of these hold:

- Every content-bearing element (entity, relation, number, date) is entailed by the
  context or is a faithful paraphrase.
- The answer resolves the slot the question asks for (a name for a name question, a
  number for a number question, a yes/no for a yes/no question).
- The answer adds no specific detail that the context does not support.

Treat the answer as **not supported** when any of these hold:

- It names a different entity, relation, number, or date.
- It gives a description or category where the question asks for a specific name or
  value.
- It addresses only part of a multi-part question, or answers a different question.
- It directly contradicts an explicit context statement.

Special cases:

- A correct paraphrase is supported. If the benchmark calls it hallucinated, record
  `artifact_label_noise`.
- A description that uniquely identifies an entity but does not name it is a form
  defect, not full support for a name-seeking question. Use
  `generic_or_descriptive_answer` and note that the description is true.
- A context that is cut off or malformed is not evidence for or against the answer. Use
  `artifact_truncated_context` and set confidence at most 3.
- World knowledge never overrides the retrieved context. If the answer is true in the
  world but unsupported by the context, code the answer-side defect and put the
  external fact in the rationale only as a caveat.

---

## 4. Plausibility rubric for the explanation sheet

`ai_suggested` on `p0_explanation_audit.csv` is one of `plausible`, `implausible`, or
`unsure`. It judges the **explanation**, not the answer.

**Definition.** An explanation is plausible when the top signed TreeSHAP terms are
consistent with the evidence and with the model's decision. The terms do not need to be
causally sufficient. They must point in a direction an author can defend from the text.

Use the following ordered test:

1. Decide whether the answer is grounded or not from the context (Section 3).
2. Decide whether the model flagged the answer (`raw_score >= 0.5`).
3. If the answer verdict and the model decision agree, ask whether the dominant signed
   terms are evidence-bearing: context overlap, context contradiction, entailment,
   entity overlap, number novelty, or claim-level NLI. If yes and the sign is correct,
   the explanation is **plausible**.
4. If the decision is right but the dominant terms are question similarity, answer
   length, or character count, the explanation is **unsure** when the score is near the
   threshold and **implausible** when the non-evidence terms clearly drive the decision.
5. If the answer verdict and the model decision disagree, the explanation is
   **implausible** unless the score is within 0.05 of the threshold and the terms are
   mixed, in which case use **unsure**.
6. If the context cannot support any judgement, use **unsure** and say why.

Sign convention: TreeSHAP values are in raw margin (log-odds). A positive value pushes
toward hallucinated; a negative value pushes toward grounded.

### 4.1 Worked example: plausible

`P0X001` / `q_1402_correct`. The answer `"Walt Disney World Conference Center"` is
verbatim in the context. The score is 0.401 (below threshold). The largest term is
`overlap_answer_context = 1` at `-3.41`, with contradiction `+0.85` and question
similarity `+0.65` too small to flip it. The top term is a genuine support signal and
points the right way. **plausible**, confidence 4.

### 4.2 Worked example: implausible

`P0X009` / `q_3061_correct`. The answer `"portrait"` is supported by "he painted more
than 400 portraits". The score is 0.864 (flagged). The largest term is
`overlap_answer_context = 0` at `+2.70`, a zero-overlap penalty; `n_words = 0.69` adds
`-1.65` but does not rescue it. The explanation says "no lexical overlap", not "this
answer is unsupported". **implausible**, confidence 5.

### 4.3 Worked example: unsure

`P0X021` / `q_7742_correct`. The answer is the statute name and is verbatim in the
context. The score is 0.515, barely above threshold. `overlap_answer_context = 1`
contributes `-2.96` in the correct direction, but question overlap `+0.77`, length
`+0.63`, cosine `+0.52`, and a small contradiction `+0.47` push it over. The evidence
term supports the answer while the non-evidence terms decide the flag. **unsure**,
confidence 3.

### 4.4 Plausibility labels at a glance

| Label | Use when |
|---|---|
| `plausible` | The top signed terms are evidence-bearing and consistent with the score and the text. |
| `implausible` | The top terms are lexical shortcuts, or the score and the evidence disagree, so the explanation misdescribes the case. |
| `unsure` | The score is borderline, the top terms are mixed, or the context cannot settle the answer. |

---

## 5. Claim-verdict rubric

`ai_suggested` on `p0_claim_verdicts.csv` is one of `supported`, `contradicted`,
`unsupported`, or `unsure`. The unit is one atomic claim and one evidence sentence.

| Verdict | Definition | Typical cue |
|---|---|---|
| `supported` | The evidence entails the claim, including faithful paraphrase and numeric/date match. | The evidence sentence states the same relation with the same arguments. |
| `contradicted` | The evidence states the opposite or a conflicting value for the same slot. | Different number, date, entity, or reversed relation. |
| `unsupported` | The evidence is relevant but does not entail the claim; the claim adds a detail the evidence does not state. | The evidence mentions the topic but is silent on the asserted relation. |
| `unsure` | The evidence sentence is off-topic, truncated, or ambiguous, so the claim cannot be judged. | The retrieved sentence shares no content words or is a fragment. |

Rules:

1. Judge the claim against the **evidence sentence shown**, not against world knowledge
   and not against the full context. The evidence sentence is the one the NLI step
   selected. If it is the wrong sentence, the claim is `unsure` or `unsupported`, not
   `supported` by something you remember.
2. A claim can be true in the world and still be `unsupported` if the evidence does not
   state it.
3. A claim that only restates the question is not supported. Mark it `unsupported`.
4. A claim with a wrong number, date, or named entity is `contradicted` when the
   evidence gives the correct value, and `unsupported` when the evidence is silent.
5. `unsure` is a real verdict. Use it when the evidence is a fragment, a list marker, or
   an unrelated sentence. Do not force a binary call.

### 5.1 Worked example: supported

`P0C003` / `ragtruth:12445`. Claim: "Essential amino acids are those that cannot be made
by the body and must be obtained from the diet." Evidence: "Nonessential amino acids can
be made by the body, while essential amino acids cannot be made by the body so you must
get them from your diet." The evidence states the same relation in the same direction.
**supported**, confidence 5.

### 5.2 Worked example: contradicted

`P0C042` / `ragtruth:15008`. Claim: "The split of the developing zygote into two embryos
usually takes place between the eighth and ninth day of fertilization." Evidence:
"Conjoined twins: When the division of the developing zygote into 2 embryos occurs, 99%
of the time it is within 8 days of fertilization." The evidence gives a different
boundary for the same event. **contradicted**, confidence 4.

### 5.3 Worked example: unsure (and why not unsupported)

`P0C001` / `ragtruth:11908`. Claim: "with the highest average pay in Alaska ($23.70 per
hour or $49,400 per year) and the lowest average pay in Mississippi ($18.60 per hour or
$38,900 per year)." Evidence returned: "passage 3:104 months ago." The evidence is a
timestamp fragment, not a proposition. The claim may be supported elsewhere in the
passages, but this sentence cannot decide it. **unsure**, confidence 2. Do not mark
`unsupported` merely because the sentence is short; `unsupported` requires a relevant
sentence that is silent on the claim.

---

## 6. Confidence scale

| Value | Meaning |
|---:|---|
| 5 | Clear from the text; no reasonable alternative reading. |
| 4 | Likely; a minor ambiguity or one small inferential step. |
| 3 | Two readings are possible; one is preferred. |
| 2 | Heavily ambiguous, context incomplete, or the answer form is contestable. |
| 1 | Cannot judge from the available text; use only with a truncated or missing context. |

Confidence is about the label, not about how interesting the case is. A confidence of 5
on `artifact_label_noise` means the context clearly supports the answer, not that the
benchmark is wrong for every purpose.

---

## 7. Workflow for authors

1. Read this codebook, especially Section 3 and the rubric for the sheet you are about
   to edit.
2. Open the sheet in an editor that preserves UTF-8 and quoted commas. Keep a backup.
3. For each row, print the full case with the command in `P0_HUMAN_REVIEW_GUIDE.md`.
   Do not label from the snippet alone.
4. Apply the support test. Then apply the sheet's rubric.
5. Edit `author_label` and, where the sheet has one, the category or verdict column.
   Leave `ai_*` columns untouched as the pre-annotation record.
6. Set `annotator_id` to your initials, `changed` to `yes` or `no`, and
   `review_status` to `author_verified` or `author_corrected`.
7. If you disagree with the AI suggestion, say why in `notes`. A short reason is enough.
8. Do not compute agreement until every author has completed the blind sheet
   independently.

### 7.1 Column semantics

| Column | Meaning |
|---|---|
| `label` | Benchmark label from the source data. Never edit. |
| `primary_category` | AI-suggested error category. Authors may correct it in place for the final record; `ai_suggested` keeps the original for the agreement check. |
| `secondary_category` | AI-suggested second category, or blank. |
| `ai_suggested` | AI-suggested label in the sheet's label domain: the category code on the taxonomy sheet, `plausible`/`implausible`/`unsure` on the explanation sheet, and the claim verdict on the claim sheet. |
| `ai_rationale` | Why the AI suggested that label. Quote or paraphrase the decisive text. |
| `ai_confidence_1to5` | AI confidence in the suggestion. |
| `author_label` | Author's independent label. Blank until the author fills it. |
| `final_label` | Adjudicated label after discussion. Blank until adjudication. |
| `annotator_id` | Author initials. Blank until the author fills it. |
| `changed` | `yes` if the author changed the AI suggestion, `no` otherwise. Blank until review. |
| `review_status` | `needs_author_verification` until an author reviews the row. |
| `notes` | Free text for disagreements, caveats, and adjudication. |

---

## 8. Relationship to the earlier WP5 codebook

This document is a v2 expansion of `docs/rebuild/wp5_codebook.md`. The 20 categories
and their boundaries are unchanged. The additions are the plausibility rubric, the
claim-verdict rubric, the three worked examples per rubric, the benchmark-label rule,
and the rank-defined selection caveat for the 25 false-positive slots. The earlier WP5
sheets remain valid as historical drafts. Do not merge their counts with P0 counts
without stating the different selection rules.

---

## 9. Change log

| Date | Change |
|---|---|
| 2026-11-11 | Initial P0 v2 codebook. Adds plausibility and claim-verdict rubrics, three worked examples per rubric, benchmark-label rule, and the 25-slot selection caveat. |
