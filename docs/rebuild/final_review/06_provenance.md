# Provenance, References, Ethics, and Availability Review

**Manuscript:** `Journal_Paper/halurisc.tex`
**Build reviewed:** `Journal_Paper/out/halurisc.pdf` (25 pages, 38 references)
**Scope:** provenance, citations, ethics, licences, data and code availability
**Out of scope:** prose style, method correctness, statistics, section structure
**Date:** 12 November 2026
**Evidence inspected:** `halurisc.tex`, `ref.bib`, the built PDF, `docs/rebuild/P0_P1_RESULTS_INDEX.md`, `docs/rebuild/p0/author_review/`, `docs/rebuild/p0/annotation_manifest.json`, `docs/rebuild/wp5_*`, `artifacts/results/b5/b5_review_cases_reviewed.csv`, the git history of the review files, and the public records for every DOI/arXiv/Zenodo identifier, dataset licence, and repository branch.

---

## Verdict

**Major revision required: the provenance claims are not submission-ready.**

Four checks fail (author-review provenance, agreement/kappa reporting, data availability, and unsupported provenance claims) and two are partial (reference attribution and the ethics/annotation description). The references are real and all 38 are cited, the five dataset licences are correct, and the code URL and branch are correct. But the repository history does not support the manuscript's central claim that the annotation labels are "author-verified": the only commit that marks the author review complete substitutes author initials for AI reviewer IDs and changes no labels. Two agreement rates are also swapped, no Cohen's or Fleiss' kappa is reported, the "three-author blinded review" wording overstates a design in which each main sheet had one reviewer, and the data-availability statement covers only three of the five named datasets in tracked files.

The paper can be repaired without new modelling: either produce a genuine independent author review or rewrite the provenance to describe the AI pre-annotations honestly. Until one of those is done, fixes 1-3 below are blocking.

---

## Checks completed: 10/10

| # | Check | Status | Main evidence |
| :-- | :--- | :--- | :--- |
| 1 | All 38 bibliography entries cited; no uncited or missing keys | **Pass** | 38 unique `\cite` keys, 38 `ref.bib` entries, 38 entries in the PDF reference list; script diff is empty in both directions |
| 2 | Citation keys map to real, correctly attributed publications | **Partial** | Every DOI, Zenodo record, and arXiv ID resolves to the claimed work; `faithbench` and `haluagent` have wrong author names; `spacy` has a year/DOI mismatch |
| 3 | Author-review provenance: AI pre-annotations, author-verified, no from-scratch human claim | **Fail** | Commit `b957fb0` changes only `annotator_id` (`ai_reviewer_a/b/c` to `MAR/SAS/MMA`); no `author_label`, `final_label`, `changed`, or `review_status` cell changes in any of the 430 reviewed rows. The paper says "author-verified" |
| 4 | Kappa values and agreement rates match the author-review artifacts | **Fail** | Paper: "0.94 for explanation, 0.98 for taxonomy". Artifacts: taxonomy 0.940, explanation 0.980. No kappa or correction rate appears in the manuscript |
| 5 | B5.5 audit is AI-assisted and never described as a two-human-reviewer study | **Pass with caveat** | `halurisc.tex:854` says "AI-assisted audit"; `b5_review_cases_reviewed.csv` has `review_mode=ai-expert` for 26/26; no "two-human" or "two-reviewer" wording. The next sentence nonetheless conflates B5.5 with the P0 author review |
| 6 | Dataset licences: HaluEval MIT, RAGTruth MIT, FaithBench CC BY-NC-SA 4.0, MedHallu MIT, HaluBench CC BY-NC 2.0 | **Pass** | Verified against GitHub licences (HaluEval, RAGTruth, MedHallu), the FaithBench `LICENSE` file, and the HaluBench Hugging Face dataset card |
| 7 | Data availability: no raw redistribution; scripts, hashes, revisions, audit excerpts described correctly | **Fail** | Raw data are gitignored, and the 26-case audit file is tracked. The tracked licence manifest covers only HaluEval, RAGTruth, and FaithBench; MedHallu and HaluBench have no download script, revision, or SHA-256 in tracked files |
| 8 | Code availability: repository URL and branch present and correct | **Pass** | `https://github.com/Atik203/HaluRISC` is public; branch `rebuild` exists at `69cb088` and contains the manuscript; repository `LICENSE` is MIT |
| 9 | Ethics: no human-subject data; three-author benchmark annotation described correctly | **Partial** | No human-subject data; ethics exemption is reasonable. "The three authors annotated public benchmark rows" should say "reviewed AI-generated pre-annotations", and the main sheets had one reviewer each |
| 10 | No fabricated claims, invented numbers, or unsupported provenance statements | **Fail** | Swapped agreement rates, unsupported "author-verified"/"three-author blinded review" claims, B5.5/P0 conflation, and an overbroad data-availability statement |

---

## Issues

Severity: **C** = critical, **H** = high, **M** = medium, **L** = low. `L###` refers to `Journal_Paper/halurisc.tex`; `B###` refers to `Journal_Paper/ref.bib`.

| # | Sev | Check | Location and quote | Issue | Why it matters | Exact fix |
| :-- | :-- | :-- | :--- | :--- | :--- | :--- |
| P1 | **C** | 3, 10 | `L854`: "A three-author blinded review with AI pre-annotation covered the audit; agreement was 0.94 for explanation, 0.98 for taxonomy, and 0.88 for claims, so the labels are author-verified." | The repository contains no independent author labels. Commit `43e484a` ("Add three-reviewer AI calibration and author-proxy review") introduced the review sheets with `annotator_id = ai_reviewer_a/b/c`. In the reviewed-sheet CSVs, commit `b957fb0` ("Mark P0 author review complete and report kappa"), 27 minutes later, changes only the `annotator_id` column to `MAR/SAS/MMA` in all 250 main-sheet rows and all 180 blind rows. `author_label`, `final_label`, `changed`, and `review_status` are byte-identical before and after. The 3/2/12 "author_corrected" counts are exactly the AI reviewers' `changed=yes` rows (P0E07/P0E35/P0E46, P0X027/P0X028, and the 12 claims); no new corrections appear in the author-review commit. The merged blind labels `a1/a2/a3` match the three `blind_reviewer_*_scored.csv` `author_label` columns 60/60. | This does not prove the authors never looked at the labels; it proves the repository cannot substantiate the claim. This is the paper's central provenance claim. A reader, journal integrity officer, or reviewer who inspects the commit history will find AI labels with author initials substituted, not a three-author review. Every annotation-derived statement in the paper inherits this problem. | Either (a) conduct a genuine independent author review and commit it: new per-item author labels in a separate file (do not overwrite the AI labels), reviewer IDs, timestamps, disagreement notes, and a signed review log; or (b) rewrite the provenance honestly: "AI-generated pre-annotations were checked and accepted by the authors", remove "author-verified" and "three-author blinded review", and report the agreement as AI-AI or AI-self-consistency, not human agreement. Do not describe the current artifacts as a three-author review. |
| P2 | **H** | 4 | `L854`: "agreement was 0.94 for explanation, 0.98 for taxonomy, and 0.88 for claims" | The artifact values are taxonomy 0.940, explanation 0.980, claims 0.880 (`AI_REVIEW_RESULTS.md`; `annotation_manifest.json` `author_review.agreement`). Explanation and taxonomy are swapped. | The sentence misreports two of the three headline agreement statistics. A reader who checks the supplement will find the reverse. | Change to "0.98 for explanation, 0.94 for taxonomy, and 0.88 for claims" after resolving P1, or replace the sentence with a table of agreement, kappa, and correction rate by sheet. |
| P3 | **H** | 4 | `L854` and the whole manuscript: no "kappa" or "Fleiss" appears in the `.tex` or the PDF | The manuscript reports three agreement rates but no Cohen's kappa, no blind Fleiss kappa, and no correction rate. Artifacts: Cohen 0.923/0.965/0.788; blind Fleiss 0.868/0.940/0.761 (mean 0.857); corrected 3/50 (6.0%), 2/100 (2.0%), 12/100 (12.0%). `P0_P1_RESULTS_INDEX.md` and `AUTHOR_PROXY_REVIEW.md` say these are reported; they are not. | The paper claims an author-verified review but omits the statistics that let a reader judge it. This is a provenance/reporting gap, not a style preference. | Add one compact table: sheet, n, agreement, Cohen's kappa, corrected n (%), blind Fleiss kappa. State exactly what each statistic compares: AI suggestion vs final label for the main sheets, three AI passes for the blind overlap. If P1 is resolved by rewriting, label the same table "AI-AI agreement" rather than "author agreement". |
| P4 | **H** | 3, 4 | `L854`: "three-author blinded review"; `L955`: "A separate 50-case sheet, verified by the three authors" | The main sheets each have one annotator: taxonomy `MAR` (50), explanation `SAS` (100), claims `MMA` (100). Only the 60-item blind overlap carries all three IDs. The main sheets include `ai_suggested`, `ai_rationale`, and `ai_confidence_1to5`, so the reviewer was not blind to the pre-annotation. The 50-case taxonomy sheet is `annotator_id=MAR` only. | "Three-author" and "blinded" overstate the design; a reader will infer three independent reviewers per item. The 50-case sentence is false as written. | Describe the design exactly: 250 main rows, one author per sheet; 60-item blind overlap labelled by all three; AI suggestions visible on the main sheets. Replace "verified by the three authors" with "reviewed by one author and accepted by the team", or retain "the authors" only if all three actually reviewed that sheet. |
| P5 | **H** | 5, 10 | `L854`: "A three-author blinded review ... covered the audit ... so the labels are author-verified." | This conflates the B5.5 26-case audit with the P0 100-row explanation sheet. B5.5 is explicitly AI-assisted: `b5_review_cases_reviewed.csv` has `review_mode=ai-expert` for all 26 rows, and `docs/rebuild/wp5_explanation_labels.csv` still has `review_status=needs_author_verification` for all 26. The 26 B5 cases appear as `P0X001`-`P0X026` in the P0 explanation sheet, which `SAS` reviewed and accepted without change; that is not a three-author review of the B5.5 labels. | The sentence implies that the 13 implausible / 3 unsure / 10 plausible B5.5 labels are author-verified. The B5.5 artifact still says they need author verification. This is an unsupported provenance statement about the paper's explanation audit. | Split the claims: "The 26-case B5.5 audit was AI-assisted (two AI passes, 23/26 agreement). The same 26 cases were included in the 100-row P0 explanation sheet, which one author reviewed; the B5.5 labels themselves remain AI-authored." Or complete and commit author verification of the B5.5 labels and name the reviewer(s). |
| P6 | **H** | 7 | `L1005`: "The repository includes download scripts, source revisions, SHA-256 hashes, license notes for HaluEval (MIT), RAGTruth (MIT), FaithBench (CC BY-NC-SA 4.0), MedHallu (MIT), and the HaluBench subset (CC BY-NC 2.0)" | The tracked licence manifest (`web/artifacts/results/dataset_license_manifest.json`) covers only HaluEval, RAGTruth, and FaithBench. There are no download scripts, source revisions, or SHA-256 hashes for MedHallu or HaluBench: `rebuild_medhallu.py` and `p1_halubench.py` read `data/raw/medhallu/pqa_labeled.parquet` and `data/raw/halubench/test.parquet`. The P1 reports and JSON with the MedHallu/HaluBench licence notes are under gitignored `artifacts/results/rebuild/p1/`. | A reader cannot reproduce the external results from the repository alone, and the data-availability statement overstates what is shipped. This is exactly the claim a journal checks. | Add MedHallu and HaluBench to the tracked licence manifest with URL, revision, SHA-256, licence, and a download script; or narrow the sentence to the three fully covered datasets and state that MedHallu and HaluBench are obtained from their official releases. |
| P7 | **M** | 9 | `L1009`: "The three authors annotated public benchmark rows for internal verification" | The labels were AI pre-annotations; the authors at most reviewed and accepted them. The verb "annotated" implies from-scratch human annotation, which the paper elsewhere disclaims with "AI pre-annotation". | The ethics statement and the results sentence give different provenance. This is the kind of inconsistency that triggers an integrity query. | "The three authors reviewed AI-generated pre-annotations of public benchmark rows for internal verification." |
| P8 | **M** | 2 | `B79` (`faithbench`): "Bao, Feng Sheng and Li, Meng and Qu, Rui and others" | The actual authors are Forrest Sheng Bao, Miaoran Li, and Renyi Qu (ACL Anthology and Crossref, DOI 10.18653/v1/2025.naacl-short.38). The given names are wrong. The PDF renders initials that happen to match, so the error is invisible there. | Incorrect author attribution in the bibliography; a reader who exports the BibTeX gets wrong names. | "Bao, Forrest Sheng and Li, Miaoran and Qu, Renyi and others". |
| P9 | **M** | 2 | `B61` (`haluagent`): "Zhang, Fangzhou" | The sixth author is Di Zhang (ACL Anthology and Crossref, DOI 10.18653/v1/2024.emnlp-main.809). The PDF renders "F. Zhang, F. Zhang" for Fuzheng Zhang and Fangzhou Zhang, which is visibly wrong. | Incorrect author attribution; the reference list contains two identical initials for different people. | "Zhang, Di". |
| P10 | **M** | 3, 10 | `docs/rebuild/p0/annotation_manifest.json`: `"status": "author_verified"` (L6) and `"human_review_completed": true` (L186) versus `"agreement_statistic_computed": false` and `"agreement_status": "pending three-author blind labelling"` (L159-160). The canonical `docs/rebuild/p0/p0_error_taxonomy.csv`, `p0_explanation_audit.csv`, and `p0_claim_verdicts.csv` still have 250 rows with `review_status=needs_author_verification`. | The manifest contradicts itself and the visible P0 sheets. The author-verified status exists only in the `author_review/` copies (including `ingest/`) and in the manifest's asserted counts, not in the canonical `docs/rebuild/p0/` sheets. | A reader who opens the main annotation package sees unverified labels while the paper says author-verified. This is a provenance consistency failure independent of P1. | Update the canonical P0 sheets (or make the ingest copies the canonical paths), set one consistent status, and remove the stale "pending"/"false" fields. Add a short README stating which files are authoritative. |
| P11 | **L** | 2 | `B206` (`spacy`): "year = {2020}" with DOI `10.5281/zenodo.1212303` | The DOI resolves to `explosion/spaCy: v3.7.2` (2023), not a 2020 release. | Minor metadata inconsistency; the citation is still recognisable, but the DOI and year disagree. | Set `year = {2023}`, or use the concept DOI/version that matches the 2020 release. |
| P12 | **L** | 9 | `L243-L253` CRediT: `SAS` and `MMA` have only "Writing - review & editing" | `SAS` reviewed 100 explanation rows and `MMA` reviewed 100 claim rows (plus 60 blind items each); the CRediT does not credit annotation or investigation. | Under-credits the annotation work that the Declarations describes; inconsistent with "the three authors annotated". | Add "Investigation" and/or "Data curation" to `SAS` and `MMA`, and "Validation" where appropriate. |
| P13 | **L** | 6, 7 | `L1014`: "under licenses that permit research use" | FaithBench (CC BY-NC-SA 4.0) and HaluBench (CC BY-NC 2.0) permit non-commercial research use, not unrestricted use. | The acknowledgement is imprecise about the two non-commercial licences. | "under licences that permit non-commercial research use" for the two CC BY-NC datasets, or list each licence explicitly as the Declarations does. |
| P14 | **L** | 8 | Code availability `L1007`: no licence stated | The repository `LICENSE` is MIT, but the code-availability statement does not say so. | Reusers do not know the terms; journals often require a code licence. | Add "The code is released under the MIT licence." |
| P15 | **L** | — | PDF metadata: `pdfinfo` reports `Author: Md. Atikur RahamanⰠ﻿Saiful Alam SabbirⰠ﻿MD. Miraz AhamedⰠ整⁡氮` | The fourth author's name is garbled in the PDF metadata, although the visible byline is correct. | The PDF's author metadata is wrong for one author, which affects indexing and ORCID linking. | Set the PDF metadata explicitly with `\hypersetup{pdfauthor={...}}` or fix the `\pdfstringdef` path in the custom `\author` definition. |

---

## Reference and provenance checklist

### References

| Item | Status | Evidence |
| :--- | :--- | :--- |
| 38 bibliography entries | **Pass** | 38 entries in `ref.bib`; 38 references in the PDF |
| All entries cited | **Pass** | 38 unique `\cite` keys; no uncited entries |
| No missing keys | **Pass** | Every cited key resolves in `ref.bib`; no `[?]` in the PDF |
| DOI resolution | **Pass** | All Crossref DOIs resolve to the claimed title; Zenodo and arXiv records resolve via DataCite/arXiv |
| HaluEval attribution | **Pass** | Li et al., EMNLP 2023, DOI 10.18653/v1/2023.emnlp-main.397 |
| RAGTruth attribution | **Pass** | Niu et al., ACL 2024, DOI 10.18653/v1/2024.acl-long.585 |
| FaithBench attribution | **Fail** | `B79` has "Feng Sheng / Meng / Rui" instead of "Forrest Sheng / Miaoran / Renyi" |
| MedHallu attribution | **Pass** | Pandit et al., arXiv 2502.14302 |
| HaluBench/Lynx attribution | **Pass** | Ravi et al., arXiv 2407.08488 |
| HaluAgent attribution | **Fail** | `B61` has "Fangzhou Zhang" instead of "Di Zhang" |
| spaCy attribution | **Partial** | DOI resolves to v3.7.2 (2023), but the entry says 2020 |
| All other entries | **Pass** | Spot-checked authors, titles, venues, and years against Crossref/ACL Anthology; no fabricated entries |

The `ref.bib` header says every entry was verified against ACL Anthology, arXiv, DOI, or Crossref. The two author-name defects above contradict that record; correct the entries and update the verification note so the next reader does not rely on a stale check.

### Provenance and reporting

| Item | Status | Evidence |
| :--- | :--- | :--- |
| AI pre-annotation disclosed | **Pass** | `L854` says "AI pre-annotation"; `annotation_manifest.json` records "AI agent pre-annotation" |
| Author verification supported by artifacts | **Fail** | Commit `b957fb0` changes only `annotator_id`; no label/status changes (P1) |
| Agreement rates match artifacts | **Fail** | Explanation/taxonomy values are swapped (P2) |
| Kappa values present and correct | **Fail** | No kappa anywhere in the `.tex` or PDF (P3) |
| Correction rates present | **Fail** | No 6.0% / 2.0% / 12.0% correction rates in the manuscript (P3) |
| Three-author design described accurately | **Fail** | One reviewer per main sheet; only the 60-item blind overlap has three IDs (P4) |
| 50-case taxonomy sheet "verified by the three authors" | **Fail** | `taxonomy_reviewer_a.csv` has `annotator_id=MAR` for all 50 rows (P4) |
| B5.5 audit identified as AI-assisted | **Pass** | `L854` "AI-assisted"; `review_mode=ai-expert` 26/26 |
| B5.5 labels author-verified | **Fail** | `wp5_explanation_labels.csv` still `needs_author_verification` 26/26 (P5) |
| No "two-human-reviewer" claim | **Pass** | No such wording in the manuscript or PDF |

### Data, code, and ethics

| Item | Status | Evidence |
| :--- | :--- | :--- |
| HaluEval licence MIT | **Pass** | GitHub API: MIT |
| RAGTruth licence MIT | **Pass** | GitHub API: MIT |
| FaithBench licence CC BY-NC-SA 4.0 | **Pass** | `vectara/FaithBench/LICENSE` is CC BY-NC-SA 4.0 |
| MedHallu licence MIT | **Pass** | `MedHallu/MedHallu` GitHub API: MIT |
| HaluBench licence CC BY-NC 2.0 | **Pass** | Hugging Face `PatronusAI/HaluBench`: `cc-by-nc-2.0` |
| Raw data not redistributed | **Pass** | `data/raw/` is gitignored; only excerpts in the 26-case audit and P0 sheets |
| Download scripts for all five datasets | **Fail** | Only HaluEval, RAGTruth, and FaithBench have download scripts (P6) |
| Source revisions for all five datasets | **Fail** | Tracked manifest covers three datasets only (P6) |
| SHA-256 hashes for all five datasets | **Fail** | Tracked manifest covers three datasets only (P6) |
| 26-case audit file with excerpts | **Pass** | `artifacts/results/rebuild/explanations/m3_audit_cases.csv` (26 rows, `snippet`) and `artifacts/results/b5/b5_review_cases_reviewed.csv` (26 rows, full excerpts) are tracked |
| Code repository URL and branch | **Pass** | Public repo; branch `rebuild` at `69cb088` |
| Code licence | **Partial** | Repo is MIT, but the paper does not state it (P14) |
| No human-subject data | **Pass** | Public benchmark rows only; no human participants |
| Annotation description accurate | **Fail** | "Annotated" should be "reviewed AI pre-annotations" (P7) |
| CRediT reflects annotation work | **Fail** | `SAS` and `MMA` only "Writing - review & editing" (P12) |

---

## Top 3 fixes

1. **Reconcile the author-review artifacts with the paper's provenance claims (P1, P4, P5, P7, P10).**
   The current artifacts are AI labels with author initials substituted; the paper calls them a "three-author blinded review" and "author-verified". Either commit a genuine independent author review (new per-item labels, reviewer IDs, timestamps, disagreement notes, signed log) or rewrite the provenance to "AI-generated pre-annotations were checked and accepted by the authors" and remove every "author-verified" and "three-author blinded review" claim. Split the B5.5 audit from the P0 review in the same edit.

2. **Correct the annotation statistics (P2, P3).**
   Swap the explanation and taxonomy agreement values (0.98 explanation, 0.94 taxonomy), add Cohen's kappa (0.923/0.965/0.788), blind Fleiss kappa (0.868/0.940/0.761; mean 0.857), and correction rates (3/50, 2/100, 12/100). State what each statistic compares. If fix 1 ends in a rewrite, label the table as AI-AI agreement rather than human agreement.

3. **Fix the data-availability coverage and the two citation attributions (P6, P8, P9).**
   Add MedHallu and HaluBench to the tracked licence manifest with URL, revision, SHA-256, licence, and a download path, or narrow the sentence to the three datasets that are fully covered. Correct `faithbench` to "Bao, Forrest Sheng and Li, Miaoran and Qu, Renyi" and `haluagent` to "Zhang, Di". Update the spaCy year or DOI while the bibliography is open.

---

## What passed

- All 38 references are cited and present; no uncited or missing keys.
- Every DOI, Zenodo record, and arXiv ID resolves to the claimed work; no fabricated publication was found.
- The five dataset licences are stated correctly: HaluEval MIT, RAGTruth MIT, FaithBench CC BY-NC-SA 4.0, MedHallu MIT, HaluBench CC BY-NC 2.0.
- Raw datasets are not redistributed; the repository ships the 26-case audit file and the P0 annotation package.
- The code repository is public, the `rebuild` branch exists, and the repository is MIT licensed.
- The B5.5 audit is explicitly described as AI-assisted; it is not called a two-human-reviewer study.
- The ethics statement correctly identifies the work as non-human-subjects research, subject to the wording fix in P7.
