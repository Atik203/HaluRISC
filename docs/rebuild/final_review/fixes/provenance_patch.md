# Provenance and reference patch — exact replacements

**Targets:** `Journal_Paper/halurisc.tex`, `Journal_Paper/ref.bib`
**Review:** `docs/rebuild/final_review/06_provenance.md`
**Prepared:** 12 November 2026
**Status:** prepared, not applied. No manuscript, bibliography, PDF, or artifact was modified.

The provenance wording below is the author-approved two-sentence formulation. It is used verbatim in Patch 1. No extra provenance sentence is added.

> The labels are author-checked pre-annotations: LLM tools assisted the annotation, and the authors reviewed and verified the final labels. Agreement with the pre-annotations was 0.98 for explanation, 0.94 for taxonomy, and 0.88 for claims.

Tone compliance: the replacement is exactly the two sentences above. It uses “reviewed and verified the final labels” and contains no multi-reviewer, blinding, or hyphenated status label, and no independent-human-annotation disclaimer.

Apply the replacements manually. For repeated strings such as the `\credit` lines, use the surrounding author block shown here so the edit is unique.

---

## Patch 1 — §5.5 provenance sentence

**File:** `Journal_Paper/halurisc.tex:854` (last sentence of the explanation-results paragraph)

### Current text (exact)

```latex
A three-author blinded review with AI pre-annotation covered the audit; agreement was 0.94 for explanation, 0.98 for taxonomy, and 0.88 for claims, so the labels are author-verified.
```

### Replacement (exact)

```latex
The labels are author-checked pre-annotations: LLM tools assisted the annotation, and the authors reviewed and verified the final labels. Agreement with the pre-annotations was 0.98 for explanation, 0.94 for taxonomy, and 0.88 for claims.
```

### Why

- Removes the unsupported multi-reviewer and blinded-review claim and the claimed verification status.
- Corrects the swapped agreement rates: the artifacts give explanation 0.980 and taxonomy 0.940, not the reverse.
- Keeps the claims rate at 0.880.
- Uses the approved two-sentence wording with no added clause.

### Evidence

- `docs/rebuild/p0/author_review/AI_REVIEW_RESULTS.md:24-30`:

  | Sheet | Rows | AI-author agreement | Cohen's kappa | Corrected | Blind Fleiss kappa |
  |---|---:|---:|---:|---:|---:|
  | Taxonomy | 50 | 0.940 | 0.923 | 6.0% | 0.868 |
  | Explanation | 100 | 0.980 | 0.965 | 2.0% | 0.940 |
  | Claims | 100 | 0.880 | 0.788 | 12.0% | 0.761 |

- `docs/rebuild/p0/annotation_manifest.json:202-207` records `author_review.agreement` as taxonomy 0.94, explanation 0.98, claims 0.88. The artifact table calls this column “AI-author agreement”; the approved wording reports the same rates as agreement with the pre-annotations.
- `06_provenance.md` P2 documents the swap, and P5 documents the unsupported verification wording.
- `06_provenance.md` P1 documents the repository-history caveat: commit `b957fb0` changes only `annotator_id`, not labels. This patch adopts the author-approved provenance statement; the artifact-history caveat is not resolved by wording and remains open (see “Open items”).

### Author decision

**[AUTHOR DECISION] B5.5 scope.** The exact two-sentence wording does not name the P0 package. Because the paragraph is about the 26-case B5.5 audit, a reader can read “The labels” as the audit labels. The B5.5 artifact still has `review_mode=ai-expert` for 26/26 rows and `agreement=yes` for 23/26. The approved wording is used here without an added scope sentence to honour the “keep it to these two sentences” instruction. If the author wants the scope explicit, a separate sentence is needed; it is not part of this ready patch.

---

## Patch 2 — §5.6 50-case sheet wording

**File:** `Journal_Paper/halurisc.tex:955`

### Current text (exact)

```latex
A separate 50-case sheet, verified by the three authors, adds 31 artifact cases and 19 content cases across 12 categories.
```

### Replacement (exact)

```latex
A separate 50-case sheet, reviewed by one author, adds 31 artifact cases and 19 content cases across 12 categories.
```

### Why

The 50-row taxonomy sheet has one reviewer, not three. `verified by the three authors` is false as written. The replacement is consistent with the approved collective verification sentence in Patch 1: one author reviewed this sheet, and the authors verified the final labels across the package.

### Evidence

- `docs/rebuild/p0/author_review/taxonomy_reviewer_a.csv`: 50 rows, all `annotator_id=MAR`.
- `docs/rebuild/p0/author_review/README.md:13-16`: Reviewer A handled taxonomy (50 rows), Reviewer B explanation (100 rows), Reviewer C claims (100 claims).
- `06_provenance.md` P4 records the same finding.

---

## Patch 3 — Ethics sentence

**File:** `Journal_Paper/halurisc.tex:1009`

### Current text (exact)

```latex
The three authors annotated public benchmark rows for internal verification; this does not constitute human-subjects research, so ethics approval was not required.
```

### Replacement (exact)

```latex
The authors checked LLM-assisted pre-annotations of public benchmark rows for internal verification; this does not constitute human-subjects research, so ethics approval was not required.
```

### Why

“Annotated” implies from-scratch human annotation. The labels were LLM-assisted pre-annotations that the authors checked. This also removes the unsupported “three authors” description.

### Evidence

- `docs/rebuild/p0/annotation_manifest.json:178`: `human_review` is recorded as “completed; three authors reviewed and accepted the AI pre-annotations on 2026-11-11”.
- `06_provenance.md` P7 recommends “reviewed AI-generated pre-annotations” and flags the verb “annotated”.
- The task-approved wording uses “checked LLM-assisted pre-annotations” and is applied exactly.

---

## Patch 4 — Data availability

**File:** `Journal_Paper/halurisc.tex:1005`

### Current text (exact)

```latex
  \item \textbf{Data availability:} Full raw datasets are not redistributed. The repository includes download scripts, source revisions, SHA-256 hashes, license notes for HaluEval (MIT), RAGTruth (MIT), FaithBench (CC BY-NC-SA 4.0), MedHallu (MIT), and the HaluBench subset (CC BY-NC 2.0), and a 26-case audit file with benchmark excerpts. Splits and feature matrices can be reproduced from the original dataset sources. Model artifacts and the experiment manifest are available from the corresponding author on reasonable request.
```

### Replacement (exact)

```latex
  \item \textbf{Data availability:} Full raw datasets are not redistributed. For HaluEval (MIT), RAGTruth (MIT), and FaithBench (CC BY-NC-SA 4.0), the repository includes download scripts and a tracked license manifest with the source revisions and SHA-256 hashes, together with a 26-case audit file with benchmark excerpts. MedHallu (MIT) and the HaluBench subset (CC BY-NC 2.0) were obtained from their official releases. Splits and feature matrices can be reproduced from the original dataset sources. Model artifacts and the experiment manifest are available from the corresponding author on reasonable request.
```

### Why

The tracked manifest and download scripts cover HaluEval, RAGTruth, and FaithBench only. MedHallu and HaluBench have no tracked download script, source revision, or SHA-256 in the repository. The replacement narrows the claim and states the correct source for the other two datasets without inventing hashes or scripts.

### Evidence

- `web/artifacts/results/dataset_license_manifest.json` is tracked and lists exactly three datasets — `halueval`, `ragtruth`, `faithbench` — each with `url`, `license`, `download_revision`, and file-level `sha256`.
- `git ls-files` shows the tracked download scripts `src/data/download.py` (HaluEval), `src/data/download_ragtruth.py`, and `src/data/download_faithbench.py`.
- `src/models/rebuild_medhallu.py` and `src/models/p1_halubench.py` read `data/raw/medhallu/pqa_labeled.parquet` and `data/raw/halubench/test.parquet`; the P1 reports and JSON carrying the MedHallu/HaluBench notes are under gitignored `artifacts/results/rebuild/p1/`.
- The MedHallu script identifies the source as `UTAustin-AIHealth/MedHallu`, config `pqa_labeled`, MIT licence; the HaluBench script identifies Patronus AI and CC BY-NC 2.0. “Official releases” is therefore accurate and does not add a URL or hash.
- `06_provenance.md` P6 records the coverage gap.

---

## Patch 5 — Code licence

**File:** `Journal_Paper/halurisc.tex:1007`

### Current text (exact)

```latex
  \item \textbf{Code availability:} The training pipeline, FastAPI inference service, and Next.js dashboard are available at \url{https://github.com/Atik203/HaluRISC} (branch \texttt{rebuild}).
```

### Replacement (exact)

```latex
  \item \textbf{Code availability:} The training pipeline, FastAPI inference service, and Next.js dashboard are available at \url{https://github.com/Atik203/HaluRISC} (branch \texttt{rebuild}). The code is released under the MIT license.
```

### Why

The repository is MIT licensed, but the manuscript did not state the code licence. The sentence adds the missing term.

### Evidence

- `LICENSE:1` reads `MIT License`.
- The repository is public, and the current branch is `rebuild` at `69cb088`.
- `06_provenance.md` P14 records the missing licence statement.

---

## Patch 6 — Acknowledgement licence wording

**File:** `Journal_Paper/halurisc.tex:1014`

### Current text (exact)

```latex
The authors thank the maintainers of HaluEval, RAGTruth, FaithBench, MedHallu, and HaluBench for releasing their datasets under licenses that permit research use.
```

### Replacement (exact)

```latex
The authors thank the maintainers of HaluEval, RAGTruth, FaithBench, MedHallu, and HaluBench for releasing their datasets under licenses that permit research use. FaithBench is released under CC BY-NC-SA 4.0 and HaluBench under CC BY-NC 2.0. Both permit non-commercial research use.
```

### Why

“Licenses that permit research use” is imprecise for the two non-commercial CC datasets. The replacement names both licences and states the non-commercial condition.

### Evidence

- `06_provenance.md` check 6 and P13: FaithBench CC BY-NC-SA 4.0, HaluBench CC BY-NC 2.0, HaluEval MIT, RAGTruth MIT, MedHallu MIT.
- `web/artifacts/results/dataset_license_manifest.json` records FaithBench as `CC BY-NC-SA 4.0`.
- HaluBench’s Hugging Face dataset card records `cc-by-nc-2.0` (verified in `06_provenance.md`).

---

## Patch 7 — CRediT roles for the two annotation reviewers

**File:** `Journal_Paper/halurisc.tex:245-253`

### Current text (exact)

Second author:

```latex
% Second author
\author[1]{Saiful Alam Sabbir}[orcid=0009-0009-8460-1295]
\ead{sabbir2310105@bscse.uiu.ac.bd}
\credit{Writing - review \& editing}
```

Third author:

```latex
% Third author
\author[1]{MD. Miraz Ahamed}[orcid=0009-0001-1709-8812]
\ead{mahamed2310524@bscse.uiu.ac.bd}
\credit{Writing - review \& editing}
```

### Replacement (exact)

Second author:

```latex
% Second author
\author[1]{Saiful Alam Sabbir}[orcid=0009-0009-8460-1295]
\ead{sabbir2310105@bscse.uiu.ac.bd}
\credit{Data curation, Validation, Writing - review \& editing}
```

Third author:

```latex
% Third author
\author[1]{MD. Miraz Ahamed}[orcid=0009-0001-1709-8812]
\ead{mahamed2310524@bscse.uiu.ac.bd}
\credit{Data curation, Validation, Writing - review \& editing}
```

### Why

Both authors reviewed a 100-row annotation sheet plus the 60-item blind overlap, but their CRediT entries list only writing and review. `Data curation` and `Validation` describe the annotation check without overstating it as from-scratch data collection.

### Evidence

- `docs/rebuild/p0/author_review/explanation_reviewer_b.csv`: 100 rows, all `annotator_id=SAS`.
- `docs/rebuild/p0/author_review/claims_reviewer_c.csv`: 100 rows, all `annotator_id=MMA`.
- `docs/rebuild/p0/author_review/p0_blind_overlap_merged.csv`: 60 items with `a1`, `a2`, `a3` labels.
- `docs/rebuild/p0/author_review/README.md:13-16` describes the sheet assignments.
- `06_provenance.md` P12 recommends adding Investigation and/or Data curation, and Validation where appropriate.

### Author decision

**[AUTHOR DECISION] Optional “Investigation”.** If the author wants the review work credited more broadly, add `Investigation` to the same `\credit` lists. `Data curation` and `Validation` are the narrower, evidence-backed terms and are used in the ready replacement.

---

## Patch 8 — PDF author metadata

**File:** `Journal_Paper/halurisc.tex:280-284`

### Current text (exact)

```latex
\begingroup
\hfuzz=120pt
\maketitle
\endgroup
\linenumbers
```

### Replacement (exact)

```latex
\begingroup
\hfuzz=120pt
\maketitle
\endgroup
\hypersetup{
  pdfauthor={Md. Atikur Rahaman, Saiful Alam Sabbir, MD. Miraz Ahamed, Ohidujjaman Tuhin},
  pdftitle={Calibrated Hallucination Risk for Large Language Models Under Domain and Task Shift with Evidence-Consistent XGBoost}
}
\linenumbers
```

### Why

The class-generated PDF metadata garbles the fourth author. Setting `pdfauthor` explicitly after `\maketitle` overrides the class value. `pdftitle` is set to the exact manuscript title for consistency.

### Evidence

- `pdfinfo Journal_Paper/out/halurisc.pdf` currently reports the author line with garbled separators and the fourth author rendered as `整⁡氮` instead of `Ohidujjaman Tuhin`.
- The visible byline is correct: `Journal_Paper/halurisc.tex:240-256` lists Md. Atikur Rahaman, Saiful Alam Sabbir, MD. Miraz Ahamed, and Ohidujjaman Tuhin.
- `hyperref` is loaded at `Journal_Paper/halurisc.tex:32`, so `\hypersetup` is available.
- `06_provenance.md` P15 records the metadata defect.

### Verification after rebuild

`pdfinfo Journal_Paper/out/halurisc.pdf` should show:

```text
Author:          Md. Atikur Rahaman, Saiful Alam Sabbir, MD. Miraz Ahamed, Ohidujjaman Tuhin
Title:           Calibrated Hallucination Risk for Large Language Models Under Domain and Task Shift with Evidence-Consistent XGBoost
```

---

## Patch 9 — `ref.bib` FaithBench authors

**File:** `Journal_Paper/ref.bib:79`

### Current text (exact)

```bibtex
  author    = {Bao, Feng Sheng and Li, Meng and Qu, Rui and others},
```

### Replacement (exact)

```bibtex
  author    = {Bao, Forrest Sheng and Li, Miaoran and Qu, Renyi and others},
```

### Why

The given names are wrong. The paper has 16 authors; `and others` remains appropriate for the remaining 13. The PDF renders only initials, so the error is invisible there but is present in the BibTeX.

### Evidence

- Crossref DOI `10.18653/v1/2025.naacl-short.38` returns, in order: Bao, Forrest Sheng; Li, Miaoran; Qu, Renyi; Luo, Ge; Wan, Erana; Tang, Yujia; Fan, Weisi; Tamber, Manveer Singh; Kazi, Suleman; Sourabh, Vivek; Qi, Mike; Tu, Ruixuan; Xu, Chenyu; Gonzales, Matthew; Mendelevitch, Ofer; Ahmad, Amin.
- `06_provenance.md` P8 records the same correction.

---

## Patch 10 — `ref.bib` HaluAgent sixth author

**File:** `Journal_Paper/ref.bib:61`

### Current text (exact)

```bibtex
  author    = {Cheng, Xiaoxue and Li, Junyi and Zhao, Wayne Xin and Zhang, Hongzhi and Zhang, Fuzheng and Zhang, Fangzhou and Gai, Kun and Wen, Ji-Rong},
```

### Replacement (exact)

```bibtex
  author    = {Cheng, Xiaoxue and Li, Junyi and Zhao, Wayne Xin and Zhang, Hongzhi and Zhang, Fuzheng and Zhang, Di and Gai, Kun and Wen, Ji-Rong},
```

### Why

The sixth author is Di Zhang, not Fangzhou Zhang. The current entry produces two consecutive `F. Zhang` initials in the PDF, which is visibly wrong.

### Evidence

- Crossref DOI `10.18653/v1/2024.emnlp-main.809` returns the author order ending `... Zhang, Fuzheng; Zhang, Di; Gai, Kun; Wen, Ji-Rong`.
- `06_provenance.md` P9 records the same correction.

---

## Patch 11 — `ref.bib` spaCy year/DOI consistency

**File:** `Journal_Paper/ref.bib:206-207`

### Current text (exact)

```bibtex
  year      = {2020},
  doi       = {10.5281/zenodo.1212303},
```

### Replacement (exact, recommended)

```bibtex
  year      = {2023},
  doi       = {10.5281/zenodo.1212303},
```

### Why

`10.5281/zenodo.1212303` is the spaCy concept DOI. It currently resolves to the v3.7.2 record, published 2023-10-16, not a 2020 release. Keeping the concept DOI and setting the year to 2023 makes the entry internally consistent and preserves the general software citation.

### Evidence

- Zenodo API record `1212303`: `conceptdoi = 10.5281/zenodo.1212303`, latest `doi = 10.5281/zenodo.10009823`, version `v3.7.2`, publication date `2023-10-16`.
- Zenodo API record `10009823`: title `explosion/spaCy: v3.7.2: Fixes for APIs and requirements`, version `v3.7.2`, DOI `10.5281/zenodo.10009823`.
- `06_provenance.md` P11 records the year/DOI mismatch.

### Alternative (version-specific)

If the author wants to cite v3.7.2 exactly, use:

```bibtex
  year      = {2023},
  doi       = {10.5281/zenodo.10009823},
```

This pins the version DOI rather than the concept DOI. The recommended replacement keeps the concept DOI because the manuscript cites spaCy as general software, not a specific patch release.

### Author decision

**[AUTHOR DECISION] Concept DOI vs version DOI.** The recommended form is the concept DOI with year 2023. Choose the version DOI only if a version-specific citation is preferred.

---

## Patch 12 — `ref.bib` verification note (optional, recommended)

**File:** `Journal_Paper/ref.bib:3-6`

### Current text (exact)

```bibtex
% Every entry verified against ACL Anthology / arXiv / DOI / Crossref before
% insertion (first pass 2026-09-17; extended 2026-09-23 with the survey,
% semantic-entropy, MiniCheck, and RAGAs entries).
% Re-check DOIs at submission time (roadmap §17.3).
```

### Replacement (exact)

```bibtex
% Every entry verified against ACL Anthology / arXiv / DOI / Crossref before
% insertion (first pass 2026-09-17; extended 2026-09-23 with the survey,
% semantic-entropy, MiniCheck, and RAGAs entries).
% Author-name and software-metadata corrections applied 2026-11-12
% (faithbench, haluagent, spacy); re-check DOIs at submission time (roadmap §17.3).
```

### Why

The header claims every entry was verified, but the author-name defects and the spaCy metadata mismatch contradicted that record. The added note records the corrections without claiming a full re-verification.

### Evidence

- `06_provenance.md` “References” section: “The `ref.bib` header says every entry was verified ... The two author-name defects above contradict that record; correct the entries and update the verification note”.
- Patches 9-11 are the corrections referenced.

---

## Open items not patched

These are provenance or reporting issues from `06_provenance.md` that cannot be fixed by the approved two-sentence wording or by a text replacement in the manuscript.

### 1. Repository history does not independently show label-level author edits (P1)

`06_provenance.md` P1 reports that commit `b957fb0` changes only `annotator_id` from `ai_reviewer_a/b/c` to `MAR/SAS/MMA`; `author_label`, `final_label`, `changed`, and `review_status` are byte-identical before and after. The approved wording is the author’s provenance statement, but a journal integrity check may ask for a review record that is separate from the AI labels. A signed review log or dated reviewer notes would address the provenance gap; a README naming the authoritative files would address the consistency issue in P10. This patch does not create or modify any review record.

### 2. Kappa and correction rates remain unreported (P3)

The manuscript still reports no Cohen’s kappa, blind Fleiss kappa, or correction rate. The artifacts contain them (`docs/rebuild/p0/annotation_manifest.json:208-219`; `docs/rebuild/p0/author_review/AI_REVIEW_RESULTS.md:24-30`):

- Cohen’s kappa: taxonomy 0.9227, explanation 0.9645, claims 0.7878.
- Blind Fleiss kappa: taxonomy 0.868, explanation 0.9404, claims 0.7612; mean 0.8565.
- Corrected rows: taxonomy 3/50 (6.0%), explanation 2/100 (2.0%), claims 12/100 (12.0%).

`06_provenance.md` P3 recommends a compact table. The approved provenance wording is only two sentences, so no table is added here. This is an author decision if the journal requires the statistics.

### 3. B5.5 audit labels and the P0 package are not explicitly separated (P5)

The approved two-sentence wording follows a paragraph about the 26-case B5.5 audit. The B5.5 artifact still has `review_mode=ai-expert` for 26/26 rows and `agreement=yes` for 23/26; the same 26 cases appear as `P0X001`-`P0X026` in the 100-row P0 explanation sheet. The exact wording does not say “P0”, so a reader may read “The labels” as the audit labels. Adding a scope sentence would fix this, but the instruction is to keep the passage to the two approved sentences. This is flagged for author decision.

### 4. Canonical P0 sheets and manifest still disagree on review status (P10)

`docs/rebuild/p0/p0_error_taxonomy.csv`, `p0_explanation_audit.csv`, and `p0_claim_verdicts.csv` still contain 250 rows with `review_status=needs_author_verification`. `docs/rebuild/p0/annotation_manifest.json` says `status: author_verified` and `human_review_completed: true` while also carrying `agreement_statistic_computed: false` and `agreement_status: pending three-author blind labelling`. This is an artifact-consistency problem, not a manuscript sentence. It is outside this patch because the task prohibits editing artifacts. A README naming the authoritative files and one consistent status would fix it.

---

## Summary

### Coverage of the provenance findings

| Finding | Status in this patch |
| :--- | :--- |
| P1 author-review artifacts | Open. The text uses the author-approved wording; the artifact-history caveat remains. |
| P2 swapped agreement rates | Fixed by Patch 1. |
| P3 kappa and correction rates | Open. Author decision if the journal requires the statistics. |
| P4 multi-reviewer and blinded wording | Fixed by Patches 1-2. |
| P5 B5.5/P0 conflation | Partially addressed. The B5.5 scope ambiguity is flagged as an author decision. |
| P6 data availability | Fixed by Patch 4. |
| P7 ethics wording | Fixed by Patch 3. |
| P8 FaithBench attribution | Fixed by Patch 9. |
| P9 HaluAgent attribution | Fixed by Patch 10. |
| P10 canonical sheets and manifest | Open. Artifact fix, outside this text patch. |
| P11 spaCy metadata | Fixed by Patch 11; concept-DOI choice is an author decision. |
| P12 CRediT | Fixed by Patch 7. |
| P13 acknowledgement licences | Fixed by Patch 6. |
| P14 code licence | Fixed by Patch 5. |
| P15 PDF metadata | Fixed by Patch 8. |

### Ready replacements

| # | Item | Location | Replacement status |
| :-- | :--- | :--- | :--- |
| 1 | §5.5 provenance | `halurisc.tex:854` | Ready, author-approved two sentences |
| 2 | §5.6 50-case sheet | `halurisc.tex:955` | Ready |
| 3 | Ethics sentence | `halurisc.tex:1009` | Ready |
| 4 | Data availability | `halurisc.tex:1005` | Ready, narrowed to three tracked datasets |
| 5 | Code licence | `halurisc.tex:1007` | Ready, MIT |
| 6 | Acknowledgement licences | `halurisc.tex:1014` | Ready |
| 7 | CRediT for SAS and MMA | `halurisc.tex:246-253` | Ready |
| 8 | PDF metadata | `halurisc.tex:280-284` | Ready |
| 9 | FaithBench authors | `ref.bib:79` | Ready |
| 10 | HaluAgent sixth author | `ref.bib:61` | Ready |
| 11 | spaCy year | `ref.bib:206-207` | Ready, recommended |
| 12 | `ref.bib` verification note | `ref.bib:3-6` | Ready, optional |

### Decisions needed

1. **B5.5 scope (Patch 1).** The exact two-sentence wording does not say “P0”. Accept the wording as written, or add a scope sentence later to make clear that the 26-case audit labels are separate from the P0 package. No scope sentence is included in the ready patch because the instruction is to keep it to the two sentences.
2. **spaCy DOI (Patch 11).** Keep the concept DOI `10.5281/zenodo.1212303` with year 2023 (recommended), or pin the v3.7.2 version DOI `10.5281/zenodo.10009823`.
3. **CRediT scope (Patch 7).** `Data curation` and `Validation` are the evidence-backed additions. Add `Investigation` only if the author wants broader credit for the annotation review.
4. **P1, P3, P10.** These remain open outside the text patch: the repository history does not show label-level author edits, kappa and correction rates are absent from the manuscript, and the canonical P0 sheets still say `needs_author_verification`. They need artifact or reporting work, not a wording change.

### Post-apply verification

After the main session applies the patches:

1. `git diff --check` for whitespace errors.
2. `latexmk -pdf -outdir=out halurisc.tex` from `Journal_Paper/`; confirm a clean build with no unresolved citations.
3. `pdftotext -layout out/halurisc.pdf -` and search for `three-author`, `blinded`, `author-verified`, `verified by the three authors`, `annotated public benchmark`, and `not independent human`; none should remain. Confirm `reviewed and verified the final labels` appears once.
4. Check the reference list: HaluAgent should show `F. Zhang, D. Zhang` and FaithBench should show `F. S. Bao, M. Li, R. Qu`.
5. `pdfinfo out/halurisc.pdf` and confirm the author and title metadata.
6. Re-run the relevant audit if the build process requires it; no artifact was changed by this patch.
