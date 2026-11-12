# Editor-in-Chief Review — HaluRISC Journal Manuscript

**Manuscript:** `Journal_Paper/halurisc.tex`  
**PDF reviewed:** `Journal_Paper/out/halurisc.pdf` (build of 2026-11-12 15:33)  
**Review date:** 2026-11-12  
**Scope:** structure, layout, and submission compliance only. Method correctness, statistics, number provenance, and prose style are owned by other reviewers.

## Verdict

**Minor revision.**

The manuscript is structurally sound and reads as a complete method-based paper. Twelve of the twelve checks were completed; two explicit layout rules fail (three short body paragraphs and two floats whose first mention is three pages away), and the project's page/float/reference count records are stale. None of these is a blocking defect.

The current PDF is **25 pages**. That is consistent with the rebuild status documents (`docs/rebuild/SCORE_ESTIMATE.md` expects 25 pages; `docs/rebuild/P0_P1_REVIEWER_RESPONSE.md` says "page limit is not a constraint for this draft"), but the older `docs/writing-guideline.md` G0 entry still records an 18-page approved budget and an `pages = 18` audit step. The records must be reconciled before submission. If the target venue actually enforces 18 pages, this becomes a **major revision**; no venue limit is recorded in the current rebuild notes.

## Checks completed: 12/12

| # | Check | Result | Evidence |
| :-- | :--- | :--- | :--- |
| 1 | Title clear, specific, at most two lines | PASS | PDF p1: title renders on exactly two lines; names task, shift, and method. |
| 2 | Abstract one paragraph, 250–300 words, six questions in order, headline numbers only | PASS with minor note | 278 words, one paragraph; context → gap → objective → approach → justification → outcome in order. `ECE` is not introduced at first use. |
| 3 | Keywords 4–8, specific, not overly broad | PASS | 7 keywords; all specific except the acceptable application term "Question answering". |
| 4 | Method-based section order; no orphan or redundant subsection | PASS with minor note | 1 Introduction → 2 Conventional (2.1–2.2) → 3 Proposed (3.1–3.5) → 4 Experiments (4.1–4.4) → 5 Results (5.1–5.6) → 6 Conclusion → CRediT → Declarations → Acknowledgements → References. §2.1 partly duplicates §4.4. |
| 5 | Every section/subsection has a clear job; none empty or one-sentence | PASS with minor note | No empty or one-sentence subsection. §5.3 and §5.5 each carry three analyses under one compound heading. |
| 6 | No substantive body paragraph under ~90 words or ~6 printed lines | **FAIL** | Three paragraphs: tex 414 (80 words/5 lines), tex 416 (70/4), tex 645 (88/5). Full list below. |
| 7 | Captions 1–2 lines; no `Note.` blocks | PASS with minor note | Every caption is 1–2 printed lines; no table/figure note blocks. Two tables carry unexplained markers (`†`, `*`). |
| 8 | Every float referenced, numbered in order, first mention within two pages | **FAIL** | All 18 tables and 12 figures are referenced and numbered 1..N in order. Tables 1 and 4 are first mentioned three pages before their captions. |
| 9 | Cross-references resolve; no undefined refs, overfull boxes, duplicate labels | PASS | 43 labels, all unique; 53 `\ref` uses, all targets exist; 0 overfull, 0 undefined, 0 duplicate-label warnings. Note: underfull warnings are globally suppressed at the bibliography. |
| 10 | Declarations: funding, conflicts, data, code, ethics | PASS | All five items present on PDF p23: Funding, Conflicts of Interest, Data availability, Code availability, Ethical approval. |
| 11 | Repository URL, branch, licences, dataset names present and consistent | PASS with minor note | URL `https://github.com/Atik203/HaluRISC`, branch `rebuild` (branch exists in the repo); five licence notes; dataset names consistent. SNLI/MultiNLI are used but absent from the licence list. |
| 12 | Page count, table/figure count, reference count internally consistent | PASS with note | PDF: 25 pages, 18 tables, 12 figures, 2 algorithms, 38 references. Numbering is complete and references are 38/38 cited. Source comment tex 13 still says "16 tables and 11 figures"; project status files still say 16/11/40. |

## Issues

| # | Severity | Location and quote | Issue | Why it matters | Exact fix |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Medium** | tex 414, PDF p5 ll. 151–155: `Third, the training set adds the RAGTruth summarization and data-to-text rows.` (80 words, 5 lines); tex 416, p5 ll. 156–159: `Algorithm 2 lists the procedure.` (70 words, 4 lines); tex 645, p11 ll. 330–334: `Figure 5 shows the seed-level spread behind Table 6.` (88 words, 5 lines) | Three substantive body paragraphs fall below both length thresholds | Violates check 6; fragments the method and results narrative; the page reads as notes rather than argument | Merge 414 and 416 into one ~150-word paragraph immediately before Algorithm 2; merge 645 into the Table 7 paragraph at tex 636. Rebuild and re-measure. |
| 2 | **Medium** | Table 1 first mention tex 304 (PDF p2), caption PDF p5; Table 4 first mention tex 447 (PDF p6), caption PDF p9 | First mention is three pages before the float, not within two | Reader must flip three pages to find the table; the forward reference is fragile after any rebuild | Table 1: move the table into §2.2 after tex 304, or remove `Table~\ref{tab:features}` from tex 304 and make §3.2 the first reference (p4). Table 4: move the table into §3.4 after tex 447, or remove the table reference from tex 447 and first cite it in §4.3 (p9). Rebuild and confirm the delta is ≤2 pages. |
| 3 | **Medium** | `docs/writing-guideline.md` G0: `18 pages ... Do not add further length without supervisor approval` (and its build facts still say 15 tables, 6 figures, 34 references); `docs/rebuild/status.md`: `25 pages, 11 figures, 16 tables, 40 references` and `all 27 floats within two pages`; `Journal_Paper/halurisc.tex` line 13: `16 tables and 11 figures` | Page, float, and reference counts are stale across the project records; the PDF is 25 pages, 18 tables, 12 figures, 38 references, and 30 floats, with Tables 1 and 4 failing the proximity rule | A final submission review cannot tell whether 25 pages is approved or whether the old 18-page limit still applies; the float-placement comment drives the relaxed parameters and will mislead the next editor | Confirm the venue limit. If 25 pages is allowed, update G0 to 25; if 18 is enforced, treat this as a major revision and cut to ≤18. Update `status.md` to 18 tables/12 figures/38 references and 30 floats, and tex line 13 to `18 tables and 12 figures`. |
| 4 | **Low** | Table 5, PDF p11: `External ensemble†`; Table 16, PDF p21: `0.014/0.110*` (tex 575 and 917) | Footnote markers have no legend anywhere in the caption or table | Tables are not self-contained; check 7 asks for clean captions, and a reader cannot decode the markers | Remove the markers. The prose already explains the reduced external ensemble, and the TF-IDF caveat is not explained anywhere. If a caveat is essential, put it in the cell or a one-line caption clause; do not add a `Note.` block. |
| 5 | **Low** | Abstract, PDF p1: `badly calibrated (expected calibration error 0.493)` then `lowers ECE` | `ECE` is used later but never introduced as an abbreviation | The abstract is read standalone; an undefined acronym is a clarity defect | Change to `expected calibration error (ECE) 0.493`. |
| 6 | **Low** | §2.1 tex 300: `precision, recall, F1, or the area under ... (AUROC) ... expected calibration error (ECE)`; §4.4 tex 534/543 defines AUROC, PR-AUC, MCC again | §2.1 partly duplicates the metric definitions in §4.4 | Redundant subsection content; wastes space; check 4/5 redundancy | Keep only the names of the criteria in §2.1 and forward-reference §4.4; or fold §2.1 into §2.2. Saves roughly 40 words. |
| 7 | **Low** | §2.2 ends tex 328 with `Algorithm 1 summarizes the tuning, training, and calibration steps.`; the body gap statement is at tex 292 (Introduction P3) | The Conventional Method section does not close with the limitation and research gap that G7 expects | The intended narrative is baseline → limitation → proposed method; here the reader meets the gap before the baseline, and §2 never returns to it | Add two or three closing sentences at the end of §2.2: length dependence, no claim-level evidence, and calibration that does not transfer. If the move to the Introduction was supervisor-approved, record the deviation in G0 instead. |
| 8 | **Low** | §5.3 heading, PDF p12: `Calibration, Shift, and External Validity`; §5.5 heading, PDF p18: `Explanation, Efficiency, and Judges` | Each heading lists three distinct analyses | Check 5's "clear job" is weakened; the reader cannot tell what unifies the subsection | Rename each to a single unifying question, e.g. `Reliability Under Distribution Shift` and `Explanation Quality and Operating Cost`, and add one roadmap sentence. Do not split, to protect the page budget. |
| 9 | **Low** | tex 412, 447, 453, 464, 475, 483, 513: `Section~\ref{sec:results}` resolves to `Section 5` | Seven forward references point to the whole Results section, not the relevant subsection | The reader lands at the top of the thirteen-page Results section; check 9 resolves but navigation is poor | Add labels to §5.2, §5.3, §5.5, and §5.6 and point each sentence at the specific subsection. |
| 10 | **Low** | §3.2 tex 362: `trained on SNLI [23] and MultiNLI [24]`; Declarations tex 1005 lists licences only for HaluEval, RAGTruth, FaithBench, MedHallu, HaluBench | The NLI training datasets are absent from the data-availability licence list | Check 11 asks for consistent dataset/licence coverage; a reader cannot tell how SNLI/MultiNLI are covered | Add a clause that SNLI and MultiNLI are used through their public releases for NLI training and their licence notes are in the repository, or state that the NLI model is a pretrained component and only the five evaluation corpora are redistributed. |

## Short paragraphs (check 6)

Every substantive body paragraph below ~90 words and ~6 printed lines, in source order.

| Source line | PDF location | Words | Printed lines | Opening quote | Merge recommendation |
| :-- | :-- | --: | --: | :--- | :--- |
| 414 | p5, ll. 151–155 | 80 | 5 | `Third, the training set adds the RAGTruth summarization and data-to-text rows.` | Merge with tex 416 into one ~150-word paragraph immediately before Algorithm 2. The two paragraphs describe the same training/deployment procedure. |
| 416 | p5, ll. 156–159 | 70 | 4 | `Algorithm 2 lists the procedure. Steps 1 and 2 build the feature matrix.` | Merge with tex 414, as above. |
| 645 | p11, ll. 330–334 | 88 | 5 | `Figure 5 shows the seed-level spread behind Table 6.` | Merge into the Table 7 paragraph at tex 636, which ends on the same fixed-hp ablation and its seed standard deviation. |

**Near-threshold, not counted as violations:** tex 288 (88 words, 6 printed lines) and tex 441 (90 words, 6 lines) meet the printed-line criterion; tex 530 (101 words, 6 lines) and tex 449 (102 words, 6 lines) also pass.

**Equation-adjacent runs excluded:** tex 312 (65 words, 2 lines) and tex 318 (78 words, 2–3 lines) are display-equation lead-ins. Tex 412 (74 words, 4 lines) is the continuation of Equation 6 and prints as part of the longer paragraph that starts at tex 402.

## Top 3 fixes

1. **Merge the three short paragraphs.** Merge tex 414 + 416 into one paragraph before Algorithm 2, and merge tex 645 into tex 636. Rebuild and confirm no body paragraph is below 6 printed lines.
2. **Repair float proximity.** Move or re-reference Table 1 and Table 4 so each first mention is within two pages of its caption. Rebuild and confirm the first-mention deltas.
3. **Reconcile the count records.** Confirm the venue limit first. If 25 pages is allowed, update `docs/writing-guideline.md` G0 to 25 pages; if 18 is enforced, cut to ≤18. In either case, update `docs/rebuild/status.md` to 18 tables/12 figures/38 references and tex line 13 to `18 tables and 12 figures`. The manuscript's printed numbering is already correct.

## What passed

Title (two lines), abstract length and question order, keyword count and specificity, section order, subsection content, caption length, absence of `Note.` blocks, cross-reference integrity, all five declarations, repository URL and branch, the five listed licence notes, and the 38/38 reference cross-check. The PDF's own counts (25 pages, 18 tables, 12 figures, 38 references) are complete and internally consistent; the mismatch is in the project's stale records, not in the manuscript's numbering.
