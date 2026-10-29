# HaluRISC — AI-detection rewrite plan

> Temporary working file. Delete after the final Turnitin pass. Not part of the manuscript.

**Baseline:** `halurisc-B (2).pdf`, 17 pages, 11,062 words, submitted 2026-10-29 13:34, **22%** detected as AI.
**Target:** under **10%** (≤1,106 flagged words). Working target ≤9% for buffer.
**Paper status:** AD-01, AD-02, AD-03, AD-04, AD-06, AD-07, AD-10 and AD-15 (data availability) applied; 7 blocks pending. Build verified after the edits.

---

## 1. Rules for this pass

- Every number, table cell, figure, and algorithm body stays exactly as it is. Captions may be reworded.
- Page count stays 17. Rebuild after each round.
- No first person. No new claims, no new citations, no deleted results.
- **Rewrite whole paragraphs, not sentences or synonyms.** Turnitin flags segments of 20+ words; a synonym swap inside a flagged segment usually stays flagged.
- Keep the register that already passes: abstract, §1.1–1.2, §5.1 prose, §5.5, §5.6, §5.9. Do not touch those.
- One detector submission per round, not per block.

---

## 2. Diagnosis (measured from the report)

Turnitin flagged **2,132 words in 135 lines** (22% of 11,062 ≈ 2,434, so the map is complete).

Flagged vs clean text on the same pages:

| Signal | Flagged | Clean | Ratio |
|---|---:|---:|---:|
| Sentences opening with a signpost ("The model…", "The goal…", "Figure 2…", "Table 13…", "This work…") | **14.0%** | 4.4% | 3.2× |
| Mean sentence length | 17.6 | 21.0 | 0.84 |
| Sentence-length sd | 11.2 | 13.6 | 0.82 |
| Sentences under 12 words | 35.1% | 23.4% | 1.5× |
| Sentences over 30 words | 12.3% | 16.6% | 0.74 |
| Copula in first five words | 22.8% | 22.0% | 1.04 |
| Transitions per 100 words | 0.28 | 0.23 | 1.2 |

The vocabulary is not the problem. Connectives and copulas are identical. The flagged prose is **more uniform and more signposted**: short topic sentence, then orderly elaboration, at a metronomic rhythm.

The five recurring moves:

1. **Inventory openings** — "The model uses 26 base features in seven categories (Table 1)…"; "The system accepts…"; "The goal is to estimate…".
2. **Announced structure** — "For deployment, there are two key evaluation considerations. First… Second…"; "Several tests and controls… First… Second… Third… Finally…".
3. **Meta-commentary** — "The contribution is the combination. Each piece is standard on its own…"; "This work assesses both of these properties…".
4. **Even numeric chains** — the judge paragraph ("27 cases… 2 cases… 20 answers… 6 ambiguous… 12 cases… 2 false alarms").
5. **Boilerplate** — table rows, captions, CRediT roles, Declarations.

The clean sections do the opposite: they open on a concrete subject or number, vary length hard, and report rather than explain.

---

## 3. Arithmetic

- 10% of 11,062 = **1,106 words**.
- Flagged now = **2,132 words**. Must clear **1,030–1,330**.
- Structural floor (cannot rewrite): table cells ~180, Algorithm steps ~50, CRediT ~54, references ~18 → **~300 words = 2.7%**.
- Rewritable flagged text: **~1,780 words**.

Expected residue if a share of rewrites clears:

| Clear rate | Words left | Score |
|---|---:|---:|
| 80% | ~710 | ~6.4% |
| 60% | ~1,065 | ~9.6% |
| 50% | ~1,240 | ~11.2% |

Tier 2 is the buffer. Tier 1 alone is not enough if only half the rewrites clear.

---

## 4. Blocks

### Tier 1 — round 1

| ID | Section | Paper p. | Flagged | Rewritable | Anchor |
|---|---|---|---:|---:|---|
| AD-01 | §5.7 cost + Table 13 discussion | 13 | 170 | 170 | "Local costs were based on the reference laptop…" |
| AD-02 | §5.7 judge paragraph + Table 14 | 14 | 292 | 160 | "[0.781, 0.898], Qwen 3.5 4B produced…" |
| AD-03 | §3.1 + Figure 1 | 3–4 | 250 | 230 | "The entire proposed pipeline is illustrated in Figure 1…" |
| AD-04 | §3.2 Features | 4 | 107 | 107 | "The model uses 26 base features…" |
| AD-05 | §3.5 System + Figure 2 | 6 | 209 | 170 | "The verdict card sets the risk to high…" |
| AD-06 | §2.1 Preliminaries | 2 | 137 | 137 | "Current test beds include HaluEval…" |
| AD-07 | §2.3 Limitations | 3 | 119 | 119 | "working with this dataset. However, under domain shift…" |
| AD-08 | §5.8 Error analysis | 15 | 130 | 130 | "The key aspect of the data is how the answer length…" |
| | **Tier 1** | | **1,414** | **1,223** | |

### Tier 2 — round 2, driven by the round-1 score

| ID | Section | Paper p. | Flagged | Rewritable | Anchor |
|---|---|---|---:|---:|---|
| AD-09 | §5.2 | 9 | 59 | 59 | "Table 6 shows the results of the component ablation…" |
| AD-10 | §5.3 + Table 9 | 9–11 | 128 | 100 | "Figure 4 shows the raw and target-calibrated…" |
| AD-11 | §5.4 | 11 | 54 | 54 | "Its external results are zero-shot…" |
| AD-12 | §3.3 + §3.4 | 5 | 156 | 106 | "The contribution is the combination…" |
| AD-13 | §4.3 + §4.4 | 7–8 | 90 | 90 | "trained with binary cross-entropy loss…" |
| AD-14 | Conclusion | 15–16 | 87 | 87 | "A useful hallucination detector must be fast…" |
| AD-15 | Declarations | 16 | 115 | 60 | "Data availability: The raw evaluation datasets…" |
| | **Tier 2** | | **689** | **556** | |

### Never rewrite

Table 5 model rows, Table 12 component rows, Table 13 baseline rows, Algorithm 2 body, CRediT roles, reference entries. Captions and Declarations prose are fair game.

### Applied so far

| ID | Section | Date | Old | New | Build | Detector |
|---|---|---|---:|---:|---|---|
| AD-01 | §5.7 cost paragraph (author text) | 2026-10-29 | 132 w | 144 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-02 | §5.7 judge paragraph + Table 14 (author text) | 2026-10-29 | 189 w | 196 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-03 | §3.1 + Figure 1 caption (author text) | 2026-10-29 | 100 w | 112 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-04 | §3.2 Features (author text) | 2026-10-29 | 85 w | 63 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-06 | §2.1 Preliminaries (author text) | 2026-10-29 | 165 w | 165 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-07 | §2.3 Limitations (author text) | 2026-10-29 | 166 w | 175 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-10 | §5.3 Calibration (author text) | 2026-10-29 | 120 w | 109 w | 17 pp, 0 warnings, 34/34 | pending |
| AD-15 | Declarations, data availability (author text) | 2026-10-29 | 61 w | 56 w | 17 pp, 0 warnings, 34/34 | pending |

Author edits also landed in §3.4 (SHAP reliability paragraph), the Figure 2 caption, §3.5 (reproducibility sentence), and the §5.3 Figure 4 paragraph. These were kept as written and are covered by AD-05 and AD-10.

Note: the author §3.2 text omits the numeric/hedging sentence. Both groups remain in Table 1, so the content is still covered.

---

## 5. Rewrite recipe (per paragraph)

1. Kill the signpost opening. Start on the concrete subject: the artifact, the corpus, the number, the failure.
2. Break the enumeration. Merge `First/Second/Third` into one claim plus one exception, or drop the labels and let the order carry it.
3. Delete meta-commentary. "The contribution is…" and "This work assesses…" become a direct statement of what was done.
4. Vary length hard. One 8-word sentence next to one 35-word sentence. The flagged mean is 17.6 with sd 11.2; the target is a wider spread than the clean text.
5. Keep every number and hedge exactly. "About", "up to", "mean over seeds" stay.
6. Read it aloud. If it sounds like a methods explainer, rewrite again.

---

## 6. Round protocol

For each round:

- [ ] Apply the blocks in the round.
- [ ] `latexmk -pdf -outdir=out halurisc.tex`; confirm 17 pages, 0 warnings, 34/34 citations, 203/203 audit.
- [ ] Submit to Turnitin; record score and newly flagged pages.
- [ ] If ≥10%, apply the next tier and repeat.

Stop when the score is under 10% or the author accepts the result.

---

## 7. Deletion

Delete this file after the final accepted score, together with the temporary extraction folder. Do not commit it.

---

## 8. Status log

| Date | Round | Blocks | Score | Notes |
|---|---|---|---:|---|
| 2026-10-29 | Analysis | — | 22% | Report read; 2,132 words mapped; paper untouched |
| 2026-10-29 | Verification fixes | AD-01-AD-15 | pending | Fixed the 20-40x latency claim, the review_number_audit.py attribution, Figure 2's 35% card value, the FPR wording, the cost/latency wording, remaining semicolons in the touched sections, w.r.t, and the missing article. 17 pages, 63 lines on p17, 0 warnings, 34/34, 203/203. Not pushed. |
