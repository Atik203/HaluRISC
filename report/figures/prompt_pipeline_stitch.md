# Figure 1 — HaluRISC Pipeline Infographic (Google Stitch design brief)

This is the single source for the methodology figure. It contains the full
design (canvas, palette, page layout, every label), the copy-paste prompt for
**Google Stitch** (or Gemini/Ideogram), and the export steps.

## Output requirements

- **Canvas:** 16:9 landscape, 2400 x 1350 px, PNG, 300 DPI, no watermark
- **Save as:** `report/figures/architecture.png`
- **Style:** one-page infographic styled like a polished product UI page.
  Light gray background, white content cards with rounded corners and thin
  slate borders, blue-violet-cyan gradient header band, simple geometric
  icons per stage, subtle shadows allowed. No clipart.

## Full color palette (exact hex values)

| Role | Hex |
|---|---|
| Page background | `#f8fafc` |
| Header gradient | `#7c3aed` → `#4f46e5` → `#38bdf8` |
| Header text | white `#ffffff` |
| Body text | `#0f172a` |
| Muted text | `#64748b` |
| Border | `#cbd5e1` |
| Violet (features) | `#7c3aed` |
| Indigo (model) | `#4f46e5` |
| Cyan (calibrator) | `#0891b2` |
| Emerald (low risk) | `#059669` |
| Amber (medium risk) | `#d97706` |
| Rose (high risk) | `#e11d48` |
| Teal (evaluation) | `#0d9488` |
| Blue (dataset chips) | `#2563eb` |
| Card tints | 10–15% of each accent (e.g. `#ede9fe`, `#eef2ff`, `#ecfeff`, `#f0fdfa`, `#eff6ff`) |

## Page layout (top to bottom, one page)

**Header band (full width, gradient):** title `HaluRISC`, subtitle
`Hallucination Risk Estimation Pipeline`, white pill badge
`black-box - calibrated - explainable`.

**Stage 1 — Inputs (three cards in a row, badge 1):**
`Question` · `Context / Evidence` · `Answer (black-box LLM)` with chip
`no weights needed`. Arrow flows down.

**Stage 2 — Feature extraction (one wide card, badge 2):**
Title `Feature Extraction - 26 Features, 7 Groups`, seven chips in a row:
`Length`, `Lexical`, `Entity`, `NLI`, `Numeric`, `Hedging`, `Semantic`.
Muted caption `NLI + embeddings + NER, cached once`.
Arrow down labeled `26 features`.

**Stage 3 — Model and calibration (two stacked cards, badge 3):**
- Indigo card `XGBoost Classifier` with chips `grouped 5-fold CV` and
  `seeds 42/123/456`
- Cyan card `Platt Calibrator` with chips `source: HaluEval val` and
  `target: RAGTruth`
- Muted caption `fit on validation only`
- Arrow down labeled `threshold 0.5`

**Stage 4 — Outputs (three cards in a row, badge 4):**
`Calibrated Risk Score` (green-amber-red gauge dot) · `Per-Claim NLI
Verdicts` (chips `supported`, `contradicted`, `unsupported`) · `SHAP
Explanation` (two small bars).

**Evaluation band (full width, teal tint, three columns):**
- `Cross-Domain Zero-Shot` → sublabels `RAGTruth`, `FaithBench`
- `Explanation Reliability` → sublabels `perturbations`, `stability`
- `Conversational Chat` → sublabels `auto risk cards`, `citations`

**Footer strip:** three chips `HaluEval 20K`, `RAGTruth 17.8K`,
`FaithBench 750`, muted note
`grouped leakage-free split - 26 features - 3 seeds`.

## Copy-paste prompt (Stitch or Gemini/Ideogram)

```
Design a one-page infographic styled like a polished product UI page for an
academic machine learning paper about hallucination risk detection in large
language models. Light gray background, white content cards with rounded
corners and thin slate borders, and a blue-violet-cyan gradient header band.
Simple geometric icons per stage. Subtle shadows are fine. No clipart, no
watermark. The layout must read top to bottom as a clear pipeline.

Canvas 16:9, landscape, 2400x1350 pixels. Use this exact palette: page
background #f8fafc, header gradient #7c3aed to #4f46e5 to #38bdf8, white
header text, body text #0f172a, muted text #64748b, border #cbd5e1, violet
#7c3aed, indigo #4f46e5, cyan #0891b2, emerald #059669, amber #d97706,
rose #e11d48, teal #0d9488, blue #2563eb. Card tints at 10-15% of each
accent color.

Header band (full width): title "HaluRISC", subtitle "Hallucination Risk
Estimation Pipeline", and a white pill badge reading "black-box - calibrated
- explainable".

Stage 1, numbered badge 1, three cards in a row: "Question",
"Context / Evidence", "Answer (black-box LLM)" with a small chip
"no weights needed". An arrow flows down to stage 2.

Stage 2, badge 2, one wide card titled "Feature Extraction - 26 Features,
7 Groups" containing exactly seven chips in a row: "Length", "Lexical",
"Entity", "NLI", "Numeric", "Hedging", "Semantic". Small muted caption
"NLI + embeddings + NER, cached once". Arrow down labeled "26 features".

Stage 3, badge 3, two stacked cards: an indigo card "XGBoost Classifier"
with chips "grouped 5-fold CV" and "seeds 42/123/456"; below it a cyan card
"Platt Calibrator" with chips "source: HaluEval val" and "target: RAGTruth".
Muted caption "fit on validation only". Arrow down labeled "threshold 0.5".

Stage 4, badge 4, three cards in a row: "Calibrated Risk Score" with a
green-amber-red gauge dot, "Per-Claim NLI Verdicts" with chips "supported",
"contradicted", "unsupported", and "SHAP Explanation" with two small bars.

Evaluation band, full width, teal tint, three columns: "Cross-Domain
Zero-Shot" with sublabels "RAGTruth", "FaithBench"; "Explanation
Reliability" with sublabels "perturbations", "stability"; "Conversational
Chat" with sublabels "auto risk cards", "citations".

Footer strip with three chips: "HaluEval 20K", "RAGTruth 17.8K",
"FaithBench 750" and a muted note "grouped leakage-free split - 26 features
- 3 seeds".

Keep every label short and spelled exactly as written above. Do not add any
text that is not listed. No watermark.
```

## Step-by-step in Stitch

1. Open a blank Stitch canvas.
2. Paste the copy-paste prompt above as the design intent.
3. After rendering, inspect every technical label (the seven feature chips,
   the six evaluation sublabels, and the footer chips fail most often).
   Select any wrong text layer and retype the exact label.
4. Match colors to the palette table if Stitch improvised.
5. Export File -> Export / Download as PNG (300 DPI).
6. Save to `report/figures/architecture.png` (overwrites the Gemini version;
   the paper already points to this filename).

## Quality checklist

- [ ] Header: `HaluRISC` + subtitle + `black-box - calibrated - explainable`
- [ ] Seven feature chips exact: Length, Lexical, Entity, NLI, Numeric,
      Hedging, Semantic
- [ ] `grouped 5-fold CV` and `seeds 42/123/456` chips present
- [ ] `source: HaluEval val` and `target: RAGTruth` chips present
- [ ] Three output cards, verdict chips supported/contradicted/unsupported
- [ ] Evaluation band: all three modules with sublabels
- [ ] Footer: `HaluEval 20K`, `RAGTruth 17.8K`, `FaithBench 750`
- [ ] Arrow labels `26 features` and `threshold 0.5` present
- [ ] Landscape 16:9, no watermark, no extra text
