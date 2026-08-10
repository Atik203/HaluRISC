# Figure 1 — HaluRISC Pipeline Infographic (one-page UI-style design brief)

Use this brief in **Google Stitch** (or Gemini Flash Image / Ideogram) to
generate the end-to-end methodology figure as a **rich one-page infographic
styled like a polished product UI page**. The generated page IS the image we
use in the paper (Figure 1, Methodology chapter).

## Output requirements

- **Canvas:** 16:9 landscape, 2400 x 1350 px, PNG, 300 DPI, no watermark
- **Save as:** `report/figures/architecture.png`
- **Style:** clean dark-and-light hybrid UI page: white content cards on a
  soft light background with a blue-violet gradient header band. Rounded
  corners, thin borders, subtle shadows allowed here (this is a UI-style
  figure, not a plain schematic). Simple geometric icons per stage are
  welcome. All technical text must be exact (list below).

## Full color palette (use these exact hex values)

| Role | Hex |
|---|---|
| Page background | `#f8fafc` (light gray) |
| Header gradient | `#7c3aed` → `#4f46e5` → `#38bdf8` (violet-indigo-cyan) |
| Header text | white `#ffffff` |
| Input stage | slate `#475569` / tint `#f1f5f9` |
| Feature chips | violet `#7c3aed` on `#ede9fe` |
| Model boxes | indigo `#4f46e5` on `#eef2ff` |
| Calibrator box | cyan `#0891b2` on `#ecfeff` |
| Outputs | emerald `#059669`, amber `#d97706`, rose `#e11d48` (tints 10-15%) |
| Evaluation band | teal `#0d9488` on `#f0fdfa` |
| Dataset chips | blue `#2563eb` on `#eff6ff` |
| Border color | `#cbd5e1` |
| Body text | `#0f172a` (dark slate) |
| Muted text | `#64748b` |

## Page layout (top to bottom, one page)

**1. Header band (full width, gradient violet→indigo→cyan, white text):**

- App-style title: `HaluRISC`
- Subtitle: `Hallucination Risk Estimation Pipeline`
- A small white pill badge: `black-box · calibrated · explainable`

**2. Stage 1 — Inputs (three white cards in a row, numbered badge 1):**

- Card 1: `Question`
- Card 2: `Context / Evidence`
- Card 3: `Answer (black-box LLM)` with a small chip `no weights needed`
- A single arrow flows down into stage 2.

**3. Stage 2 — Feature extraction (one wide card, badge 2):**

- Title: `Feature Extraction — 26 Features, 7 Groups`
- Inside, seven chips in a row, violet tint, each with a tiny icon dot:
  `Length`, `Lexical`, `Entity`, `NLI`, `Numeric`, `Hedging`, `Semantic`
- Small muted caption: `NLI + embeddings + NER, cached once`
- Arrow down with the label `26 features`.

**4. Stage 3 — Model and calibration (two stacked cards, badge 3):**

- Top card (indigo): `XGBoost Classifier` with chip row:
  `grouped 5-fold CV`, `seeds 42/123/456`
- Bottom card (cyan): `Platt Calibrator` with chip row:
  `source: HaluEval val`, `target: RAGTruth`
- Small muted caption: `fit on validation only`
- Arrow down with the label `threshold 0.5`.

**5. Stage 4 — Outputs (three cards in a row, badge 4):**

- `Calibrated Risk Score` (emerald/amber/rose mini gauge dot)
- `Per-Claim NLI Verdicts` with three mini chips:
  `supported`, `contradicted`, `unsupported`
- `SHAP Explanation` with two tiny horizontal bars

**6. Evaluation band (full width, teal tint, white card):**

Three columns with small icons and sublabels:

- `Cross-Domain Zero-Shot` → sublabels `RAGTruth`, `FaithBench`
- `Explanation Reliability` → sublabels `perturbations`, `stability`
- `Conversational Chat` → sublabels `auto risk cards`, `citations`

**7. Footer strip (light band):**

- Three dataset chips: `HaluEval 20K`, `RAGTruth 17.8K`, `FaithBench 750`
- Muted note: `grouped leakage-free split · 26 features · 3 seeds`

## Step-by-step in Stitch

1. Open a blank Stitch canvas.
2. Paste this whole brief as the design prompt. Say: "build this one-page
   pipeline infographic exactly as specified, 16:9, styled like a product
   UI page".
3. After rendering, inspect EVERY technical label against the exact list
   above. Stitch sometimes rewrites or drops chips (the seven feature names
   and the six sublabels are the most common failures). Select the wrong text
   layer and retype it.
4. Match colors to the palette if Stitch improvised.
5. Export File -> Export / Download as PNG (300 DPI).
6. Save to `report/figures/architecture.png`.

## Quality checklist

- [ ] Header shows HaluRISC + subtitle + badge
- [ ] Seven feature chips exact: Length, Lexical, Entity, NLI, Numeric,
      Hedging, Semantic
- [ ] "grouped 5-fold CV" and "seeds 42/123/456" chips present
- [ ] "source: HaluEval val" and "target: RAGTruth" chips present
- [ ] Three output cards with verdict chips supported/contradicted/unsupported
- [ ] Evaluation band has all three modules with sublabels
- [ ] Footer has the three dataset chips with numbers
- [ ] Arrow labels "26 features" and "threshold 0.5" present
- [ ] Landscape 16:9, no watermark

## Alternative tools (same spec)

- **Gemini Flash Image / Ideogram:** paste the whole brief as the image
  prompt (see `prompt_pipeline_ai.md` for the matching prompt).
- **draw.io / Google Drawings:** manual build for pixel-perfect text if the
  AI keeps garbling labels.
