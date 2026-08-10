# Figure 1 — HaluRISC Pipeline Infographic (Google Stitch design brief)

Use this brief to build the end-to-end methodology figure in **Google Stitch**
(or any diagram tool: draw.io, Google Drawings, PowerPoint). It becomes
**Figure 1** in the paper's Methodology chapter.

## Output requirements

- **Canvas:** 16:9 landscape, 2400 x 1350 px (any wide landscape export works;
  LaTeX scales it with `\textwidth`)
- **Format:** PNG, 300 DPI, white background (transparent also fine)
- **Save as:** `report/figures/architecture.png`
- **Style:** flat design, rounded-corner boxes, thin straight arrows, no 3D,
  no shadows, no clipart, no watermark. All text short and legible.

## Color palette

| Element | Hex |
|---|---|
| Inputs | slate `#475569` |
| Feature group chips | violet `#7c3aed` |
| Model boxes | indigo `#4f46e5` |
| Output boxes | emerald `#059669`, amber `#d97706`, rose `#e11d48` |
| Evaluation band | teal `#0d9488` |
| Background | white `#ffffff`, box fill tints at ~10-15% of the accent |

## Layout and exact labels (copy these words exactly)

Top-to-bottom flow with a full-width evaluation band at the bottom.

**Band 1 — Inputs (three boxes side by side):**

```
[ Question ]   [ Context / Evidence ]   [ Answer (black-box LLM) ]
```

**Band 2 — Feature extraction (one wide box):**

```
Feature Extraction — 26 Features, 7 Groups
    chips: Length | Lexical | Entity | NLI | Numeric | Hedging | Semantic
```

**Band 3 — Model and calibration (two stacked boxes):**

```
[ XGBoost Classifier ]
   grouped 5-fold CV · seeds 42/123/456

[ Platt Calibrator ]
   source (HaluEval val) → target (RAGTruth)
```

**Band 4 — Outputs (three boxes side by side):**

```
[ Calibrated Risk Score ]  [ Per-Claim NLI Verdicts ]  [ SHAP Explanation ]
```

**Bottom band — Evaluation and deployment (three boxes in a row):**

```
[ Cross-Domain Zero-Shot ]     [ Explanation Reliability ]    [ Conversational Chat ]
   RAGTruth · FaithBench          perturbations · stability      auto risk cards · citations
```

**Arrow labels:**

- Between Band 2 and Band 3: `26 features`
- Between Band 3 and Band 4: `threshold 0.5`

## Step-by-step in Stitch

1. Open Google Stitch and start a blank design canvas.
2. Paste this whole brief into the design prompt (Stitch reads natural-language
   design intent). Say: "build this exact pipeline diagram, 16:9 landscape".
3. After it renders, inspect every text label. Fix any garbled or missing word
   by selecting the text layer and retyping the exact label from this file.
4. Adjust colors to the palette above if Stitch picked its own.
5. Export: File -> Export / Download as PNG (300 DPI if offered).
6. Save to `report/figures/architecture.png`.

## Quality checklist before saving

- [ ] All seven feature chips spelled exactly: Length, Lexical, Entity, NLI,
      Numeric, Hedging, Semantic
- [ ] "grouped 5-fold CV" and "seeds 42/123/456" present
- [ ] "source (HaluEval val) to target (RAGTruth)" present
- [ ] Bottom band has all three modules with sublabels
- [ ] Arrow labels "26 features" and "threshold 0.5" present
- [ ] Ratio is clearly landscape (16:9)

## Alternative tools (same spec)

- **draw.io / diagrams.net:** manual build, export File -> Export as PNG
  (zoom 100%, 300 DPI).
- **Google Drawings:** manual build, File -> Download -> PNG.
- If Stitch output keeps garbling text, fall back to the AI image prompt in
  `prompt_pipeline_ai.md`, or build manually in draw.io.
