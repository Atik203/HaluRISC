# Figure 1 — Pipeline Infographic (AI image generator prompt)

Fallback route for `prompt_pipeline_stitch.md`: use this with **Gemini**
(Flash Image / Imagen), **Ideogram**, or **Microsoft Designer**. It produces
the same rich one-page infographic (styled like a product UI page) as an
image. AI generators often garble small text, so verify every label after
generation; if a retry still fails, build the figure manually in draw.io.

## Output requirements

- 16:9 landscape, 2400 x 1350 px, PNG, no watermark
- Save as: `report/figures/architecture.png`

## The prompt (copy-paste)

```
Design a one-page infographic styled like a polished product UI page for an
academic machine learning paper about hallucination risk detection in large
language models. Light gray background, white content cards with rounded
corners and thin slate borders, and a blue-violet-cyan gradient header band.
Simple geometric icons per stage. Subtle shadows are fine. No clipart, no
watermark.

Canvas 16:9, landscape, 2400x1350 pixels. Use this palette: page background
#f8fafc, header gradient #7c3aed to #4f46e5 to #38bdf8, slate text #0f172a,
muted text #64748b, violet #7c3aed, indigo #4f46e5, cyan #0891b2, emerald
#059669, amber #d97706, rose #e11d48, teal #0d9488, blue #2563eb, border
#cbd5e1, tints at 10-15%.

Header band (full width): title "HaluRISC", subtitle "Hallucination Risk
Estimation Pipeline", and a white pill badge reading "black-box - calibrated
- explainable".

Stage 1, numbered badge 1, three cards in a row: "Question",
"Context / Evidence", "Answer (black-box LLM)" with chip "no weights
needed".

Stage 2, badge 2, one wide card titled "Feature Extraction - 26 Features,
7 Groups" containing seven chips in a row: "Length", "Lexical", "Entity",
"NLI", "Numeric", "Hedging", "Semantic". Caption "NLI + embeddings + NER,
cached once". Arrow down labeled "26 features".

Stage 3, badge 3, two stacked cards: indigo card "XGBoost Classifier" with
chips "grouped 5-fold CV" and "seeds 42/123/456"; cyan card "Platt
Calibrator" with chips "source: HaluEval val" and "target: RAGTruth".
Caption "fit on validation only". Arrow down labeled "threshold 0.5".

Stage 4, badge 4, three cards in a row: "Calibrated Risk Score" with a
green-amber-red gauge dot, "Per-Claim NLI Verdicts" with chips "supported",
"contradicted", "unsupported", and "SHAP Explanation" with two small bars.

Evaluation band, teal tint, three columns: "Cross-Domain Zero-Shot" with
sublabels "RAGTruth", "FaithBench"; "Explanation Reliability" with sublabels
"perturbations", "stability"; "Conversational Chat" with sublabels "auto
risk cards", "citations".

Footer strip with three chips: "HaluEval 20K", "RAGTruth 17.8K",
"FaithBench 750" and a muted note "grouped leakage-free split - 26 features
- 3 seeds".

Keep every label short and spelled exactly as written above. No extra text.
```

## After generation

1. Zoom in and check every label against the exact list (the seven feature
   chips, the six evaluation sublabels, and the footer chips fail most often).
2. Retry once with any wrong word in quotes at the end of the prompt.
3. If it still fails, build manually in draw.io / Google Drawings using the
   spec in `prompt_pipeline_stitch.md`.
4. Save the final PNG to `report/figures/architecture.png`.
