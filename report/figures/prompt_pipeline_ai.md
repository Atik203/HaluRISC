# Figure 1 — Pipeline Infographic (AI image generator prompt)

Fallback route for `prompt_pipeline_stitch.md`: use this with **Gemini**
(Flash Image / Imagen), **Ideogram**, or **Microsoft Designer** to generate the
same figure as an image. AI generators often garble small text, so check every
label after generation and prefer the Stitch/draw.io route if any word is
wrong after one retry.

## Output requirements

- 16:9 landscape, 2400 x 1350 px, PNG, no watermark
- Save as: `report/figures/architecture.png`

## The prompt (copy-paste)

```
Create a clean, professional academic journal figure for a machine learning
paper about hallucination risk detection in large language models. Flat
design, white background, rounded rectangles, thin straight arrows, no 3D
effects, no shadows, no clipart, no watermarks. Consistent blue-violet color
palette: inputs in slate gray, feature groups in violet, the model in indigo,
outputs in emerald/amber/rose, evaluation modules in teal. Canvas 16:9,
landscape, 2400x1350 pixels.

Layout, left to right in four horizontal bands:

Band 1 (top): three input boxes side by side labeled "Question",
"Context / Evidence", "Answer (black-box LLM)". One arrow joins them downward.

Band 2: a single wide box labeled "Feature Extraction - 26 Features, 7
Groups", containing seven small chips in a row labeled exactly: "Length",
"Lexical", "Entity", "NLI", "Numeric", "Hedging", "Semantic". Arrow downward.

Band 3: two stacked boxes. Top box: "XGBoost Classifier" with a small
subtitle "grouped 5-fold CV - seeds 42/123/456". Bottom box: "Platt
Calibrator" with subtitle "source (HaluEval val) to target (RAGTruth)".
Arrow downward.

Band 4 (outputs): three boxes side by side labeled exactly:
"Calibrated Risk Score", "Per-Claim NLI Verdicts", "SHAP Explanation".

Bottom band (full width): three boxes in a row labeled exactly:
"Cross-Domain Zero-Shot Evaluation" with sublabels "RAGTruth", "FaithBench";
"Explanation Reliability" with sublabels "perturbations", "stability";
"Conversational Chat" with sublabels "auto risk cards", "citations".

Arrow labels: "26 features" between band 2 and 3, "threshold 0.5" between
band 3 and 4. Keep all text short, legible, and perfectly spelled. No extra
text anywhere else.
```

## After generation

1. Zoom into the image and verify every label against the exact list in
   `prompt_pipeline_stitch.md` (especially the seven feature chips and the
   subtitles).
2. If any word is garbled, retry once with that word in quotes at the end of
   the prompt: e.g. "...make sure the words 'Hedging' and 'FaithBench' appear
   exactly".
3. If it still fails, build the figure manually in draw.io or Google Drawings
   using the same spec file.
4. Save the final PNG to `report/figures/architecture.png`.
