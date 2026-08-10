# Figure 6 — System Architecture Diagram (AI image generator prompt)

Use this with **Gemini** (Flash Image / Imagen), **Ideogram**, or **Microsoft
Designer**. It produces the two-tier deployment diagram for the System
chapter. Same palette and visual language as `prompt_pipeline_ai.md`, so
Figure 6 matches Figure 1. AI generators often garble small text, so verify
every label after generation; if a retry still fails, the paper automatically
falls back to the built-in TikZ diagram.

## Output requirements

- 16:9 landscape, 2400 x 1350 px, PNG, no watermark
- Save as: `report/figures/system_architecture.png`
- The paper picks this file up automatically when present (no code change)

## The prompt (copy-paste)

```
Create a clean two-tier system architecture diagram for an academic paper
about an LLM hallucination detection web application. Flat infographic
style, light gray background, white cards with rounded corners and thin
slate borders, blue-violet-cyan accents, simple thin arrows. Match the
visual language of a modern product documentation diagram. No watermark.

Canvas 16:9, landscape, 2400x1350 pixels. Use this palette: page background
#f8fafc, header gradient #7c3aed to #4f46e5 to #38bdf8, slate text #0f172a,
muted text #64748b, violet #7c3aed, indigo #4f46e5, cyan #0891b2, emerald
#059669, amber #d97706, rose #e11d48, teal #0d9488, blue #2563eb, border
#cbd5e1, tints at 10-15%.

Top layer, one wide card: "Next.js 16 + assistant-ui" with sublabel
"chat & analyze pages".

Middle layer, one card below it: "Next.js BFF /api/chat" with sublabel
"Vercel AI SDK streaming".

Bottom layer, two side-by-side cards. Left card: "GPT 5.6 Luna" with
sublabel "conversational analysis + tool calls". Right card:
"FastAPI :8000" with sublabels "/predict /explain /verify" and
"/index /retrieve /feedback".

Below the FastAPI card, a wide card titled "ML Inference" with chips:
"XGBoost + Platt", "SHAP", "NLI (DeBERTa-v3)", "NER (spaCy)",
"SBERT (MiniLM)", "BM25 + FAISS", "Tavily web search", "document index".

Arrows: chat page down to BFF; BFF to GPT 5.6 Luna labeled "tool call";
BFF to FastAPI labeled "/api/ml/*"; FastAPI down to ML Inference; a
dashed arrow back up from BFF to the chat page labeled "streamed response".

Keep every label short and spelled exactly as written. Do not add text
that is not listed. No watermark.
```

## After generation

1. Zoom in and check every label against the exact list (the four endpoint
   chips and the eight ML chips fail most often).
2. Retry once with any wrong word in quotes at the end of the prompt.
3. If it still fails, the paper keeps using the TikZ fallback, or rebuild
   the diagram manually in draw.io using the same palette.
4. Save the final PNG to `report/figures/system_architecture.png`.
