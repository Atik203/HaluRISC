# Slide guide and image generation prompts

This file has two parts. Part 1 is the presentation outline. Part 2 holds ready-to-paste prompts for the images on slides 2, 3, 4, and the two optional images on slides 5 and 12.

---

## Part 1. Presentation outline

1. Title Slide - Current Slide no need to change
2. Introduction - Key points, if have images, please add them
3. Motivation - Key points, if have images, please add them
4. Objective - Key points, if have images, please add them
5. Dataset - Sources, Details etc
6. Conventional Method, Baselines
7. Proposed Method, working Principles
8. Flow Diagram - Architecture of the Project from start to end
9. Experimental Setups, table
10. Results, table
11. Application, if any
12. UI Demonstration
13. Conclusion

Suggested final deck: 16 slides. Items 1 to 13 from the list, with the UI part shown on two slides, plus one extra Results slide and a Thank You slide.

| # | Slide | Guide item | Image |
|---|-------|-----------|-------|
| 1 | Title | 1 | keep current |
| 2 | Introduction | 2 | prompt 2A |
| 3 | Motivation | 3 | prompt 2B |
| 4 | Objective | 4 | prompt 2C |
| 5 | Dataset | 5 | prompt 2D (optional) |
| 6 | Conventional Method and Baselines | 6 | none |
| 7 | Proposed Method | 7 | none |
| 8 | Flow Diagram | 8 | generated SVG, not an AI image |
| 9 | Experimental Setup | 9 | none |
| 10 | Results: in-domain | 10 | none |
| 11 | Results: shift and calibration | 10 | none |
| 12 | Application | 11 | prompt 2E (optional) |
| 13 | UI: Chat | 12 | existing screenshot |
| 14 | UI: Analyze | 12 | existing screenshots |
| 15 | Conclusion | 13 | none |
| 16 | Thank You | - | keep current |

Slide 8 uses a vector diagram that ships with the deck, so it stays sharp at any projector size and needs no image generation.

---

## Part 2. Image generation prompts

### 1. Shared style contract

Every image must follow this contract, so all slides look like one deck. Paste the contract rules together with the prompt for the slide.

```
STYLE CONTRACT (apply to every image)
Canvas: 16:9 aspect ratio, 2560 x 1440 pixels, PNG, sRGB color.
Background: pure white #FFFFFF. Keep the outer 4 percent margin empty.
Look: clean flat vector illustration. Uniform outlines about 3 pixels at this
size. Rounded corners about 24 pixels. Flat solid fills, no photographic
texture, no 3D render, no bevel, no glossy highlight. One soft neutral drop
shadow only, light grey, low opacity.
Palette, use only these colors:
  blue #1E40AF, teal #0F766E, amber #B45309, red #B91C1C,
  ink #101828, slate #475569,
  tint blue #EEF2FF, tint teal #F0FDFA, tint amber #FFFBEB, tint red #FEF2F2.
Line and text color: ink #101828 for headings, slate #475569 for small labels.
Text: keep it minimal. Only the labels listed for the slide. Bold geometric sans
font, like Inter or Poppins, dark ink. Do not add any other words, numbers,
letters, logos, watermarks, or signatures.
People: no human faces and no realistic people. Use simple objects, cards,
charts, arrows, and icons only.
Mood: calm, academic, technical, trustworthy. Not playful. Not promotional.
Avoid: dark background, neon glow, gradient mesh, clipart style, stock photo
style, clutter, busy patterns.
```

If the generator adds broken or invented text, use the same prompt with the no-text variant at the end of each prompt, then add the labels inside the slide in the deck, where they stay crisp and consistent.

### 2. Slide 2, Introduction, prompt 2A

Purpose: a hero image that shows the core problem in one glance, a fluent answer that contradicts the provided evidence.

```
Create a flat vector illustration for a research presentation slide.

Scene: one horizontal composition on a white background, read from left to
right. On the left, a white document card with a grey header bar and four light
grey text lines; the second line is highlighted with a soft amber tint and
carries the short label "5 days". In the middle, a thick red arrow points from
the document to a chat bubble on the right, and a small red cross sits at the
arrow head to mark contradiction. On the right, a large rounded chat bubble in
blue #1E40AF with white text inside that carries the short label "10 days", and
a small blue speech tail at its lower left. Above the arrow, a small amber
warning triangle badge with a dark exclamation mark. Under the document, a tiny
slate label "EVIDENCE". Under the chat bubble, a tiny slate label "ANSWER".

Composition: the document card takes the left third, the arrow the middle, the
chat bubble the right third. Keep the top 15 percent of the canvas empty for a
slide title bar. Center everything vertically.

Style: follow the STYLE CONTRACT. Bold rounded outlines, flat fills, generous
white space, no gradients, no shadow except one soft neutral drop shadow under
the document card and the chat bubble.

Keep the labels exactly as written and no other text.
Negative: no realistic people, no photorealism, no dark background, no extra
words, no numbers other than 5 and 10, no logos, no watermark.
```

No-text variant: replace the label sentences with "no text anywhere" and add "a red cross marker instead of any words".

### 3. Slide 3, Motivation, prompt 2B

Purpose: show the three gaps the study covers, one panel each.

```
Create a flat vector illustration for a research presentation slide.

Scene: three equal panels in one row on a white background, each panel inside a
rounded rectangle with a thin slate border and a very light tint fill.

Left panel, tint blue #EEF2FF: a circular gauge with a blue needle and tick
marks, and next to it a small bar chart whose tallest bar, in amber #B45309,
clearly does not match the needle position. Small slate label "CALIBRATION"
under the panel.

Middle panel, tint teal #F0FDFA: a small horizontal attribution chart with five
bars, four teal bars of different lengths and one amber bar, and a dashed teal
outline that shows the same bar changing length, to suggest an unstable
explanation. Small slate label "EXPLANATIONS" under the panel.

Right panel, tint amber #FFFBEB: a straight dashed vertical divider in slate
that crosses the panel, a blue arrow crossing the divider from left to right,
and a red line that drops downward after crossing, to suggest a performance
drop on new data. Small slate label "DOMAIN SHIFT" under the panel.

Composition: the three panels fill the canvas with equal width and equal gaps.
Keep the top 15 percent of the canvas empty for a slide title bar.

Style: follow the STYLE CONTRACT. Flat fills, rounded corners, thin outlines,
no gradients, one soft neutral drop shadow per panel.

Keep the three labels exactly as written and no other text.
Negative: no realistic people, no photorealism, no dark background, no extra
words, no logos, no watermark, no clutter.
```

### 4. Slide 4, Objective, prompt 2C

Purpose: show the objective and the method in one glance, inputs to a calibrated risk output.

```
Create a flat vector illustration for a research presentation slide.

Scene: a single horizontal pipeline on a white background, read from left to
right in five stages, connected by thick blue arrows.

Stage 1, three small stacked rounded cards in slate gray, each with two light
grey text lines, labelled with a small slate caption bar. One caption reads
"QUESTION", one reads "CONTEXT", one reads "ANSWER".
Stage 2, a rounded rectangle with a thin blue border containing a grid of seven
small rounded chips in blue #1E40AF, teal #0F766E and amber #B45309, with the
small slate caption "35 FEATURES" centered above the rectangle.
Stage 3, a rounded rectangle in ink #101828 with a simple white decision tree
icon inside, three levels, and the small white caption "EC-XGB" inside the
rectangle at the bottom.
Stage 4, a rounded rectangle with a thin teal border containing a circular gauge
with a teal needle and a small tick list, and the small slate caption
"CALIBRATED RISK" below the rectangle.
Stage 5, three small rounded output cards in a vertical stack, one blue with a
gauge icon, one amber with a checklist icon, one red with a small bar chart
icon. Only the first card carries a short white label, "SCORE".

Composition: five equal stages in a row, aligned on one horizontal center line.
Keep the top 15 percent of the canvas empty for a slide title bar.

Style: follow the STYLE CONTRACT. Flat fills, rounded corners, thin outlines,
no gradients, one soft neutral drop shadow per card.

Keep the labels exactly as written, QUESTION, CONTEXT, ANSWER, 35 FEATURES,
EC-XGB, CALIBRATED RISK, SCORE, and no other text. If the generator cannot keep
seven chips, use five chips of equal size.
Negative: no realistic people, no photorealism, no dark background, no extra
words, no logos, no watermark.
```

Note: this image overlaps the Flow Diagram on slide 8. If the deck starts to feel repetitive, use the Introduction image here and generate this one only if the slide looks empty. Alternative use: the objective slide can reuse the pipeline image as a faint background band with the objective text over it.

### 5. Slide 5, Dataset, prompt 2D, optional

```
Create a flat vector illustration for a research presentation slide.

Scene: three dataset cards in one row on a white background, and one split bar
under them.
Card 1, thin blue border, tint blue fill: a small document stack icon and three
short grey text lines. Small slate label "HaluEval QA".
Card 2, thin teal border, tint teal fill: a small chat bubble icon and three
short grey text lines. Small slate label "RAGTruth".
Card 3, thin amber border, tint amber fill: a small summary page icon and three
short grey text lines. Small slate label "FaithBench".
Below the three cards, one long horizontal bar divided into three parts, blue
70 percent, teal 15 percent, amber 15 percent, with a thin border and no other
markings.

Composition: cards occupy the upper two thirds, the split bar the lower third.
Keep the top 15 percent of the canvas empty for a slide title bar.

Style: follow the STYLE CONTRACT. Flat fills, rounded corners, thin outlines.
Keep only the three dataset labels and no other text.
Negative: no realistic people, no photorealism, no dark background, no logos,
no watermark, no invented numbers.
```

### 6. Slide 12, Application, prompt 2E, optional

```
Create a flat vector illustration for a research presentation slide.

Scene: three use-case panels in one row on a white background, each in a
rounded rectangle with a thin slate border and a very light tint fill.
Left panel, tint blue: a chatbot window with a blue header bar, two grey message
lines, and one green check mark badge in the corner. Small slate label
"CHAT ASSISTANT".
Middle panel, tint teal: a document with a magnifying glass, three grey text
lines, and one amber highlight bar on the second line. Small slate label
"DOCUMENT QA".
Right panel, tint amber: a checklist with four rows, two rows marked with a
green check and two rows marked with a red cross, and a small red flag icon.
Small slate label "CONTENT REVIEW".

Composition: three equal panels, equal gaps, aligned and centered. Keep the top
15 percent of the canvas empty for a slide title bar.

Style: follow the STYLE CONTRACT. Flat fills, rounded corners, thin outlines,
no gradients, one soft neutral drop shadow per panel.
Keep only the three case labels and no other text.
Negative: no realistic people, no photorealism, no dark background, no extra
words, no logos, no watermark.
```

---

## Part 3. Output rules for generated images

- Format: PNG, 16:9, 2560 x 1440, white background, under 2 MB if possible.
- File names: `intro-answer-vs-evidence.png`, `motivation-three-gaps.png`, `objective-pipeline.png`, `dataset-sources.png`, `application-use-cases.png`.
- Save them into `web/public/slides/`.
- The deck crops them with object fit contain, so do not bake a border, a title, or a caption into the image; the slide adds those.
- Check each image against the style contract before adding it, palette, background, no extra text, no people.
- If an image breaks the contract, either regenerate it or switch to the no-text variant and add the labels in the deck.
