# Figure 2 — UI Composite Mockup (Google Stitch design brief)

Use this brief to build the user-interface figure for the **System** chapter
(`6.system.tex`) in **Google Stitch** (or draw.io / PowerPoint / Canva).

## Output requirements

- **Canvas:** 16:9 landscape, 2400 x 1350 px, PNG, 300 DPI
- **Save as:** `report/figures/ui_figure.png`
- **Style:** clean product mockup, same palette as the app (dark theme cards
  on a light figure background is fine; rounded corners, thin borders,
  realistic but simplified UI). All visible text short.

## Color palette

| Element | Hex |
|---|---|
| Cards / panels | white `#ffffff` with slate border `#cbd5e1` |
| Accent (brand) | violet `#7c3aed` / indigo `#4f46e5` |
| Low risk | emerald `#059669` |
| Medium risk | amber `#d97706` |
| High risk | rose `#e11d48` |
| Background | light gray `#f8fafc` |

## Layout — three panels side by side

**Panel A — Chat with auto-risk card (top of a chat window):**

- A small assistant message bubble with two lines of gray placeholder text.
- Below it the auto-risk card:
  - a small gauge or pill labeled `Low risk` in green,
  - a line of claim verdict chips: `supported (2)` `contradicted (1)`,
  - one chip row showing `Evidence says: ...` in red,
  - a source link line: `source`.
- A chat input bar at the bottom with a send button.

**Panel B — Analyze compare view:**

- Two side-by-side result cards labeled `Answer A` and `Answer B`.
- Each card has a semicircular gauge (one green, one red), a score line
  (`12%` / `88%`), a label (`Low risk` / `High risk`), and two small SHAP bar
  rows.
- Shared input fields above the two cards: `Question`, `Context`, `Answer A`,
  `Answer B`.

**Panel C — Evidence panel:**

- A box titled `Evidence context (optional)` with:
  - a small textarea (two gray lines),
  - an upload button labeled `Upload documents (PDF/DOCX/TXT)`,
  - a toggle labeled `Search the web (Tavily)`,
  - a status line: `3 indexed passages`.
- A small checkmark row: `Auto risk check (per answer)`.

**Overall labels (small, top of figure):**

- Panel titles: `Conversational chat`, `Compare mode`, `Evidence sources`.

## Step-by-step in Stitch

1. Open a blank Stitch canvas and paste this brief as the design prompt.
   Say "build this 3-panel UI mockup, 16:9 landscape".
2. Replace any placeholder text Stitch invents with the labels above.
3. Keep panels in one row (three columns) with equal spacing.
4. Export File -> Export / Download as PNG (300 DPI).
5. Save to `report/figures/ui_figure.png`.

## Alternative option (faster)

If the mockup feels unnecessary, use a **screenshot grid** instead: place the
existing real screenshots side by side in one image (PowerPoint / Google
Slides / Stitch):

- `report/screenshots/hallucinated_date.png`
- `report/screenshots/grounded_correct.png`
- `report/screenshots/borderline.png`

Caption suggestion: "Analyze page outputs for a hallucinated, a grounded, and
a borderline answer." This option is always valid and uses real product
screenshots.

## Quality checklist

- [ ] Three panels in one row, roughly equal width
- [ ] Panel A shows the auto-risk card with verdict chips and `Evidence says`
- [ ] Panel B shows two gauges with different risk levels
- [ ] Panel C shows upload + web toggle + passage count
- [ ] No fake long paragraphs; all text short and spelled correctly
- [ ] Landscape 16:9
