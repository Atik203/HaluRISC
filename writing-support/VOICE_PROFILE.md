# VOICE_PROFILE.md

- **Confidence:** low. No reliable author-written sample was found.
- **Samples used (author-written only):** none. The manuscript has been edited by agents across many
  sessions, `docs/prompt.md` is a style guide, and `README.md` / `roadmap.md` are project documents whose
  authorship is mixed. Using them would imitate agent prose, so they are excluded.
- **Samples excluded and why:** `Journal_Paper/halurisc.tex` (agent-edited), `Review_Checklist.md`
  (agent-maintained), `docs/prompt.md` (rules, not prose), chat messages (informal, not academic register).

## What to do at rung B

Use plain, neutral academic English. Keep the terminology lock from `PROJECT_BRIEF.md` §8. Do not invent a
personal voice, idioms, or stylistic quirks. Preserve the author's existing clear wording when an edit is not
needed, and keep the edits small.

| Feature | Observed in the author's writing |
|---|---|
| Person and voice | Not established. The paper uses impersonal prose and "this study"; keep that. |
| Tense habits | Past for runs and observations, present for facts and tables. Keep. |
| Sentence rhythm | Not established. Prefer 15–30 word sentences and split anything over 35 words. |
| Paragraph shape | One job per paragraph; the author's current draft mixes result batches in §5.4 and §5.8. |
| Connectors the author really uses | Not established. Use "however", "therefore", "in contrast" only for real relations. |
| Hedging style | Moderate: "stays near chance", "on par with", "does not transfer". Keep this strength. |
| How tables, figures, and numbers are introduced | Explicit table/figure reference, then the reading, then the numbers. Keep. |
| Preferred terms and spelling variety | US spelling ("calibration", "modeling" not used). EC-XGB, conventional XGBoost, HaluEval QA. |
| Formality and punctuation | Formal, no contractions, no semicolons in prose, no em dashes. Keep. |
| Recurring language issues to fix | Long stacked clauses, occasional vague "same"/"it"/"passes it", and paragraphs that carry two jobs. |

**Keep:** short declarative claims with a number attached, direct comparisons ("0.475 to 0.582"), and the
honest negative-result framing.

**Never copy:** the ambiguity patterns above, or any chat-register phrasing.

**Learned from the author's edits to agent output:** no recorded edits yet. Update this file when the author
rewrites a passage in their own words.
