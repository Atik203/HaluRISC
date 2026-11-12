# Fix I-01: Apply `log1p` length scaling on every EC-XGB serving path

## What changed and why

EC-XGB m3 (`b6-ec-xgb-v1.0`) is trained and calibrated on `log1p(n_chars)` and
`log1p(n_words)`; the training helper is `run_b6_modified.feature_matrix(...,
log_lengths=True)`. The API built the EC matrix from the raw
`extract_all_features_single` counts in `/predict`, `/explain`, `/analyze`, and
`/predict/compare`, so the deployed booster and its Platt display calibrator
were scoring inputs on a different scale from the checkpoint. The mismatch was
largest for short answers and also corrupted SHAP values, the EC row in
`/predict/compare`, and any UI or paper numbers produced through the API.

The fix adds one serving helper, `_ec_feature_matrix`, that builds the row in
the exact `ec["feature_cols"]` order and applies
`np.log1p(np.clip(x, 0.0, None))` only to `n_chars` and `n_words`. Every EC
path now calls it. The B2 model, logistic regression, random forest, and the
overlap heuristic still consume raw counts.

## Files and line-level summary

`src/api/main.py`

- `:445-464` — `EC_LOG_LENGTH_COLUMNS = ("n_chars", "n_words")` and the new
  `_ec_feature_matrix(feats, feature_cols)` helper. It mirrors
  `run_b6_modified.feature_matrix(..., log_lengths=True)` and preserves the
  caller's feature order.
- `:761-765` — `/predict` builds the EC matrix with the helper. The B2 legacy
  matrix at `:756-758` remains raw.
- `:793-813` — `/explain` feeds SHAP the log-scaled EC matrix when the active
  explainer is the EC-XGB explainer. The reported `FeatureImpact.value` now
  comes from the raw `feats` entry (`:810`) so the explanation card stays
  consistent with `prediction.features`; only the SHAP impact is computed on
  the transformed model input. The non-EC branch keeps raw B2 values.
- `:851-855` — `/predict/compare` builds the `ec_xgb` row with the helper. The
  standard XGBoost, random forest, logistic regression, and heuristic rows are
  unchanged and raw (`:849`, `:860-874`).
- `:826-828` (`/analyze`) and `:1047-1049` (`/verify`) need no separate change:
  they call `predict_risk` / `explain_risk`, so both inherit the EC transform.

`tests/test_api.py`

- `:133-143` — `FakeRaw` now records the matrix it was scored on (`last_X`).
- `:275-313` — `test_ec_matrix_applies_log1p_but_b2_stays_raw` posts to
  `/predict` and `/predict/compare` with a fake EC bundle whose feature order
  is deliberately shuffled. It asserts the EC matrix is
  `[log1p(100), 1.0, log1p(20), 0.9]` while the B2 matrix stays
  `[100, 20, 0.9]`.
- `:321-334` — `test_ec_feature_matrix_matches_training_transform` compares the
  serving helper directly with
  `run_b6_modified.feature_matrix(..., log_lengths=True)`.
- `:337-374` — `test_explain_uses_ec_log1p_matrix` asserts SHAP receives the
  log-scaled EC matrix in EC column order and that displayed length values stay
  raw.

No models, artifacts, manuscript files, or figure files were modified. No
retraining was performed.

## Test command and result

From the repo root:

```powershell
& .venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

Result: **29 passed, 7 warnings in 42.49s** (exit code 0). The warnings are the
pre-existing Starlette/httpx deprecation notices, an XGBoost deserialization
notice, a SHAP colormap pending-deprecation notice, and a torch
`expandable_segments` platform notice. The focused subset also passed
independently: **3 passed, 26 deselected in 6.71s** for
`-k "ec_matrix or explain_uses_ec or matches_training"`.

## Remaining risk

- **SHAP units:** impacts for `n_chars` and `n_words` are now correctly computed
  on `log1p` inputs, but a SHAP point is a change in log-count, not a change of
  one character or one word. The API reports raw counts as the feature value,
  which is good for display but should not be read as a per-unit attribution.
  A frontend tooltip or caption saying "length features are log-scaled for the
  model" would remove the ambiguity.
- **Stale screenshots and paper figures:** any UI capture or Figure 3 generated
  through the old API path shows pre-fix scores. They were not regenerated here
  (manuscript and artifacts are out of scope for this fix) and should be
  recaptured before submission.
- **Changed deployed outputs:** `risk_score`, `calibrated_score`, labels, and
  EC SHAP values will move for existing inputs, especially short answers. Any
  recorded demo numbers, snapshot tests outside `tests/test_api.py`, or cached
  frontend payloads should be refreshed.
- **Feature cache:** `FEATURE_CACHE` stores raw extracted features, not matrices,
  so existing cache entries remain valid and no invalidation is needed. The
  transform is applied after cache lookup.
- **Transform duplication:** serving mirrors the training helper rather than
  loading a persisted preprocessing pipeline. The new parity test pins the two
  implementations together, but a future change to
  `run_b6_modified.LOG_LENGTH_COLUMNS` or its transform must be mirrored in
  `src/api/main.py`.
- **Out of scope:** the EC source indicator remains fixed at `1.0` for every
  request (review issue I-05), and checkpoint naming/identity (I-02) is
  unchanged. Neither is affected by this fix.
