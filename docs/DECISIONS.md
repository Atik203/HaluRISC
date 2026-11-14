# Decisions Log

Record of approved hardware, driver, CUDA, or environment changes (required by `AGENTS.md` §10 before confirmatory work).

## 2026-11-14 — University GPU access approved

- Access: SSH via local alias `uiu-h200` only.
- GPU: H200 MIG `1g.18gb` slice, 18 GB VRAM maximum.
- Driver `580.178.04`, CUDA `13.5`. No changes approved.
- Storage cap: 200 GB, project space only.
- Python: project-local venv or conda env only.
- No approved deviations from §10 at this time.

## 2026-11-14 — H200 appendix environment notes

- Runtime: user-space `uv` + managed CPython 3.12 venv at `~/halurisc/.venv-h200`. No system
  packages and no `sudo` (the node has no `python3-venv` and no `pip`).
- All caches stay inside the project space: `.hf-cache/` (Hugging Face), `.tools/` (uv, managed
  Python, Ollama binary).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **not** used on this MIG slice. It makes
  `.to("cuda")` fail with "CUDA driver error: operation not supported". Expandable segments are
  unsupported on `1g.18gb`. The 6 GB API path keeps the setting; batch scripts here omit it.
- Observed VRAM ceiling: `cross-encoder/nli-deberta-v3-large` at batch 256 peaks at ~15.6 GB of
  the 15.9 GB usable slice memory. Large encoders fit, but batch 256 is the practical maximum.
- Ollama (approved 2026-11-14): user-space binary in `~/halurisc/.tools`, `ollama serve` bound to
  `127.0.0.1:11434` only, models under `~/halurisc/.ollama`. Stopped and deleted at cleanup.
- Frozen artifacts untouched: all runs write to `artifacts/results/h200/` via
  `HALU_FEATURES_FULL`, `HALU_CLAIM_FEATURES`, `HALU_EXTERNAL_FEATURES`, and `--results-dir`.
- Confirmatory runs use fp16 for the encoder features (unchanged from the frozen pipeline).

## 2026-11-15 — No page limit for the manuscript

- The author confirmed there is no page limit. The 18-page budget recorded in
  `docs/writing-guideline.md` (2026-11-05) is treated as superseded, and the H200 appendix is not
  trimmed to fit it. Current build is 26 pages with Appendix A included.
- Do not delete appendix content to chase a page count. Do not add length either without a reason
  that survives the number audits.

## 2026-11-15 - H200 run complete

- E1 encoder ablation: base / DeBERTa-large / BGE-base, m0-m2, seeds 42/123/456. All
  differences non-significant (McNemar p = 0.63 and 0.77). Results in
  `artifacts/results/h200/encoder_ablation.json`.
- E2 length-matched external control: frozen m3 read as-is, no retraining. Results in
  `artifacts/results/h200/length_matched.json`.
- E3 local judge: Qwen3.5-9B and 4B both measured on the slice, 2,700 calls each, zero
  failures. `artifacts/results/h200/judge_qwen{9b,4b}_metrics.json`.
- E4 throughput and VRAM: `artifacts/results/h200/efficiency.csv`. Batch 128 is the
  slice optimum at 361 rows/s and 3.2 GB.
- No appendix. The four controls sit in Sections 5.1, 5.5, and 5.6 (Table 6, Figure 10,
  Table 19, Table 21).
