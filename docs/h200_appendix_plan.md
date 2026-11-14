# H200 Appendix Plan — `uiu-h200` (`1g.18gb`, 18GB)

Follow `AGENTS.md` §6 (seeds 42/123/456, 5-fold CV where tuned, Platt on validation only) and §10 (ask-first, `uiu-h200` only, no host changes, project-local env, `tmux`/`screen`, 200GB cap, no secrets). Log every model/env change in `docs/DECISIONS.md` before confirmatory runs. Frozen `b6-ec-xgb-v1.0` stays intact; all new work writes to `artifacts/results/h200/` and is appendix-only.

## 0. Server check (paste-only, no changes)

```bash
nvidia-smi  # must show single MIG 1g.18gb slice
df -h; du -sh ~/halurisc  # stay under 200GB total
python3 --version  # 3.12 expected
```

## 1. Env setup (project-local only)

```bash
cd ~/halurisc  # project space, repo-root-relative paths only
python3 -m venv .venv-h200
source .venv-h200/bin/activate
pip install --upgrade pip
# torch CUDA 12.8 runtime works under driver 580.178.04 / CUDA 13.5; do NOT touch driver/CUDA
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
# keep xgboost==3.3.0, HALU_XGB_DEVICE=cpu (portability)
export HALU_XGB_DEVICE=cpu
export HF_HUB_CACHE=$PWD/.hf-cache  # keep cache inside project space
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Ollama (only if admin allows user-space install; else skip E3-big and run E3-4B via existing daemon):

```bash
ollama pull qwen3.5:4b
ollama pull qwen2.5:7b-instruct-q4_0    # ~4.5GB, fits 18GB
ollama pull qwen3.5:9b                 # fallback big model if 14B too slow or Q4 does not fit
ollama pull qwen2.5:14b-instruct-q4_0  # ~8.5GB, fits 18GB, 1g compute = slow
# DO NOT pull 32B+ (OOM on 18GB). concurrency=1 always for local judge.
```

## 2. Experiment matrix (run inside tmux/screen, stop when idle)

### E1 — Large-encoder ablation (highest reviewer value per hour)

Goal: answer "are results encoder-dependent?" Same 35 features, only encoder swap.

```bash
source .venv-h200/bin/activate
export HALU_XGB_DEVICE=cpu
# A: NLI large
HALU_NLI_MODEL=cross-encoder/nli-deberta-v3-large \
 python src/features/extract_features.py --device cuda --batch-size 256
python src/models/train_pipeline.py  # full §6: 70/15/15, 5-fold CV, seeds 42 123 456
# B: embed large (revert NLI first, change one variable at a time)
# edit src/features/semantic_features.py EMBEDDING_MODEL -> sentence-transformers/bge-base-en-v1.5
python src/features/extract_features.py --device cuda --batch-size 512
python src/models/train_pipeline.py
```

Outputs: `artifacts/results/h200/e1_{nli-large,bge-base}_{metrics,per_seed,stat_tests}.json`.
Expect: in-domain F1 +0 to +0.5% (NLI ablation cap -0.0040), MedHallu AUROC +1 to +3%. Report mean±std, McNemar, Wilcoxon. If delta < bootstrap CI width, report as null.

### E2 — Length-bias control (do this even if E1 skipped)

No model change. Length-matched slices of MedHallu/HaluBench + same rows for EC-XGB vs length-only vs judge.

```bash
source .venv-h200/bin/activate
python src/models/rebuild_judge.py --comparisons-only
python src/models/p1_halubench.py  # shard defaults device=cpu batch 64; use --device cuda --batch-size 64 on long contexts
```

Outputs: `artifacts/results/h200/e2_length_matched.csv`.
Expect: HaluBench 0.600 -> 0.62-0.65 max; length-only collapses on matched slices (cf. `tab:judge` 158-row control EC-XGB 0.907 vs length-only 0.476). This is the validity fix.

### E3 — Local judge at scale (Qwen big)

```bash
source .venv-h200/bin/activate
# 4B baseline (already scripted, concurrency 1)
python src/models/rebuild_judge.py --backend qwen --limit 0 --repeats 3 --concurrency 1 --out-tag full
# 7B: set QWEN_MODEL / QWEN_URL per src/models/rebuild_judge.py:96-98, review_llm_judge_qwen.py:33
QWEN_MODEL=qwen2.5:7b-instruct-q4_0 python src/models/rebuild_judge.py --backend qwen --limit 0 --repeats 3 --concurrency 1 --out-tag qwen7b
# 9B fallback if 14B too slow or Q4 does not fit: fewer params than 14B, fits 18GB quantized
ollama pull qwen3.5:9b
QWEN_MODEL=qwen3.5:9b python src/models/rebuild_judge.py --backend qwen --limit 0 --repeats 3 --concurrency 1 --out-tag qwen9b
# 14B pilot last (slowest on 1g slice): 200 rows, 1 repeat first
QWEN_MODEL=qwen2.5:14b-instruct-q4_0 python src/models/rebuild_judge.py --backend qwen --limit 200 --repeats 1 --concurrency 1 --out-tag qwen14b-pilot
```

Outputs: `artifacts/results/rebuild/judge/*qwen7b*, *qwen9b*, *qwen14b*` (copy to `h200/`).
Expect: F1 flat vs 4B (0.60-0.65 natural), win is $0/1K + reproducibility + p50 latency. Order: 7B -> 9B -> 14B-pilot. Stop 14B if tokens/s < 5 or OOM; use 9B as the big-model result; record choice in DECISIONS.md.

### E4 — Efficiency report (one figure, no paper-number risk)

```bash
source .venv-h200/bin/activate
python src/models/rebuild_judge.py --time-encoder  # RoBERTa-base 200 rows batch 8 fp16, peak_gpu_memory_mb
# batch sweep (record wall time + peak VRAM each):
python src/features/extract_features.py --device cuda --batch-size 128
python src/features/extract_features.py --device cuda --batch-size 256
python src/features/extract_features.py --device cuda --batch-size 512
```

Outputs: `artifacts/results/h200/e4_efficiency.csv` (batch, rows/s, peak MB, p50 ms).

## 3. Verify before touching paper

```bash
source .venv-h200/bin/activate
python -m pytest tests -v
python src/models/verify_artifacts.py
python src/models/rebuild_number_audit.py
python src/models/review_number_audit.py
python src/models/p1_number_audit.py
python src/models/float_proximity_audit.py
```

Gate: new artifacts load + predict on this machine; 116/298/108 + 29/29 floats stay green for frozen numbers.

## 4. Paper delta (appendix-only, keep 24pp from growing)

- Text to add: ~600-900 words total. Appendix (~450-650 words: E1 setup + E2 matched-slice result + E3 judge cost + E4 batch table) + §5.5/§5.6 one paragraph each (~150-250 words). No changes to §§1-4, no renumbering of Eqs. 1-17.
- Text to update: 3-5 sentences only (E2 length-matched attribution for Tab. 18, E3 $/latency line, E4 batch footnote). Keep author wording; fix numbers/claims/citations only.
- Tables: 1 new (Appx. Tab. A1: encoder ablation mean±std + McNemar p; E2 matched-slice F1/AUROC; judge $/latency). Optionally extend Tab. 11 with one matched-slice row. No edits to Tabs. 1-18 values.
- Figures: 1 new (Appx. Fig. A1: batch-size vs rows/s + peak VRAM, or Qwen 4B/7B/9B/14B cost-accuracy scatter). Reuse `plot_m3_shap.py` label map if SHAP re-rendered. No changes to Figs. 1-11.
- Build: `latexmk -pdf -outdir=out halurisc.tex` from `Journal_Paper/`; check 0 undefined, 0 overfull, floats within 2pp.

## 5. Stop conditions

Stop E1 if in-domain delta < 0.002 after seed 42 pilot. Stop E3-14B on OOM or <5 tok/s and fall back to 9B as the big-model result. Stop any run exceeding 150GB project total. Record all stops in `docs/DECISIONS.md`.
