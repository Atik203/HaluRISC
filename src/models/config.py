"""
HaluRISC single source of truth for experiment configuration
(blueprint A18: "All random seeds documented in a single config file").

Imported by train_pipeline, shap_analysis, error_analysis, eval_llm_judge,
eval_efficiency — never redefine seeds elsewhere.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Experiment seeds (blueprint A9: repeat every experiment with 42, 123, 456)
SEEDS = [42, 123, 456]

# Bootstrap / sampling seeds (fixed, separate from experiment seeds)
BOOTSTRAP_SEED = 777
SAMPLE_SEED = 42

# Counts
N_BOOTSTRAP = 1000

# Paths
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DATA_PROCESSED = ROOT / "data" / "processed"

FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
FEATURES_FALLBACK = DATA_PROCESSED / "features_core.parquet"
QA_CLEAN = DATA_PROCESSED / "qa_clean.parquet"
