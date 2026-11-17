# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-18

### Added

- Evidence-consistent XGBoost scorer with 35 features, including claim-level NLI
  aggregates and a source indicator.
- Grouped 70/15/15 splits, three-seed protocol, five-fold tuning, calibration,
  conformal prediction, SHAP stability analysis, and judge stress tests.
- FastAPI inference service with `/predict`, `/explain`, `/verify`, and `/health`.
- Next.js dashboard for analysis, calibration, efficiency, explainability,
  failures, robustness, and overview pages.
- Reproduction scripts, pinned dependencies, tests, citation metadata,
  contribution guide, security policy, and code of conduct.

[1.0.0]: https://github.com/Atik203/HaluRISC/releases/tag/v1.0.0
