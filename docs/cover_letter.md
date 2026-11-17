# Cover letter

**[Journal Name]**
Editor-in-Chief

Dear Editor,

Please consider our manuscript, **"Calibrated Hallucination Risk for Large Language Models Under Domain and Task Shift with Evidence-Consistent XGBoost,"** for publication in *[Journal Name]* as an Original Research Article.

The paper asks whether a small, interpretable feature model can score hallucination risk and keep its ranking and calibration under domain and task shift. We introduce EC-XGB, a 35-feature XGBoost model built on base features, claim-level natural language inference aggregates, and a source indicator. The model is trained on HaluEval QA and RAGTruth non-QA rows, and evaluated with fixed-hyperparameter ablations, source and target calibration, conformal prediction, SHAP stability analysis, a length-only control, and judge stress tests.

The main results are:

- EC-XGB reaches F1 0.9854 and AUROC 0.9986 on HaluEval QA, but lexical overlap and length carry most of that score.
- On MedHallu it ranks the hallucinated answer above the grounded one in 86.7% of 1,000 pairs, yet it flags every row and is poorly calibrated (ECE 0.493).
- Target Platt calibration lowers ECE from 0.731 to 0.125 on RAGTruth QA without changing the ranking.
- The encoder choice is a null result: base, large-NLI, and BGE encoders differ by 0.0022 F1 or less.
- A local 9B judge does not close the gap to an API judge on natural QA.

The contribution is a careful separation of ranking from probability, with controls for answer length and domain shift. We report the negative results as well, including the failure of the display score under task shift.

This work is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the manuscript and agree with its submission. The authors declare no competing interests, and the research received no specific grant from any funding agency. Data and code availability statements are included in the Declarations section.

Thank you for your consideration.

Sincerely,

**Md. Atikur Rahaman** (corresponding author)
Computer Science and Engineering, United International University, Dhaka 1212, Bangladesh
mrahaman2310298@bscse.uiu.ac.bd

Co-authors: Saiful Alam Sabbir, MD. Miraz Ahamed, Ohidujjaman Tuhin
