# Hallucination Risk Prediction via Classical ML

Hallucination detection is an active research area: classical ML classifiers can indeed predict when an LLM’s output contains fabricated or unsupported content.  Recent studies show that lightweight models (e.g. Random Forest, XGBoost, SVM) trained on textual features can achieve strong accuracy. For example, Reed & Mason (2025) evaluated an SVM with TF-IDF and length/overlap features on the **HaluEval** benchmark (64,507 examples across QA, dialogue, summarization, and general queries) and obtained an F1≈0.751.  Likewise, HalluShield (2024) extracted features from Llama-3 activations and trained XGBoost, reporting an **F1 of 0.89** on a medical-response dataset.  In a mental-health dialog dataset, a Random Forest/XGBoost ensemble even reached F1≈0.849.  These results demonstrate classical classifiers can rival or exceed expensive “LLM-as-judge” methods, while being far faster and cheaper.

## Existing Research & Datasets

- **HaluEval (2024):** A large-scale hallucination benchmark (64.5K samples across four domains) with human labels. Reed & Mason (2025) used this dataset with a linear SVM and reported AUROC=0.835, F1=0.751. It covers both factual QA and free-form dialogue.  
- **TruthfulQA (2022):** A curated QA set designed to expose model falsehoods. Many papers (e.g. Lin et al.) use it to gauge hallucination tendencies. Although smaller (~817 Qs), it’s a useful synthetic benchmark.  
- **Domain-specific datasets:** Recent papers have created specialized corpora. For instance, a mental-health chatbot dataset (with ≈4K annotated responses) achieved F1≈0.717 (custom test) and 0.849 on a public subset. In medicine, HalluShield built a “Med-Hallu” set of truthful vs. hallucinated answers, achieving 89% F1 with XGBoost.  
- **Other resources:** Emerging benchmarks like *DefAn* and multilingual hallucination sets (e.g. *HaluBench*) are also available. In practice, you can start with HaluEval (general domain) and optionally collect a small in-domain subset (e.g. legal, medical, code) to fine-tune the detector’s focus.

Most current work emphasizes **one-pass detection** from a single LLM output. Unlike multi-query methods or white-box probing, these rely only on black-box signals: text length, overlap with input context, presence of uncertain wording (“possibly”, “I think”), citation patterns, etc. This is exactly the approach we plan.

## Feature Extraction Strategy

To make the project both simple and novel, focus on *black-box features* that do not require access to LLM internals. Potential features include:

- **Length & Overlap:** Answer length, token count, ratio of words overlapping with the prompt or known context. (Reed & Mason found context-overlap strongly reduces hallucination risk.)
- **Citation/Evidence Count:** Presence of numbers or references (if the model can cite). Hallucinations often lack verifiable citations.  
- **Linguistic Uncertainty:** Frequency of hedging words (“maybe”, “might”, “probably”). High uncertainty can signal guesswork.  
- **Semantic Consistency:** Use a lightweight entailment check (e.g. a small NLI model) to detect contradictions within the answer. A high contradiction score can indicate hallucination.  
- **Embedding-based Features:** Compute sentence embeddings (e.g. with a small model) and measure semantic similarity of consecutive sentences; abrupt semantic jumps may hint at hallucination.  

All features must be computed quickly on the generated text. No additional LLM queries or hidden-state access is needed. In our implementation, we will script extraction of these features in Python (spaCy for entity counts, NLTK for sentence count, Hugging Face transformers for tokenization or small entailment model, etc.). The feature vector for each response can be fed into the classifier.

## Model Training & Evaluation

We will train **Random Forest, XGBoost, and CatBoost** classifiers on these features. The steps are:

1. **Prepare labeled data:** Use HaluEval (which is split into train/val/test) and/or any domain-specific labeled examples we can gather. If needed, label a few hundred examples ourselves (or use an existing tool like Fake News vs. Fact).  
2. **Cross-validation:** Perform k-fold cross-val to compare models. Key metrics: **Precision, Recall, F1, ROC-AUC**. Hallucination is often the minority class, so emphasize F1 and also **Matthews Correlation Coefficient (MCC)** for robustness.  
3. **Ablation Studies:** After finding a best model, remove features one at a time to see performance drops. This identifies which signals matter most. (For instance, Reed & Mason noted features like “context overlap” had large weights.)  
4. **Explainability:** We will integrate SHAP values to make the model explainable. For each prediction, SHAP can show which features (e.g. “zero citations”, “high uncertainty terms”) contributed most to a high hallucination score. This directly addresses interpretability and is a strong differentiator.

By using 30k+ HaluEval examples for training/validation and possibly fine-tuning on a smaller in-domain set, we can mimic the workflows of published papers. The computational cost is minimal (classical models train in seconds to minutes), so we can experiment with many feature sets.

## Novelty & Contribution

To stand out as research (and not just an implementation), our project should highlight these innovations:

- **Single-pass, Black-Box Focus:** Unlike methods that sample multiple LLM outputs or inspect hidden states, our model flags hallucination risk from one response using only surface features. This makes it applicable to closed-source LLMs (GPT, Claude) and low-latency systems.  
- **Domain Adaptation:** We will test both *general* (HaluEval) and *domain-specific* scenarios. For example, one variant could be tuned on a small **legal QA** dataset, or **medical dialogues**. Past work has shown detection can improve by focusing on domain-specific hallucination patterns.  
- **Explainable SHAP Output:** Most research papers report only accuracy. We will go further by displaying SHAP-based explanations to users. For instance: *“Hallucination risk = 78%. Top features: high hedging (likes 3 hedge words), no citations, high answer/question similarity.”* This transparency is rarely shown in other demos.  
- **Real-Time Demo with Modern UI:** We will deploy the model as a web app (e.g. using React+TypeScript frontend + FastAPI backend) so that users (and graders) can paste in text and see immediate risk scores and explanations. Practical usability is often missing in academic works.

## Datasets and Tools

- **HaluEval**: Download the HaluEval snapshot (64K examples) and its train/val/test splits. Ensure we randomize or use the provided splits to avoid leakage.  
- **Custom Data (optional)**: If feasible, gather ~500 examples of LLM answers in a specific domain. For example, use GPT-4o to generate answers on law or medicine prompts, then label them (or use a smaller retrieval-augmented model to fact-check and label). This will be valuable but optional for the term deadline.  
- **Libraries**: Use `scikit-learn`, `xgboost`, `catboost`, `fastapi` (for API) on the backend. In frontend, use React/TypeScript with a charting library for the risk gauge and explanation bar chart.

## Experimentation and Baselines

- **Baselines**: Compare to a simple baseline (e.g. “flag any answer longer than X or containing no quotes as hallucination”). Also compare to an LLM-as-judge approach: e.g. send the answer to GPT-4 with a “Fact-check this” prompt and see its accuracy vs. cost. This will highlight our advantage in speed and cost.  
- **Metrics**: Report Precision/Recall/F1 for each class, plus ROC-AUC. Since HaluEval is imbalanced (some subsets only ~18% hallucinated), also report MCC. We should show confusion matrices.  
- **Latency/Cost**: Measure average inference time (ms) and API calls. A simple table like “XGBoost: 5ms vs GPT-4: 500ms per query” will be a strong selling point.

## Implementation Roadmap (2–3 months)

1. **Weeks 1–2 – Data Preparation:** Download HaluEval and parse it into input (question/context), output (model answer), hallucination label. Write feature-extraction scripts (sentence count, token count, keyword presence). Start with basic textual features; refine as you go.  
2. **Weeks 3–4 – Model Training:** Train RF, XGBoost, CatBoost on the extracted features. Use cross-validation to tune hyperparameters (e.g. tree depth, number of estimators). Evaluate on a held-out set. Compare performance; pick a winner.  
3. **Week 5 – Explainability:** Apply SHAP to the final model. Validate that the SHAP scores are sensible (e.g. “no overlap” raises risk). Prepare visualizations (bar charts of top features) for sample cases. Conduct ablation: remove one feature group and show drop in F1 to identify its impact.  
4. **Weeks 6–7 – Frontend Demo:** Build a React UI: a text input for a prompt/answer (or just answer text), a “Check Hallucination” button, and outputs: risk gauge (green/yellow/red), numeric risk %, and a SHAP bar chart explaining the decision. The backend (FastAPI) will run the feature pipeline and model in real time. Test end-to-end on various examples (open QA, factual vs fictional queries).  
5. **Week 8 – Write-Up & Refinement:** Draft the paper in journal format (IEEE/ACM style). Sections: Introduction (motivation), Related Work (cite above papers), Method (features + model), Experiments (results, metrics, ablations), Demo description (screenshots), Conclusion (future work). Iterate on clarity. Make sure to clearly state novelty (single-pass, explainability, deployment).  

This plan is realistic for a 2–3 month term project. The ML training is fast, so the main time will be data engineering and writing. Even a simple prototype can be ready by week 6, leaving weeks 7-8 for polishing results and writing. The UI can be minimal (e.g. a single-page app with plots) but should emphasize the real-time aspect.

## Feasibility in 2–3 Months

Yes — a basic demo can definitely be built in 2–3 months. Using existing tools (scikit-learn, SHAP, React starter templates) speeds things up. The core novelty (feature extraction + classical ML) is straightforward to implement. Even if time is short, an initial paper-style report and partial demo (e.g. a static set of examples with predictions) can be completed. Additional polish (domain data, extensive UX design) can come later. The project scope is intentionally scoped for an undergraduate timeline: it **demonstrates a concrete research idea with solid results** without requiring long LLM training loops. 

Overall, this project idea is both practical and research-worthy: it builds on 2024–2026 literature, adds a clear novel twist (single-pass explainability), and yields measurable results plus a working prototype by term’s end. It fits the course constraints and should earn high marks in a journal-style paper and demo. 

**Sources:** Recent studies and benchmarks of LLM hallucination detection have informed this proposal.