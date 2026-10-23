# HaluRISC — Journal Writing Guideline (extract)

Source: `docs/writing-guide.pdf` (38 pages, supervisor's guideline).
Extracted: 2026-10-17, for use together with `docs/prompt.md`.
`docs/prompt.md` controls the voice (student register, anti-AI patterns). This
page controls the structure and the content of each section of the journal
paper (`Journal_Paper/halurisc.tex`).

Last reviewed: 2026-10-23 against the 17-page build in `Journal_Paper/out/`.
Status of every rule is tracked in `G0` and in the mapping section at the end.

---

## How to use this file (Agent Instructions)

> **Agent note (not part of the supervisor's guideline):** Every block that is not marked "Agent note" is the wording of the supervisor's PDF, except section headings, short connecting lines, and minor grammar fixes.
>
> - **Stable IDs (`G1` to `G15`):** Reference these IDs when drafting or auditing specific sections of `Journal_Paper/halurisc.tex`.
> - **Checklist Execution:** Within each section, treat the bullet lists as a checklist: cover every applicable bullet in the order given.
> - **Conditional and illustrative items:** Items marked "if applicable", "where appropriate", "if provided", "may include", "may contribute to", or "for example" are conditional or illustrative, not required. Use only the ones that fit the paper. This covers the keyword examples in `G3`, the two alternative abstract journeys in `G2`, the author-contribution roles in `G12`, and the acknowledgement types in `G14`.
> - **Required Sequences:** Any line labeled **Recommended Flow**, **Recommended structure**, **Overall Flow**, or **A Simple Formula** defines the sentence/subsection ordering to follow.
> - **Guiding Prompts:** Every question (e.g., _"What is missing?"_) is a direct prompt that the corresponding sentences in the manuscript must answer.
> - **Dynamic Section Numbering:** Section numbers like `4.1` or `5.2` below are for a method-based paper; when the article type is different, keep the sub-section order and renumber accordingly.

**Normative keywords (agent-facing):**

- **MUST** — hard requirement. A draft that misses it is not submission-ready.
- **SHOULD** — recommended by the supervisor. Deviate only with a recorded reason.
- **MAY** — optional, use only when the paper has the material.
- The supervisor's original wording is descriptive. The agent-facing strength of each rule is stated in `G0`, and `G0` wins when the two differ.

**Precedence when rules conflict:**

1. Journal template (CAS single column) and page limit.
2. A dated supervisor instruction, recorded under "Supervisor overrides" below.
3. This file, in `G0` → `G15` order.
4. `docs/prompt.md` voice preferences, where this file says nothing.

**Agent workflow (MUST):**

1. Read `docs/prompt.md` before drafting or editing prose.
2. Work through `G0` to `G15` in order. Do not jump to a later section while an earlier `MUST` is unmet.
3. Run the audit procedure below after every build. Never mark a `PASS` from memory or from an earlier draft.
4. Update `G0` and the mapping section with the new status and the new build facts.
5. Record every deviation from `SHOULD` in `G0` with its reason, so the next agent does not "fix" it back.

---

## Section order by article type

| Article type                     | Section order                                                                                                                                                                                                                                                                                                             |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Method based**                 | `1 Introduction` → `2 Conventional method/methods` → `3 Proposed method` → `4 Experiments (4.1 Experimental setup, 4.2 Corpus/Dataset)` → `5 Results` → `6 Conclusion` → `Author contribution` → `Declaration` → `Acknowledgements` → `Reference`                                                                         |
| **Dataset based (Corpus based)** | `1 Introduction` → `2 Dataset/Corpus (2.1 Data collection, 2.2 details with several subsections)` → `3 Conventional method(s) (if applicable)` → `4 Proposed method` → `5 Experiments (5.1 Experimental setup)` → `6 Results` → `7 Conclusion` → `Author contribution` → `Declaration` → `Acknowledgements` → `Reference` |
| **Corpus + method based**        | `1 Introduction` → `2 Dataset/Corpus (2.1 Data collection, 2.2 details with several subsections)` → `3 Conventional method/methods` → `4 Proposed method` → `5 Experiments (5.1 Experimental setup)` → `6 Results` → `7 Conclusion` → `Author contribution` → `Declaration` → `Acknowledgements` → `Reference`            |

> **Agent note (not part of the supervisor's guideline):**
>
> - **Which row applies to `halurisc.tex`:** The last section of this file, "How the HaluRISC paper maps to this guideline", decides. It follows the method-based order, with the datasets inside the Experiment section. Use the "Method based" row for `Journal_Paper/halurisc.tex`.
> - **Declaration:** The organization slides list only Funding, Data Availability Statement, and Competing Interests. Use `G13` as the full Declaration list.

---

## G0. Compliance matrix (living)

> **Agent note:** This table is the single source of truth for "is the paper compliant". Update it after every build. `PASS` requires evidence from the current `Journal_Paper/out/halurisc.pdf`, not from an earlier draft.

| ID | Requirement | Mechanical check | Evidence anchor | Status (2026-10-23) |
| :-- | :--- | :--- | :--- | :--- |
| G1 | Title is clear, specific, and free of unexplained abbreviations | title names the task and the method | `\title` | PASS |
| G2 | Abstract is 250–300 words, one paragraph, six questions in order, headline numbers only | word count, sentence-to-question map | `abstract` | PASS (276 words) |
| G3 | 4–8 specific keywords covering topic, method, problem, domain | count `\sep` + 1 | `keywords` | PASS (7) |
| G4 | Introduction is exactly four paragraphs in P1–P4 order | paragraph count, then read each paragraph's job | `sec:intro` | PASS |
| G5 | The article follows the method-based section order | compare heading list with the table above | whole file | PASS |
| G6 | Dataset/Corpus section only for dataset-based or corpus+method papers | confirm article type | `sec:experiment` (dataset inside §4.2) | N/A |
| G7 | Conventional Method gives baseline, why it was selected, working principle, formulation, algorithm, characteristics, limitation, gap | subsection list plus one sentence on selection | `sec:conventional` | PASS |
| G8 | Proposed Method gives working principle, formulation, algorithm, and contribution | subsection list | `sec:proposed` | PASS |
| G9 | Experiments give setup, dataset, procedure, parameters, metrics | §4.1–§4.5 titles in order | `sec:experiment` | PASS (`§4.6` is an extra protocol block) |
| G10 | Results cover metric-wise results, comparison, ablation, statistics, findings | map each results subsection to one bullet | `sec:results` | PASS with deviation (see below) |
| G11 | Conclusion gives objective, contribution, findings, significance, future work, and no new references | paragraph map, citation scan | `sec:conclusion` | PASS |
| G12 | Every author has CRediT roles | one `\credit{}` per author | author block | PASS |
| G13 | Declaration lists funding, conflict of interest, data, code, ethics | item count and labels | Declarations | PASS |
| G14 | Acknowledgements name non-author support | section present | Acknowledgements | PASS |
| G15 | Every citation appears in the list, every entry is cited, style is uniform | bib cross-check | `ref.bib` | PASS (34/34) |

### Recorded deviations (do not "fix" these back)

- **G10 order (2026-10-21):** The supervisor removed the standalone limitations subsection. The Results order is comparison → components → calibration → transfer → ablation → explanations → efficiency → error analysis → discussion. Statistical tests sit inside §5.1, §5.2, and §5.5, and §5.9 is interpretation only. The limitations that `G10` asks for stay inline: encoder comparison (§5.1), FaithBench F1 drop (§5.2), held-out ranking near chance (§5.3), length-shortcut control (§5.8), and the negative-results paragraph in the Conclusion.
- **Terminology (2026-10-21):** The baseline model is always "conventional" or "conventional XGBoost", matching the §2 title. Never call it "the standard model". Generic uses such as "standard metrics", "standard logistic loss", and "standard deviation" remain correct.
- **Abstract style (2026-10-23):** The abstract is 276 words, inside the 250–300 band but above the 250 target. It stays there because every sentence answers a required question and the extra words are the cost, latency, and calibration results, not filler.
- **Page budget (2026-10-21):** 17 pages in CAS single column. Do not add length without supervisor approval.

### Audit procedure (MUST run after every edit)

```powershell
# from the repository root
cd Journal_Paper
latexmk -pdf -interaction=nonstopmode -outdir=out halurisc.tex
cd ..

# numbers in the paper against frozen artifacts
& .venv\Scripts\python.exe src\models\review_number_audit.py

# test suite
& .venv\Scripts\python.exe -m pytest tests -q
```

Then check, in this order:

1. `pages = 17`, and no reference block spills onto page 18.
2. `Underfull 0`, `Overfull 0`, `LaTeX Error 0`, undefined citation 0, undefined reference 0.
3. Abstract word count 250–300, one paragraph, all six questions answered in order.
4. Keywords 4–8.
5. One citation per claim, never three in one bracket.
6. No semicolons or em dashes in prose, no banned words from `docs/prompt.md`.
7. Every `\cite` key exists in `ref.bib` and every `ref.bib` entry is cited.
8. Effective figure resolution at least 600 DPI at the printed width.

### Known failure modes

- Adding or changing a reported number without adding it to `review_number_audit.py`.
- Reintroducing the limitations subsection that the supervisor removed.
- Writing "standard model" for the baseline, which conflicts with the §2 title.
- Pushing the abstract over 300 words while adding "just one more" result.
- Adding an unverified reference to answer a reviewer.
- Letting a table or figure float move the reference list onto an extra page.

---

## G1. Title

A research article title should be clear, concise, specific, informative, and easy to understand.

**Key principles:**

- **Clear** — communicates the research topic without ambiguity.
- **Concise** — avoids unnecessary words.
- **Specific** — identifies the main problem, method, subject, or contribution.
- **Informative** — gives readers a good idea of what the paper contains.
- **Easy to understand** — avoids unnecessary jargon, abbreviations, and complicated wording.

**Example:**

- ❌ An Investigation into the Application of Various Advanced Deep Learning-Based Computational Techniques for the Purpose of Automated Detection of Vegetable Quality
- ✔ Deep Learning for Automated Vegetable Quality Detection

---

## G2. Abstract

> **Agent note (not part of the supervisor's guideline):**
>
> - **Length:** The abstract must be 250–300 words at most. Aim for about 250 and never go above 300.
> - **Coverage:** Answer each of the six questions below in one or two sentences, in the order given. Keep only the headline numbers.
> - **Sub-questions:** The sub-questions under each point are prompts for what the sentence should cover. They are not extra sentences to add.

Answer the six questions below, in this order.

1. **Initialization — What is the research area?**
   - Identify the broad research domain, topic, and current trend.
   - Example: `Machine Learning → Deep Learning → Image Classification`
2. **Research Gap — What is missing?**
   - Identify limitations, unanswered questions, weaknesses, or unexplored areas in existing research.
   - What has not been adequately addressed by previous studies?
   - Clearly establish the need for your study.
3. **Research Objective — What do you want to do?**
   - Clearly define what the research intends to investigate, develop, improve, compare, or discover.
   - What specific problem are you trying to solve?
4. **Proposed Approach — How will you do it?**
   - Describe the methodology, models, algorithms, experimental design, dataset, and evaluation strategy.
   - What approach will you use to address the research problem?
5. **Justification — Why this Method and Dataset?**
   - Explain why the proposed method, model, dataset, or experimental design is appropriate and necessary.
   - Why should this approach be used instead of existing alternatives?
6. **Expected/Research Outcome — What will the research contribute?**
   - Identify the expected findings, performance improvement, dataset, model, framework, insights, or other research contributions.
   - What new knowledge, resource, method, or evidence will this research provide?

**A concise research journey:**

- `Research Area → Research Gap → Research Objective → Proposed Method → Justification → Experiment → Outcome → Contribution`
- **OR**
- `Research Context → Gap Identification → Objective Formulation → Methodology Design → Justification → Experimental Evaluation → Research Contribution`

**The six questions in short form:**

1. What is the context?
2. What is the gap?
3. What do you want to do?
4. How will you do it?
5. Why is your approach necessary?
6. What did you achieve?

**A simple formula:**
`Context + Gap + Objective + Method + Rationale + Outcome = Research Abstract`

---

## G3. Keywords

A simple approach is to identify 4–8 important concepts from your research.

1. **Identify the research topic**
   - What is the paper mainly about?
   - Example: _Deep Learning for Automated Vegetable Quality Detection_
   - Possible keyword: Deep learning, Vegetable quality
2. **Identify the research method**
   - What methods, algorithms, or technologies are used?
   - For example: Convolutional neural networks (CNN), Transfer learning, Object detection, Computer vision
3. **Identify the application/domain**
   - Where is the research applied?
   - For example: Agricultural image analysis, Food quality assessment, Smart agriculture
4. **Identify the specific research problem**
   - What problem does the research solve?
   - For example: Defect detection, Disease detection, Quality classification
5. **Use standard technical terminology**
   - Prefer terms that researchers actually use in the literature.
   - For example: ❌ Computer method for finding bad vegetables; ✔ Vegetable defect detection
6. **Avoid unnecessarily broad keywords**
   - ❌ Machine Learning ❌ Artificial Intelligence ❌ Computer Science
   - These are often too broad unless they are genuinely central to the paper.
   - Instead: ✔ Deep learning ✔ Object detection ✔ Vegetable defect detection ✔ Transfer learning

**A useful keyword formula:**
Keywords are selected from these categories:
`[Research Domain] + [Problem] + [Method] + [Application] + [Specific Technique]`

- **Paper title:** Real-Time Detection of Fresh and Defective Vegetables Using Deep Learning
- **Keywords:** Deep learning; vegetable defect detection; object detection; computer vision; transfer learning; agricultural image analysis.

Keywords should represent the core topic, research problem, methodology, application domain, and distinctive contribution of the study.

---

## G4. Introduction

The Introduction has four paragraphs.

### G4.1 Paragraph 1 — Current State

1. **What is currently happening/ongoing? — Current Status**
   - What is the current situation in this research area?
   - What are the existing methods, technologies, or approaches?
   - What are the recent developments or trends?
   - Why is this topic important?
2. **What is missing? — Research Gap / Challenges / Research Questions**
   - What limitations exist in current research?
   - What challenges remain unresolved?
   - What problems are not adequately addressed?
   - What research questions arise from these gaps?
3. **What do you want to do and how? — Research Objective & Approach**
   - What is the main objective of your research?
   - What do you propose or investigate?
   - How will you address the identified gap?
   - What method, model, framework, or approach will you use?
4. **What are the scopes of your Proposed Method and Dataset? — Scope & Applicability**
   - What can your proposed method do?
   - Where can it be applied?
   - What does the dataset contain?
   - Who or what can use the dataset?
   - What types of research, experiments, or applications can be supported by your method and dataset?
   - What are the boundaries or limitations of the proposed scope?

**A Simple Formula:**

- `Current State → Research Gap/Challenges → Objective & Approach → Scope & Applicability`
- Or, more simply: `What is happening? → What is missing? → What will you do and how? → What is the Scope? → Where can it be used?`

### G4.2 Paragraph 2 — Merged Literature Review

**Paragraph 2 — Existing Research + Research Gap**

- What have previous researchers done?
- What methods, models, datasets, or approaches have been used?
- What are the strengths and limitations of existing studies?
- What challenges or unresolved issues remain?
- Based on the literature, what is the specific research gap?

### G4.3 Paragraph 3 — Research Objective + Proposed Approach

**Paragraph 3 — Research Objective + Proposed Approach**

- What do you want to do?
- What is your research objective?
- How will you address the identified gap?
- What method/model/framework do you propose?
- What dataset will you use or develop?

### G4.4 Paragraph 4 — Scope and Contribution

- Scope of Proposed Method + Dataset + Contribution

### G4.5 Overall Flow

- **P1:** Current State/ongoing
- **P2:** Existing Research + Critical Literature Review + Research Gap
- **P3:** What were not done/solve/resolve/mitigate and why in literature? What You Want to Do + How You Will Do It?
- **P4:** Scope of Proposed Method + Dataset + Contribution

---

## G5. Types of article

1. Method based
2. Dataset based
3. Method + dataset based

The organization of each type is in the table under "Section order by article type".

---

## G6. Dataset / Corpus section

**For Dataset-based + Method-based Paper:**
For a dataset-based research paper, the Corpus/Dataset should normally be introduced immediately after the Introduction, because the dataset itself is the primary contribution of the paper.

- **2.1 Dataset Overview**
  - Purpose and scope
  - Dataset characteristics
  - Target application/domain
- **2.2 Data Collection**
  - Data sources
  - Collection procedure
  - Collection period
  - Inclusion/exclusion criteria
- **2.3 Data Preparation**
  - Cleaning
  - Preprocessing
  - Filtering
  - Formatting
- **2.4 Annotation/Labeling**
  - Annotation methodology
  - Label/class definitions
  - Annotation guidelines
  - Annotators
  - Quality-control procedure
  - Inter-annotator agreement, if applicable
- **2.5 Dataset Characteristics**
  - Number of instances
  - Classes/categories
  - Feature description
  - Class distribution
  - Statistical characteristics
- **2.6 Dataset Organization and Format**
  - File/folder structure
  - Data format
  - Metadata
  - Naming conventions
  - Train/validation/test organization, if provided
- **2.7 Dataset Validation and Quality Assessment**
  - Quality checking
  - Duplicate detection
  - Missing/inconsistent data
  - Label verification
  - Data quality statistics
- **2.8 Ethical, Privacy, and Licensing Considerations**
  - Consent, where applicable
  - Privacy/anonymization
  - Ethical approval, if applicable
  - Data license and usage restrictions

---

## G7. Conventional method/methods

_(Method-based: section 2. Dataset-based: section 3.)_

The Conventional Method(s) section presents the baseline or existing method(s) from which the proposed method is derived or developed. The purpose is to establish a clear technical foundation for understanding the proposed approach.

- **Baseline Method**
  - Introduce the conventional or established method.
  - Explain why it is selected as the baseline.
  - Describe its main components and assumptions.
- **Working Principle**
  - Explain how the conventional method operates.
  - Present the processing steps or workflow.
  - Demonstrate the relationship between its input, intermediate processes, and output.
- **Technical Formulation**
  - Describe the method mathematically where appropriate.
  - Include relevant equations, mathematical formulations, parameters, and objective functions.
  - Define important variables and terms clearly.
- **Algorithm / Procedure**
  - Present the method as an algorithm or step-by-step procedure when appropriate.
  - Explain the key computational operations and decision processes.
- **Main Contribution / Characteristics of the Baseline**
  - Highlight the key technical contribution or distinctive characteristics of the conventional method.
  - Explain what makes the baseline method useful or widely applicable.
- **Limitations and Research Gap**
  - Identify the specific limitations, weaknesses, or unresolved issues of the conventional method.
  - Explain how these limitations motivate the development of the proposed method.
  - End the section with the specific gap that the proposed method intends to address.

**Recommended Flow:**
`Conventional Method → Working Principle → Mathematical Formulation → Algorithm → Key Characteristics of Baseline → Limitations → Research Gap → Proposed Method`

**Important Distinction:**
The Conventional Method section should not simply repeat the Literature Review.

- The Literature Review answers: _"What have previous researchers done?"_
- Whereas the Conventional Method section answers: _"How does the baseline method technically work, and what limitation motivates the proposed method?"_

This makes the transition to the Proposed Method much clearer:
`Baseline Method → Identified Limitation → Proposed Modification/Extension → Proposed Method`

---

## G8. Proposed method

The Proposed Method section can follow a parallel structure with the Conventional Method, but it should focus on the novelty, modification, and technical contribution of your work.

The Proposed Method section describes the newly developed approach introduced to address the research gap identified in the Conventional Method. It should clearly explain how the proposed method works, how it is technically formulated, how it is implemented, and what its main contribution is.

- **3.1 Working Principle**
  - Explain the overall working mechanism of the proposed method.
  - Describe the input, processing steps, and output.
  - Explain how the proposed method differs from the conventional/baseline method.
  - Use a flowchart, block diagram, or conceptual framework where appropriate.
- **3.2 Technical Formulation**
  - Present the mathematical formulation of the proposed method.
  - Define equations, parameters, variables, constraints, objective functions, or other technical components.
  - Clearly show any modification, extension, or newly introduced component compared with the conventional method.
- **3.3 Algorithm / Procedure**
  - Present the proposed method as a step-by-step algorithm or procedure.
  - Clearly describe the computational workflow.
  - Identify the major inputs, operations, decision processes, and outputs.
  - Provide sufficient detail so that the method can be understood and reproduced.
- **3.4 Main Contribution / Characteristics of the Proposed Method**
  - Clearly identify what is new or different in the proposed method.
  - Explain how it addresses the limitation or gap of the conventional method.
  - Highlight the key technical characteristics and expected advantages.
  - Distinguish the novel contribution from components adopted from existing methods.

**Overall Methodology Flow:**

- **Conventional Method:** `Working Principle → Technical Formulation → Algorithm → Limitations → Research Gap`
- **then Proposed Method:** `Working Principle → Technical Formulation → Algorithm/Procedure → Main Contribution/Characteristics`

This creates a very clear research narrative:
`Existing Method → Limitation → Research Gap → Proposed Solution → Technical Novelty`

**A useful principle:** In the Proposed Method section, the reader should be able to answer three questions clearly:

1. What exactly is proposed?
2. How does it work technically?
3. What is new compared with the conventional method?

---

## G9. Experiments

**For Method-based Paper:**

- **4.1 Experimental Setup and Conditions**
  - Hardware configuration
  - Software/platform
  - Programming environment
  - Experimental parameters
  - Simulation or implementation conditions
  - Reproducibility settings
- **4.2 Dataset Introduction** _(This section is only for method based. For dataset-base it is avoided)_
  - Dataset source
  - Dataset size
  - Number of classes/features
  - Data characteristics
  - Data distribution
  - Training/validation/testing split
  - Preprocessing, if applicable
- **4.3 Experimental Procedure**
  - Overall experimental workflow
  - Data preparation
  - Application of the conventional/baseline method
  - Application of the proposed method
  - Training/testing procedure
  - Repeated experiments or cross-validation
  - Experimental scenarios
- **4.4 Parameter Configuration**
  - Model/method parameters
  - Hyperparameters
  - Fixed parameters
  - Optimization settings
  - Parameter selection/tuning strategy
- **4.5 Evaluation Metrics**
  - Accuracy, Precision, Recall, F1-score, etc., where applicable
  - Error/performance measures
  - Mathematical formulation of metrics
  - Statistical measures, if applicable

**Recommended structure:**
`4.1 Experimental Setup and Conditions → 4.2 Dataset Introduction → 4.3 Experimental Procedure → 4.4 Parameter Configuration → 4.5 Evaluation Metrics`

---

## G10. Results

Results should be presented and analyzed based on the defined evaluation metrics, because the metrics provide the basis for interpreting and comparing the experimental outcomes.

- **5.1 Results Based on Evaluation Metrics**
  - Accuracy-based Results
  - Precision-based Results
  - Recall-based Results
  - F1-score-based Results
  - And so on
- **5.2 Comparative Results**
  - Conventional/baseline vs. proposed method
  - Metric-wise comparison
  - Improvement/reduction in performance measures
- **5.3 Ablation Study (if applicable)**
  - An ablation study in machine learning is an experiment where you remove, disable, or replace specific components (like layers, features, or modules) of a model to measure how much they contribute to its overall performance.
- **5.4 Statistical Analysis**
  - Confidence Interval
  - Effect size
  - t/p value
- **5.5 Discussion of Experimental/Key Findings**
  - Interpretation of the metric-wise results
  - Why the proposed method performs differently
  - Strengths and limitations observed from the experiments

**Recommended structure:**
`Metric-wise Analysis → Comparison → Ablation → Statistical Validation → Findings`

---

## G11. Conclusion

A Conclusion section should briefly summarize the research, the main contribution, the key findings, and possible future work.

**Conclusion contains:**

- **Research objective/problem** – What was addressed?
- **Proposed work/contribution** – What was developed or introduced?
- **Key findings** – What were the major experimental outcomes?
- **Research significance** – What do the findings demonstrate?
- **Future work** – What can be improved or explored next?

**Recommended structure:**
`Conclusion = Objective → Contribution → Key Findings → Significance → Future Work`

**Important:** Avoid introducing new results, new methods, or new references in the Conclusion.

---

## G12. Author Contributions

"Who did what in this research?"

Authors may contribute to:

- **Conceptualization** – Research idea, objectives, and overall research goals
- **Methodology** – Development/design of the methodology or models
- **Software** – Programming, implementation, and code development
- **Validation** – Verification and reproducibility of results
- **Formal Analysis** – Statistical, mathematical, or computational analysis
- **Investigation** – Conducting experiments and data/evidence collection
- **Resources** – Providing materials, datasets, computing resources, etc.
- **Data Curation** – Managing, cleaning, annotating, and maintaining research data
- **Writing – Original Draft** – Preparing the initial manuscript
- **Writing – Review & Editing** – Reviewing, revising, and editing the manuscript
- **Visualization** – Preparing figures, graphs, and other visualizations
- **Supervision** – Research supervision and mentorship
- **Project Administration** – Coordination and management of the research project
- **Funding Acquisition** – Obtaining financial support for the research

---

## G13. Declaration

Declaration part includes:

- Funding
- Conflict of interest
- Data availability
- Code availability
- Ethical approval statement

---

## G14. Acknowledgements

The Acknowledgements section recognizes people or organizations that supported the research but do not qualify for authorship.

Acknowledgements may include:

- **Technical assistance** — laboratory, programming, equipment, or technical support
- **Research assistance** — help with data collection, annotation, experiments, or other research activities
- **Administrative support** — institutional or departmental assistance
- **Language/editing support** — proofreading or language improvement, where appropriate
- **Data/resource providers** — individuals or organizations that provided datasets, materials, or other resources
- **Institutional support** — departments, laboratories, universities, or research centers
- **Funding support** — if the journal does not require funding information in a separate section
- **Special assistance** — individuals who provided valuable advice or feedback but did not meet authorship criteria

---

## G15. References

- Use recent and relevant literature, including contemporary journal articles, conference papers, symposium proceedings, book chapters, and books, where appropriate.
- Ensure that the references are relevant to the research topic and sufficiently up to date.
- Present all references in a consistent and uniform citation style, such as IEEE, throughout the manuscript.
- Avoid unnecessary URLs/link-based references. When a URL is required, provide the access date according to the selected reference style or journal guidelines.
- Include bibliographic information such as author(s), article/chapter title, journal/conference/book title, volume, issue, page range, publication date/year, publisher, DOI, etc., as applicable.
- Follow the reference-formatting requirements of the target journal/conference. The journal's guidelines take precedence over a generic reference format.
- Ensure that every in-text citation appears in the reference list and every listed reference is cited in the manuscript.
- Verify the accuracy of author names, article titles, publication details, DOI, volume, issue, and page numbers before submission.

**One important correction:**
"Clearly mention Vol. number, Issue, PP., date, publisher" is not universally applicable. For example, an IEEE journal reference may contain volume, issue, pages, and year, while a conference paper, book, book chapter, or article with an article number may have a different structure.

---

## How the HaluRISC paper maps to this guideline

Build facts from `Journal_Paper/out/halurisc.pdf` (2026-10-23): 17 pages, 15
tables, 6 figures, 2 algorithms, 34 references, abstract 276 words, 7 keywords,
zero LaTeX warnings.

| Guideline | Location in `halurisc.tex` | Content |
| :--- | :--- | :--- |
| G1 | `\title` | Calibrated Hallucination Risk Estimation in Large Language Models Using Evidence-Consistent XGBoost |
| G2 | `abstract` | Six questions in order, headline numbers only, one paragraph |
| G3 | `keywords` | Distribution shift, evidence-consistent XGBoost, explainable AI, hallucination risk estimation, probability calibration, question answering, SHAP |
| G4 | `sec:intro` | P1 current state, P2 literature and gap, P3 objective and approach, P4 scope and contribution |
| G7 | `sec:conventional` | 2.1 Preliminaries, 2.2 Baseline Method (why selected, formulation, Algorithm 1), 2.3 Limitations and Research Gap |
| G8 | `sec:proposed` | 3.1 Working Principle and Task, 3.2 Features, 3.3 Evidence-Consistent XGBoost (Algorithm 2), 3.4 Calibration and Explainability, 3.5 System and Reproducibility |
| G9 | `sec:experiment` | 4.1 Experimental Setup and Conditions, 4.2 Dataset, 4.3 Experimental Procedure, 4.4 Parameter Configuration, 4.5 Evaluation Metrics, 4.6 Additional Protocols |
| G10 | `sec:results` | 5.1 Model comparison, 5.2 Evidence-consistent XGBoost, 5.3 Calibration, 5.4 Cross-domain transfer, 5.5 Ablation, 5.6 Explanation reliability, 5.7 Efficiency and the LLM judge, 5.8 Error analysis, 5.9 Discussion |
| G11 | `sec:conclusion` | Objective, contribution, key findings, negative results, significance, future work |
| G12 | author block | One `\credit{}` per author, printed by `\printcredits` before the Declarations |
| G13 | `Declarations` | Funding, conflicts of interest, data availability, code availability, ethical approval |
| G14 | `Acknowledgements` | Supervisor and dataset maintainers |
| G15 | `ref.bib` | 34 verified entries, every one cited, uniform Elsevier numbered style |

The Declarations and Acknowledgements are unnumbered sections, as the CAS class
expects. Author contributions appear before the Declarations, matching the
section order table at the top of this file.
