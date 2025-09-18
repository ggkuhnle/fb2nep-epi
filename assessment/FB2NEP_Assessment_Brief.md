# FB2NEP Assessment (Epidemiology & Data Analysis) — 50% of 20 credits

**Module:** FB2NEP: Nutritional Epidemiology & Public Health  
**Weighting:** 50% of module (Epi half)  
**Length/Format:** 2000–2500 words (+ figures/tables, code appendix excluded from count)  
**Submission:** Single PDF report + the provided notebook `fb2nep_assignment_template.ipynb` with executed outputs + your data mapping cell completed.  
**Assessment Notebook:** [Open in Google Colab](https://colab.research.google.com/github/ggkuhnle/fb2nep-epi/blob/main/assessment/fb2nep_assignment_template.ipynb)  
**Practice Notebook:** [Open in Google Colab](https://colab.research.google.com/github/ggkuhnle/fb2nep-epi/blob/main/assessment/fb2nep_practice.ipynb)  

**Dataset:** `fb2nep.csv` (as provided in the module repo).

---

## Provided Files
- **`FB2NEP_Assessment_Brief.md`**: This document, outlining the assessment requirements, tasks, and marking rubric. Review it carefully to understand expectations and deliverables.
- **`fb2nep_assignment_template.ipynb`**: The main notebook for the assessment. Complete the "Data mapping" cell and execute all cells to generate results for your report. Submit the executed notebook with outputs.  
  - **Link**: [Open Assessment Notebook in Google Colab](https://colab.research.google.com/github/ggkuhnle/fb2nep-epi/blob/main/assessment/fb2nep_assignment_template.ipynb)
- **`fb2nep_practice.ipynb`**: A practice notebook to explore the analysis tasks (e.g., Table 1, missingness, regressions) without needing to set up data or libraries. Use this to practice before working on the assessment.  
  - **Link**: [Open Practice Notebook in Google Colab](https://colab.research.google.com/github/ggkuhnle/fb2nep-epi/blob/main/assessment/fb2nep_practice.ipynb)
- **`FB2NEP_Cheat_Sheet.md`**: A guide explaining key Python functions (e.g., pandas, statsmodels, lifelines) used in the notebooks, with examples relevant to the assessment tasks. Refer to this for understanding code and interpreting outputs.

---

## Learning Outcomes Assessed
1. Explain core principles of nutritional epidemiology (exposure/outcome definitions, bias, confounding, measurement error).  
2. Conduct basic epidemiological data analysis (Table 1, missingness audit, regression modelling).  
3. Critically appraise the impact of measurement error and confounding on inference (biomarker vs diet diary (DD) comparison; model diagnostics; change-in-estimate).  
4. Communicate findings clearly, with correct interpretation and limitations.

---

## Academic Integrity & “LLM‑Proofing”
- This assessment requires **dataset‑specific results** that cannot be guessed. Your grade depends on the **numerical results you obtain** from `fb2nep.csv` and on your **reasoned interpretation**.
- Provide a **short Methods** section that explains exactly **what you did and why** (model choices, confounder justification, inclusion criteria). Copy‑pasting generic text will be penalised.
- Include a **code appendix** (the executed `fb2nep_assignment_template.ipynb` notebook). Keep **intermediate outputs**. We may **reproduce your results**.  
- You must complete the **“Data mapping” cell** in the notebook where you assign the correct columns for outcome, exposures, and candidate confounders.
- Where appropriate, justify your choices using **study design logic, DAG reasoning, or change‑in‑estimate rules** (≥10% rule). Merely citing “the model told me” is not acceptable.
- We may request a brief **viva** to discuss your workflow and choices.

---

## Task
Write a concise report (2000–2500 words) supported by your executed notebook that addresses the following numbered questions. Label sections accordingly in your report.

### 1) Data Audit and Table 1 (10 marks)
a) Describe the cohort derivation (inclusion/exclusion). Report **N** at each step.  
b) Produce **Table 1** (overall and, if sensible, stratified by your chosen outcome or exposure). Include **counts, %, means/SD, or medians/IQR** as appropriate.  
c) Briefly interpret notable imbalances or patterns relevant to potential confounding.

### 2) Missingness (10 marks)
a) Provide a **missingness summary** (per variable; total rows with any missing).  
b) Show a **visualisation** of missingness patterns (matrix/heatmap).  
c) Explain whether missingness is likely **MCAR/MAR/MNAR** and the implications for your analysis. State and justify your chosen handling approach (e.g., complete‑case, simple imputation for covariates, or sensitivity analysis).

### 3) Biomarker vs Diet Diary (DD) Comparison (15 marks)
a) Compare the **biomarker** exposure and the **DD‑based** exposure: produce a **scatter plot**, **correlation**, and a **Bland–Altman** plot.  
b) Quantify and interpret **agreement vs association**.  
c) Discuss likely **measurement error** structure (classical vs Berkson; differential vs non‑differential) and expected bias directions in regression.

### 4) Primary Association & Confounding (15 marks)
a) Specify a **primary outcome** (binary or continuous) and a **primary exposure** (use biomarker as the main exposure; DD as a secondary).  
b) Fit a **minimally adjusted model** and a **confounder‑adjusted model** using a **pre‑specified confounder set** (justify via DAG or domain logic).  
c) Apply a **change‑in‑estimate (≥10%)** procedure to identify additional confounders from a candidate pool. Report the % change and justify final model.  
d) Present results as **effect sizes with 95% CIs** and provide a **plain‑language interpretation** (“On average…”, “Odds are…”) that is aligned with model scale.

### 5) Sensitivity Analyses (10 marks)
Perform at least **two** of:  
- Replace the exposure (DD ↔ biomarker) and compare estimates.  
- Use **robust SEs**; check **influential points**; or re‑fit with/without outliers.  
- Add plausible **interaction** (e.g., sex×exposure) and interpret.  
- Re‑express exposure (e.g., per SD, log‑transform, or quintiles) and compare.

### 6) Reflection & Limitations (10 marks)
Discuss **strengths, limitations, and generalisability**. Address **measurement error, residual confounding, selection bias**, and how they might shift the effect size (direction and magnitude).

---

## Report Structure (Recommended)
- Title, Abstract (≤150 words)  
- Methods (data mapping, inclusion/exclusion, variables, models, confounder rationale, handling of missingness)  
- Results (sections 1–5, with numbered figures/tables that correspond to the questions)  
- Discussion (section 6)  
- References (if used)  
- Appendix: Executed `fb2nep_assignment_template.ipynb`

---

## Marking Rubric (100 marks)
- **Technical Correctness & Reproducibility (30):** Code runs; outputs reproducible; sensible diagnostics; correct effect metrics/CI.  
- **Epidemiological Reasoning (30):** Appropriate confounder handling; clear justification (DAG/change‑in‑estimate); valid interpretation of bias sources.  
- **Insight & Critical Thinking (25):** Data‑driven insights; limitations acknowledged; sensitivity analyses inform conclusions.  
- **Presentation & Clarity (15):** Coherent narrative; well‑labelled figures/tables; concise writing within word limit.

**Band Descriptors (Indicative):**  
- **High First (≥80):** Methodologically rigorous; exemplary reasoning; insightful sensitivity work that changes or sharpens conclusions.  
- **First (70–79):** Correct, well‑justified models; clear interpretation; minor issues only.  
- **Upper Second (60–69):** Mostly correct; some gaps in justification or diagnostics; interpretation broadly sound.  
- **Lower Second (50–59):** Basic pipeline present; limited justification; superficial interpretation.  
- **Third (<50):** Major errors; irreproducible; weak or incorrect interpretation.

---

## Deliverables
1. **PDF report** (2000–2500 words) named `FB2NEP_<studentID>.pdf`.  
2. **Executed notebook** `fb2nep_assignment_<studentID>.ipynb` with all cells run.  
3. Optional: a short **DAG image** (if used), included in the report.

---

## Getting Started
- Place `fb2nep.csv` in `./data/` (or adjust the path in the notebook).  
- Review the **assessment brief** (`FB2NEP_Assessment_Brief.md`) to understand the tasks and expectations.  
- Use the **practice notebook** (`fb2nep_practice.ipynb`) to experiment with the analyses. It auto-loads data and libraries, with cells for each task (Table 1, missingness, regressions). Access it via [Google Colab](https://colab.research.google.com/github/ggkuhnle/fb2nep-epi/blob/main/assessment/fb2nep_practice.ipynb).  
- Refer to the **cheat sheet** (`FB2NEP_Cheat_Sheet.md`) for explanations of Python functions used in the notebooks.  
- Open and complete the **Data mapping** cell in `fb2nep_assignment_template.ipynb`. This is essential for reproducibility and marking. Access it via [Google Colab](https://colab.research.google.com/github/ggkuhnle/fb2nep-epi/blob/main/assessment/fb2nep_assignment_template.ipynb).  
- Run all cells in the assignment notebook to generate outputs for your report.

**Note**: The Google Colab links assume the notebooks are in the `assessment` folder of the `ggkuhnle/fb2nep-epi` repository. If the repository structure or paths differ, please inform the module coordinator to ensure the links work correctly.