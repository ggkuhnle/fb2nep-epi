# FB2NEP: Nutritional Epidemiology & Public Health  

This repository hosts the **Nutritional Epidemiology** teaching materials for FB2NEP.  
It includes:

- 📓 **Notebooks** — Colab-ready, teaching key epidemiology concepts  
- 📑 **Slides (PDF)** — lecture materials  
- 📊 **Synthetic dataset** — generated via `scripts/generate_dataset.py`  
- 📝 **Assessment 1 brief & template**

A rendered Quarto site with Colab launchers is available here:  
👉 [FB2NEP website](https://ggkuhnle.github.io/fb2nep-epi/)

---

## Notebooks

### Part 1 — Nutritional Epidemiology

| Notebook | Title |
|---|---|
| 1.03 | Introduction to Jupyter and Google Colab |
| 1.04 | Data collection and cleaning |
| 1.05 | Representativeness and sampling |
| 1.06 | Data exploration and "Table 1" |
| 1.07 | Data transformation |
| 1.08 | Regression and modelling (Part 1) |
| 1.09 | Regression and modelling (Part 2) — confounding, DAGs, mediation |
| **1.09a** | **Confounding: causal diagrams and worked examples** *(companion to 1.09)* |
| 1.10 | Missing data and sensitivity analysis |

### Part 2 — Public Health Nutrition

| Notebook | Title |
|---|---|
| 2.02 | DALYs and QALYs |
| 2.03 | Health inequalities |
| 2.04 | Case study: Salt reduction |
| 2.05 | Case study: Sugar reduction |
| 2.06 | Policy simulation and resource allocation |

---

## Structure

- `notebooks/` — interactive teaching notebooks (numbered & titled for clarity)
- `slides/` — lecture slides in PDF (same naming as notebooks)
- `scripts/` — dataset generator & validator
- `metadata/` — data dictionary & provenance
- `assessment/` — Assessment 1 brief & template

---

## Quick start

- Open notebooks in Google Colab via the website, or clone locally:

```bash
git clone https://github.com/ggkuhnle/fb2nep-epi.git
cd fb2nep-epi
pip install -r requirements.txt
jupyter notebook notebooks/

