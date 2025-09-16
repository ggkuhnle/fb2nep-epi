#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FB2NEP teaching notebooks (9-part series) into notebooks/
- Colab/local bootstrap via scripts/bootstrap.py
- British English, numbered filenames, clear LO & TODOs
"""

from __future__ import annotations
import os, json, textwrap, datetime as dt

NOTEBOOKS = [
    ("01_intro_epidemiology.ipynb", "01 · Introduction to Nutritional Epidemiology", [
        "- Define epidemiology and its scope within nutrition.",
        "- Contrast landmark findings (e.g., smoking & lung cancer) with nutrition effects.",
        "- Recognise nutrition-specific challenges: complexity, misreporting, long latency."
    ], [
        ("What is epidemiology?", """
Epidemiology studies the distribution and determinants of health-related states in populations.
In nutrition, exposures are complex (dietary patterns, foods, nutrients), often mismeasured, and
effects can be modest and slow to emerge.
"""),
        ("Mini exercise — relative magnitudes", """
# TODO: Think: why do nutrition associations often look smaller than smoking?
# (No code required here; jot 3 reasons in a markdown cell below.)
"""),
        ("First look at the dataset", """
import pandas as pd
df.head(3), df.shape
"""),
        ("Sanity checks", """
# Simple content and ranges
assert {'age','sex','BMI'}.issubset(df.columns)
assert df['age'].min() >= 40
"""),
    ]),

    ("02_study_designs.ipynb", "02 · Study Designs in Nutrition Research", [
        "- Differentiate cross-sectional, cohort, case-control, and nested case-control designs.",
        "- Position RCTs vs observational evidence in nutrition.",
        "- Map strengths/limitations to real nutrition examples."
    ], [
        ("Designs overview", """
# Cross-sectional: snapshot; prevalence; temporality weak.
# Cohort: exposure precedes outcome; incidence; stronger causal claims.
# Case-control: sample on outcome; efficient for rare outcomes.
# Nested case-control: subsample within a cohort; good for costly biomarkers.
"""),
        ("Classify examples", """
# TODO: Classify each as X-sectional / Cohort / Case-control / RCT
examples = [
    "Diet diary and BP measured once in 2024 (1000 adults).",
    "Fruit & veg intake at baseline; follow-up stroke events (10 years).",
    "200 incident MI cases vs 400 controls; FFQ asking past year's diet.",
    "Salt reduction randomised for 8 weeks; change in SBP."
]
for e in examples:
    print("-", e)
# Write your answers in a markdown cell below.
"""),
        ("Where RCTs fit", """
# RCTs in nutrition: shorter timescales, intermediate outcomes (SBP, lipids),
# adherence and blinding are tricky; whole-diet interventions hard to sustain.
"""),
    ]),

    ("03_exposure_outcome_assessment.ipynb", "03 · Exposure & Outcome Assessment", [
        "- Compare self-report methods (24HR, FFQ, diaries) and their error structures.",
        "- Understand biomarker types: recovery, concentration, replacement.",
        "- Clarify endpoints: clinical vs intermediate; validity & reproducibility."
    ], [
        ("Load & basic variables", """
df[['fruit_veg_g_d','red_meat_g_d','salt_g_d','plasma_vitC_umol_L','urinary_sodium_mmol_L']].describe()
"""),
        ("Self-report vs biomarkers", """
# TODO: Compute correlation between fruit_veg_g_d and plasma_vitC_umol_L.
sub = df[['fruit_veg_g_d','plasma_vitC_umol_L']].dropna()
r = sub.corr().iloc[0,1]
print("r =", round(r,3))
assert r > 0.4, "Expect a moderate positive correlation."
"""),
        ("Event indicators and dates", """
# Incident flags and dates: ensure consistency
for flag, date in [('CVD_incident','CVD_date'),('Cancer_incident','Cancer_date')]:
    f = df[flag].astype(int)
    d = df[date].fillna("")
    assert ((f==1) <= (d!="")).all(), f"{date}: incident=1 must have a date"
"""),
    ]),

    ("04_data_foundations.ipynb", "04 · Data Foundations", [
        "- Understand collection pipelines (surveys, labs, registries).",
        "- Build a light cleaning pipeline; detect outliers.",
        "- Distinguish variable types and implications for analysis.",
        "- Identify missingness patterns: MCAR, MAR, MNAR."
    ], [
        ("Types & basic cleaning", """
# Cast some categoricals (for clarity in summaries)
for c in ['sex','smoking_status','physical_activity','SES_class','menopausal_status']:
    if c in df.columns:
        df[c] = df[c].astype('category')
df.dtypes.head(12)
"""),
        ("Outliers & bounds", """
# Simple winsor for plotting (teaching only)
import numpy as np
def winsor(s, p=0.005):
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lo, hi)
df['_salt_ws'] = winsor(df['salt_g_d'].dropna()) if 'salt_g_d' in df else None
"""),
        ("Missingness overview", """
miss = df.isna().mean().sort_values(ascending=False).round(3)
print(miss.head(12))
assert miss.max() < 0.25, "No variable should be >25% missing in this synthetic."
"""),
    ]),

    ("05_exploratory_data_analysis.ipynb", "05 · Exploratory Data Analysis", [
        "- Inspect distributions and compare groups.",
        "- Build a defensible Table 1.",
        "- Use visualisation to surface patterns."
    ], [
        ("Histogram & log-scale demo", """
import numpy as np, matplotlib.pyplot as plt
for col in ['ssb_ml_d','red_meat_g_d','energy_kcal']:
    if col in df:
        x = df[col].dropna().values
        plt.figure(); plt.hist(x, bins=40, alpha=0.85); plt.title(col); plt.xlabel(col); plt.ylabel('count'); plt.show()
        plt.figure(); plt.hist(np.log1p(x), bins=40, alpha=0.85); plt.title('log1p('+col+')'); plt.xlabel('log1p'); plt.ylabel('count'); plt.show()
"""),
        ("Table 1 (quick)", """
import pandas as pd
imp = df.copy()
for c in imp.select_dtypes(include=['float64','int64']).columns:
    imp[c] = imp[c].fillna(imp[c].median())
for c in imp.select_dtypes(include=['object','category']).columns:
    m = imp[c].mode(dropna=True)
    if len(m): imp[c] = imp[c].fillna(m.iloc[0])
by = 'Cancer_incident'
cont = [c for c in ['age','BMI','SBP','energy_kcal','fruit_veg_g_d','red_meat_g_d','salt_g_d'] if c in imp]
out = {c: imp.groupby(by)[c].agg(['mean','std','median']).round(2) for c in cont}
tbl = pd.concat(out, axis=1); tbl.head(10)
"""),
        ("# TODO — interpret a figure", """
# Pick one plot above and write 3 sentences interpreting what you see.
"""),
    ]),

    ("06_data_transformation.ipynb", "06 · Data Transformation", [
        "- Decide when to transform (skewness, comparability).",
        "- Apply log, z-score, Box–Cox (explain costs/benefits).",
        "- Demonstrate information loss from categorisation."
    ], [
        ("Simple transforms", """
import numpy as np, pandas as pd
def z(x): return (x - x.mean())/x.std()
for col in ['red_meat_g_d','ssb_ml_d','energy_kcal']:
    if col in df:
        df[col+'_log1p'] = np.log1p(df[col])
        df[col+'_z'] = z(df[col])
df.filter(regex='(red_meat|ssb|energy).*(_log1p|_z)$').head()
"""),
        ("Categorisation pitfall (demo)", """
# Compare continuous vs tertiles for salt predicting CVD (practice)
import statsmodels.api as sm, pandas as pd
d = df[['CVD_incident','salt_g_d','age','BMI','sex','IMD_quintile','smoking_status']].dropna().copy()
d = d.rename(columns={'CVD_incident':'y','salt_g_d':'salt'})
Xc = pd.get_dummies(d[['salt','age','BMI','sex','IMD_quintile','smoking_status']], drop_first=True)
Xc = sm.add_constant(Xc, has_constant='add'); fit_c = sm.Logit(d['y'], Xc).fit(disp=False)
d['salt_tert'] = pd.qcut(d['salt'], 3, labels=['T1','T2','T3'])
Xt = pd.get_dummies(d[['salt_tert','age','BMI','sex','IMD_quintile','smoking_status']], drop_first=True)
Xt = sm.add_constant(Xt, has_constant='add'); fit_t = sm.Logit(d['y'], Xt).fit(disp=False)
print("SE (continuous salt):", float(fit_c.bse['salt']))
print("SE (tertile T3 vs T1):", float(fit_t.bse.get('salt_tert_T3', float('nan'))))
"""),
        ("# TODO — reflect", """
# Which specification preserves more information? Why?
"""),
    ]),

    ("07_regression_modelling.ipynb", "07 · Regression & Modelling", [
        "- Fit and interpret linear/logistic/Cox (focus on logistic for this dataset).",
        "- Check assumptions and diagnostics at a teaching level.",
        "- Report β, OR with 95% CI; communicate clearly."
    ], [
        ("Unadjusted & adjusted logistic", """
import statsmodels.api as sm
from patsy import dmatrices
OUTCOME = 'Cancer_incident'; EXPOSURE = 'red_meat_g_d'
adj = ['age','sex','IMD_quintile','SES_class','smoking_status','BMI']
m = df[[OUTCOME,EXPOSURE]+adj].dropna()
y_u, X_u = dmatrices(f'{OUTCOME} ~ {EXPOSURE}', data=m, return_type='dataframe')
fit_u = sm.Logit(y_u, X_u).fit(disp=False)
def cat(v): 
    import pandas as pd
    return f'C({v})' if (m[v].dtype=='object' or str(m[v].dtype).startswith('category')) else v
rhs = " + ".join([cat(EXPOSURE)] + [cat(v) for v in adj])
y_a, X_a = dmatrices(f'{OUTCOME} ~ ' + rhs, data=m, return_type='dataframe')
fit_a = sm.Logit(y_a, X_a).fit(disp=False)
import numpy as np, pandas as pd
def tidy(f): 
    OR = np.exp(f.params).rename('OR')
    CI = np.exp(f.conf_int()).rename(columns={0:'2.5%',1:'97.5%'})
    return pd.concat([OR,CI], axis=1).round(3)
tidy(fit_u), tidy(fit_a)
"""),
        ("Diagnostics (quick)", """
# Pseudo R^2 and influence (outline only for teaching)
print("McFadden pseudo R^2 (adj):", getattr(fit_a, 'prsquared', None))
"""),
        ("# TODO — interpretation", """
# In 3–5 sentences: interpret the adjusted OR for red_meat_g_d.
"""),
    ]),

    ("08_confounding_colliders_mediators.ipynb", "08 · Confounding, Colliders, Mediators", [
        "- Distinguish confounders, colliders, mediators with DAGs.",
        "- Identify proper adjustment sets.",
        "- Avoid common mistakes (adjusting for colliders/mediators)."
    ], [
        ("DAG sketch (optional)", """
try:
    import networkx as nx, matplotlib.pyplot as plt
    G = nx.DiGraph()
    G.add_edge('SES','red_meat_g_d'); G.add_edge('SES','Cancer_incident')
    G.add_edge('Age','red_meat_g_d'); G.add_edge('Age','Cancer_incident')
    G.add_edge('Smoking','Cancer_incident')
    G.add_edge('red_meat_g_d','Cancer_incident')
    pos = nx.spring_layout(G, seed=11088); plt.figure(figsize=(6,4))
    nx.draw_networkx(G, pos=pos, node_size=1200, font_size=9); plt.axis('off'); plt.show()
except Exception as e:
    print("DAG skipped:", e)
"""),
        ("Backdoor logic", """
# TODO: In markdown, state a minimal sufficient adjustment set for red_meat_g_d → Cancer_incident.
# Justify briefly using the DAG above.
"""),
        ("Collider caution", """
# Example: do NOT adjust for a downstream mediator or a collider.
"""),
    ]),

    ("09_missing_data.ipynb", "09 · Dealing with Missing Data", [
        "- Recognise MCAR/MAR/MNAR patterns in practice.",
        "- Implement complete-case and simple imputation for teaching.",
        "- Run a short sensitivity analysis."
    ], [
        ("Missingness map", """
miss = df.isna().mean().sort_values(ascending=False).round(3)
miss.head(12)
"""),
        ("Complete-case vs mean-impute (teaching)", """
import statsmodels.api as sm
from patsy import dmatrices
OUTCOME, EXPOSURE = 'Cancer_incident', 'red_meat_g_d'
adj = ['age','sex','IMD_quintile','SES_class','smoking_status','BMI']
m_cc = df[[OUTCOME,EXPOSURE]+adj].dropna().copy()
# crude mean-impute for demo only
m_imp = df[[OUTCOME,EXPOSURE]+adj].copy()
for c in m_imp.select_dtypes(include=['float64','int64']).columns:
    m_imp[c] = m_imp[c].fillna(m_imp[c].median())
for c in m_imp.select_dtypes(include=['object','category']).columns:
    md = m_imp[c].mode(dropna=True)
    if len(md): m_imp[c] = m_imp[c].fillna(md.iloc[0])
def fit(df_):
    import pandas as pd
    def cat(v): 
        return f"C({v})" if (df_[v].dtype=='object' or str(df_[v].dtype).startswith('category')) else v
    rhs = " + ".join([EXPOSURE] + [cat(v) for v in adj])
    y, X = dmatrices(f"{OUTCOME} ~ " + rhs, data=df_.dropna(), return_type='dataframe')
    return sm.Logit(y, X).fit(disp=False)
fit_cc, fit_imp = fit(m_cc), fit(m_imp)
import numpy as np, pandas as pd
def tidy(f): 
    OR = np.exp(f.params).rename('OR')
    CI = np.exp(f.conf_int()).rename(columns={0:'2.5%',1:'97.5%'})
    return pd.concat([OR,CI], axis=1).round(3)
tidy(fit_cc), tidy(fit_imp)
"""),
        ("# TODO — sensitivity reflection", """
# Do the ORs meaningfully differ between complete-case and imputed?
# Briefly discuss potential biases introduced by each approach.
"""),
    ]),
]

BOOTSTRAP_CELL = """\
# Bootstrap (works in Colab or locally) — loads df via scripts/bootstrap.py
import runpy, pathlib
for p in ["scripts/bootstrap.py","../scripts/bootstrap.py","../../scripts/bootstrap.py"]:
    if pathlib.Path(p).exists():
        print(f"Bootstrapping via: {p}")
        runpy.run_path(p)
        break
else:
    raise FileNotFoundError("scripts/bootstrap.py not found")
print(df.shape, "— dataset ready")
"""

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python"},
            "colab": {"provenance":[]}
        },
        "nbformat":4, "nbformat_minor":5
    }

def md(text):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(text).strip().splitlines(keepends=True)}

def code(src):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "source":textwrap.dedent(src).strip().splitlines(keepends=True)}

def make_notebook(title:str, learning_obj:list[str], blocks:list[tuple[str,str]]):
    cells = []
    # Title + Learning Objectives
    lo = "\n".join(f"- {x}" for x in learning_obj)
    cells.append(md(f"# {title}\n\n> **Learning objectives**\n{lo}\n---"))
    # Bootstrap
    cells.append(code(BOOTSTRAP_CELL))
    # Content blocks
    for heading, body in blocks:
        cells.append(md(f"## {heading}"))
        cells.append(code(body))
    # Checkpoint
    cells.append(md("### Checkpoint\n- Note open questions and next steps."))
    return cells

def main():
    os.makedirs("notebooks", exist_ok=True)
    for fname, title, los, blocks in NOTEBOOKS:
        path = os.path.join("notebooks", fname)
        nb_json = nb(make_notebook(title, los, blocks))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=2)
        print("Wrote", path)

if __name__ == "__main__":
    main()

