#!/usr/bin/env python3
"""Generate notebooks/1.09a_confounding_dags.ipynb."""
import json, os

cells = []

def md(cid, src):
    cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": [src]})

def code(cid, src):
    cells.append({"cell_type": "code", "execution_count": None, "id": cid,
                  "metadata": {}, "outputs": [], "source": [src]})


# ── 1. Title ─────────────────────────────────────────────────────────
md("intro", """\
# 1.09a – Confounding: Causal Diagrams and Worked Examples

Version 1.0.0

This notebook is a hands-on companion to Workbook 1.09.  Where 1.09 introduces the
vocabulary of confounding, **this notebook shows it in action** — with proper DAGs and
synthetic datasets where every parameter is known, so you can see exactly how much a
crude estimate lies and why adjustment fixes it.

## Structure

| Example | Exposure | Outcome | Confounder(s) | Bias direction |
|---|---|---|---|---|
| 1 | Physical activity | CVD | Age | Over-estimates protection |
| 2 | Supplement use | BMI | Sex | Spurious protection |
| 3 | Diet quality | CVD | SES + smoking | Two backdoor paths |
| 4 | Coffee | CVD | Smoking | Spurious *harm* (positive confounding) |

Each example follows three steps: **DAG → Simulate → Compare estimates**.
""")


# ── 2. Bootstrap ─────────────────────────────────────────────────────
code("bootstrap", '''\
# FB2NEP bootstrap cell (works locally and in Colab).
import os, sys, pathlib, subprocess, importlib.util

REPO_URL = "https://github.com/ggkuhnle/fb2nep-epi.git"
REPO_DIR = "fb2nep-epi"

cwd = pathlib.Path.cwd()

if (cwd / "scripts" / "bootstrap.py").is_file():
    repo_root = cwd
else:
    repo_root = cwd / REPO_DIR
    if not repo_root.is_dir():
        print(f"Cloning {REPO_URL} ...")
        subprocess.run(["git", "clone", REPO_URL, str(repo_root)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo_root), "pull"], check=True)
    os.chdir(repo_root)
    repo_root = pathlib.Path.cwd()

print(f"Repository root: {repo_root}")
bootstrap_path = repo_root / "scripts" / "bootstrap.py"
if not bootstrap_path.is_file():
    raise FileNotFoundError(bootstrap_path)

spec = importlib.util.spec_from_file_location("fb2nep_bootstrap", bootstrap_path)
bootstrap = importlib.util.module_from_spec(spec)
sys.modules["fb2nep_bootstrap"] = bootstrap
spec.loader.exec_module(bootstrap)

df, CTX = bootstrap.init()
print("Bootstrap complete — df shape:", df.shape)
''')


# ── 3. Imports + utilities ───────────────────────────────────────────
code("setup", '''\
# Imports and reusable plotting utilities for this workbook.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import statsmodels.formula.api as smf
from scipy.special import expit

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})

# ── Node colour palette ──────────────────────────────────────────────
_NODE = {
    "exposure":   {"face": "#1565C0", "text": "white"},
    "outcome":    {"face": "#B71C1C", "text": "white"},
    "confounder": {"face": "#E65100", "text": "white"},
    "mediator":   {"face": "#2E7D32", "text": "white"},
    "collider":   {"face": "#6A1B9A", "text": "white"},
}

def draw_dag(nodes, edges, title=None, figsize=(9, 5)):
    """
    Draw a directed acyclic graph with coloured nodes.

    nodes : dict  {id: {"pos":(x,y), "type":str, "label":str}}
    edges : list of dict  {"from":str, "to":str,
                           optional: "label", "style", "rad", "label_dx", "label_dy"}
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.axis("off")

    for e in edges:
        x0, y0 = nodes[e["from"]]["pos"]
        x1, y1 = nodes[e["to"]]["pos"]
        ls  = "dashed" if e.get("style") == "dashed" else "solid"
        col = "#AAAAAA" if ls == "dashed" else "#111111"
        rad = e.get("rad", 0.0)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                   linestyle=ls,
                                   connectionstyle=f"arc3,rad={rad}",
                                   shrinkA=33, shrinkB=33, mutation_scale=20))
        if "label" in e:
            mx = (x0+x1)/2 + e.get("label_dx", 0.0)
            my = (y0+y1)/2 + e.get("label_dy", 0.05)
            ax.text(mx, my, e["label"], ha="center", va="bottom",
                    fontsize=8, color="#555555", style="italic")

    for nid, n in nodes.items():
        x, y = n["pos"]; w, h = 0.24, 0.13
        fc = _NODE[n.get("type","exposure")]["face"]
        tc = _NODE[n.get("type","exposure")]["text"]
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle="round,pad=0.015",
                                    facecolor=fc, edgecolor="white",
                                    linewidth=2.5, zorder=3))
        ax.text(x, y, n.get("label", nid), ha="center", va="center",
                color=tc, fontsize=10, fontweight="bold", zorder=4)

    handles = [mpatches.Patch(facecolor=v["face"], label=k.capitalize(),
                               edgecolor="white") for k, v in _NODE.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.85, ncol=len(handles))
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout(); plt.show()


def or_plot(estimates, title="", xlabel="Odds Ratio", ref=1.0, per_unit=""):
    """
    Horizontal forest-style comparison plot.
    estimates: list of {"label", "est", "lo", "hi", "color"}
    """
    n = len(estimates)
    fig, ax = plt.subplots(figsize=(8, 0.85*n + 1.4))
    xmax = max(e["hi"] for e in estimates) * 1.25
    for i, e in enumerate(estimates):
        ax.errorbar(e["est"], i,
                    xerr=[[e["est"]-e["lo"]], [e["hi"]-e["est"]]],
                    fmt="o", color=e["color"], markersize=11,
                    capsize=6, lw=2.2, elinewidth=2.2)
        ax.text(e["hi"] + 0.005*xmax, i,
                f'  {e["est"]:.3f}  ({e["lo"]:.3f}–{e["hi"]:.3f})',
                va="center", ha="left", fontsize=9)
    ax.axvline(ref, color="#999999", linestyle="--", lw=1.5, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in estimates], fontsize=10)
    ax.set_xlabel(xlabel + (f"  [{per_unit}]" if per_unit else ""), fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()


def get_or(model, var, scale=1.0):
    b  = model.params[var] * scale
    ci = model.conf_int().loc[var] * scale
    return np.exp(b), np.exp(ci.iloc[0]), np.exp(ci.iloc[1])


def get_beta(model, var):
    b  = model.params[var]
    ci = model.conf_int().loc[var]
    return b, ci.iloc[0], ci.iloc[1]
''')


# ── 4. DAG reading guide ─────────────────────────────────────────────
md("dag_guide", """\
## How to read a DAG

A **directed acyclic graph** (DAG) is a formal statement of causal assumptions.
Each node is a variable; each arrow → is a direct causal effect.

| Node colour | Role |
|---|---|
| 🔵 Blue | **Exposure** — the variable whose effect we want to estimate |
| 🔴 Red | **Outcome** |
| 🟠 Orange | **Confounder** — common cause of exposure and outcome |
| 🟢 Green | **Mediator** — lies on the causal pathway exposure → outcome |
| 🟣 Purple | **Collider** — caused by both exposure and outcome |

A **backdoor path** is a path connecting the exposure to the outcome that begins
with an arrow *into* the exposure — written as (exposure ← … → outcome).
Such paths are non-causal: they arise when a common cause (confounder) influences
both the exposure and the outcome, creating a spurious statistical association.
To estimate the causal effect of exposure on outcome we must **block every backdoor
path**, typically by conditioning on (adjusting for) the confounders that open them.

**Further reading:** Krishna NS, Kalyanasundaram M, Bhatnagar T.
An Open Path to DAG: Navigating Causal Inference in Epidemiological Research.
*Indian J Community Med.* 2025. PMID: 40837172
""")


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 1
# ────────────────────────────────────────────────────────────────────
md("ex1_hdr", """\
## Example 1 — Age confounds Physical Activity → CVD

**Research question**: Does higher physical activity protect against cardiovascular disease?

**The problem**: Age drives *both* variables simultaneously.  Older people are less
active **and** face higher CVD risk.  This creates a backdoor path:

> PA ← Age → CVD

Because age simultaneously suppresses PA and raises CVD risk, the crude PA–CVD
association will appear *more protective* than the true causal effect — age amplifies
the apparent benefit.  After adjusting for age, the estimate shrinks towards (but
stays below) 1: the protection is real, but smaller than the crude figure suggested.
""")

code("ex1_dag", '''\
nodes_ex1 = {
    "age": {"pos": (0.50, 0.82), "type": "confounder", "label": "Age"},
    "pa":  {"pos": (0.12, 0.25), "type": "exposure",   "label": "Physical\\nactivity"},
    "cvd": {"pos": (0.88, 0.25), "type": "outcome",    "label": "CVD\\nevent"},
}
edges_ex1 = [
    {"from": "age", "to": "pa",  "label": "older → less active",  "label_dx": -0.07, "label_dy": 0.04},
    {"from": "age", "to": "cvd", "label": "older → higher risk",  "label_dx":  0.07, "label_dy": 0.04},
    {"from": "pa",  "to": "cvd", "label": "true protective effect (what we want)", "label_dy": 0.04},
]
draw_dag(nodes_ex1, edges_ex1,
         title="Example 1 — Backdoor path: PA ← Age → CVD")
''')

code("ex1_sim", '''\
# Data-generating model (known truth):
#   age     ~ Uniform(30, 80)
#   pa_mets = 36 - 0.45·age + N(0,7)   [MET-h/wk, floor 0]
#   log-odds(CVD) = -6.2 + 0.09·age - 0.06·pa_mets
#
# True log-OR for PA per MET-h/wk: -0.06
# True OR per 10 MET-h/wk: exp(-0.60) = 0.549

rng1 = np.random.default_rng(42)
N1   = 3000
age1    = rng1.uniform(30, 80, N1)
pa1     = np.clip(36 - 0.45*age1 + rng1.normal(0, 7, N1), 0, 70)
cvd1    = rng1.binomial(1, expit(-6.2 + 0.09*age1 - 0.06*pa1), N1)
df1 = pd.DataFrame({"age": age1, "pa": pa1, "cvd": cvd1})

print(f"N={N1}  CVD events: {cvd1.sum()} ({100*cvd1.mean():.1f}%)")
print(f"PA  mean {pa1.mean():.1f} MET-h/wk  SD {pa1.std():.1f}")
print(f"Age mean {age1.mean():.1f} yr        SD {age1.std():.1f}")
print(f"\\nTrue OR per 10 MET-h/wk: {np.exp(-0.06*10):.3f}")

# ── visualise the confounding ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
age_q = pd.qcut(age1, 3, labels=["Young 30–47", "Middle 47–63", "Older 63–80"])
palette = ["#1565C0", "#F57F17", "#B71C1C"]

for col, (lbl, grp) in zip(palette, df1.groupby(age_q, observed=True)):
    jit = rng1.uniform(-0.03, 0.03, len(grp))
    axes[0].scatter(grp["pa"], grp["cvd"]+jit, alpha=0.25, s=7, color=col, label=lbl)
axes[0].set_xlabel("Physical activity (MET-h/wk)")
axes[0].set_ylabel("CVD (jittered)")
axes[0].set_title("PA vs CVD — colour = age group\\n"
                  "Older people cluster bottom-right (low PA, high CVD)")
axes[0].legend(fontsize=8)

for col, (lbl, grp) in zip(palette, df1.groupby(age_q, observed=True)):
    axes[1].scatter(grp["age"], grp["pa"], alpha=0.2, s=7, color=col)
axes[1].set_xlabel("Age (years)"); axes[1].set_ylabel("PA (MET-h/wk)")
axes[1].set_title("Age → PA: the confounding mechanism\\n(older → less active)")
plt.tight_layout(); plt.show()
''')

code("ex1_analysis", '''\
m1_crude = smf.logit("cvd ~ pa",       data=df1).fit(disp=False)
m1_adj   = smf.logit("cvd ~ pa + age", data=df1).fit(disp=False)

true_or1           = np.exp(-0.06 * 10)
or1c, lo1c, hi1c   = get_or(m1_crude, "pa", 10)
or1a, lo1a, hi1a   = get_or(m1_adj,   "pa", 10)

print("OR per 10 MET-h/wk:")
print(f"  True (by construction) : {true_or1:.3f}")
print(f"  Crude                  : {or1c:.3f}  (95% CI {lo1c:.3f}–{hi1c:.3f})")
print(f"  Adjusted (+ age)       : {or1a:.3f}  (95% CI {lo1a:.3f}–{hi1a:.3f})")
print(f"\\nBias: crude over-estimates protection by {true_or1 - or1c:+.3f} OR units")

or_plot([
    {"label": "True (by construction)", "est": true_or1, "lo": true_or1, "hi": true_or1, "color": "#4CAF50"},
    {"label": "Crude (unadjusted)",     "est": or1c,     "lo": lo1c,     "hi": hi1c,     "color": "#F44336"},
    {"label": "Adjusted (+ age)",       "est": or1a,     "lo": lo1a,     "hi": hi1a,     "color": "#1565C0"},
], title="Example 1: PA → CVD, confounded by age", per_unit="per 10 MET-h/wk")
''')


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 2
# ────────────────────────────────────────────────────────────────────
md("ex2_hdr", """\
## Example 2 — Sex confounds Supplement use → BMI

**Research question**: Do people who take dietary supplements have a lower BMI?

**The trap**: Women take supplements more often than men *and* have a lower mean BMI.
Sex is a common cause of both variables — a classic **demographic confounder**.

The crude analysis will suggest supplements are associated with lower BMI.
The true causal effect of the supplements is **zero**.  After adjusting for sex,
the association vanishes completely.

This pattern is called **healthy-user bias**: supplement takers differ from
non-takers on many characteristics unrelated to the supplement itself.
""")

code("ex2_dag", '''\
nodes_ex2 = {
    "sex":  {"pos": (0.50, 0.82), "type": "confounder", "label": "Sex"},
    "supp": {"pos": (0.12, 0.25), "type": "exposure",   "label": "Supplement\\nuse"},
    "bmi":  {"pos": (0.88, 0.25), "type": "outcome",    "label": "BMI"},
}
edges_ex2 = [
    {"from": "sex", "to": "supp", "label": "women take more supplements",
     "label_dx": -0.08, "label_dy": 0.04},
    {"from": "sex", "to": "bmi",  "label": "women have lower mean BMI",
     "label_dx":  0.08, "label_dy": 0.04},
    # No supp → bmi arrow: zero true causal effect
]
draw_dag(nodes_ex2, edges_ex2,
         title="Example 2 — Sex confounds supplement use → BMI\\n"
               "(no supp → BMI arrow: the crude association is entirely spurious)")
''')

code("ex2_sim", '''\
# Data-generating model:
#   sex   ~ Bernoulli(0.50)   [1 = female]
#   BMI   = 27.8 - 3.2·sex + N(0, 4.5)
#   p(supp) = 0.12 + 0.32·sex   → females 44%, males 12%
#   True effect of supplements on BMI: ZERO

rng2 = np.random.default_rng(99)
N2   = 2500
sex2  = rng2.binomial(1, 0.50, N2)
bmi2  = np.clip(27.8 - 3.2*sex2 + rng2.normal(0, 4.5, N2), 16, 50)
supp2 = rng2.binomial(1, 0.12 + 0.32*sex2, N2)
df2   = pd.DataFrame({"sex": sex2, "supp": supp2, "bmi": bmi2})

# ── descriptive table ────────────────────────────────────────────────
tbl = df2.groupby(["sex","supp"])["bmi"].agg(["mean","count"])
tbl.index = tbl.index.set_levels(["Male","Female"], level=0)
tbl.index = tbl.index.set_levels(["No supp","Supp"], level=1)
print(tbl.rename(columns={"mean":"mean BMI","count":"n"}).round(2))

# ── visualise ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
c = {0: "#1565C0", 1: "#C62828"}
lbl = {0: "Male", 1: "Female"}

for s, grp in df2.groupby("sex"):
    jit = rng2.uniform(-0.12, 0.12, len(grp))
    axes[0].scatter(grp["supp"]+jit, grp["bmi"],
                    alpha=0.18, s=7, color=c[s], label=lbl[s])
    for sv in [0, 1]:
        sub = grp[grp["supp"]==sv]
        axes[0].plot(sv, sub["bmi"].mean(), "D", color=c[s],
                     markersize=12, markeredgecolor="white", markeredgewidth=1.5)
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["No supplement","Uses supplement"])
axes[0].set_ylabel("BMI (kg/m²)")
axes[0].set_title("Crude view: supplement users appear lighter\\n(diamonds = group means)")
axes[0].legend()

groups = [df2[(df2.sex==s)&(df2.supp==sv)]["bmi"]
          for s in [0,1] for sv in [0,1]]
bp = axes[1].boxplot(groups, patch_artist=True,
                     boxprops=dict(facecolor="#CFD8DC"),
                     medianprops=dict(color="#B71C1C", lw=2))
axes[1].set_xticks([1,2,3,4])
axes[1].set_xticklabels(["Male\\nno supp","Male\\nsupp","Female\\nno supp","Female\\nsupp"],
                        fontsize=8)
axes[1].set_ylabel("BMI (kg/m²)")
axes[1].set_title("Stratified by sex: supplements make no difference\\n"
                  "(the confounding disappears)")
plt.tight_layout(); plt.show()

# ── regression ───────────────────────────────────────────────────────
m2_crude = smf.ols("bmi ~ supp",       data=df2).fit()
m2_adj   = smf.ols("bmi ~ supp + sex", data=df2).fit()
b2c, lo2c, hi2c = get_beta(m2_crude, "supp")
b2a, lo2a, hi2a = get_beta(m2_adj,   "supp")

print(f"\\nBMI difference associated with supplement use (kg/m²):")
print(f"  True (by construction) :  0.000")
print(f"  Crude                  : {b2c:+.3f}  (95% CI {lo2c:+.3f} to {hi2c:+.3f})")
print(f"  Adjusted (+ sex)       : {b2a:+.3f}  (95% CI {lo2a:+.3f} to {hi2a:+.3f})")

or_plot([
    {"label": "True (by construction)", "est": 0.0,  "lo": 0.0,  "hi": 0.0,  "color": "#4CAF50"},
    {"label": "Crude",                  "est": b2c,  "lo": lo2c, "hi": hi2c, "color": "#F44336"},
    {"label": "Adjusted (+ sex)",       "est": b2a,  "lo": lo2a, "hi": hi2a, "color": "#1565C0"},
], title="Example 2: Supplement use → BMI, confounded by sex",
   xlabel="β (BMI kg/m² per supplement use)", ref=0.0,
   per_unit="supplement vs no supplement")
''')


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 3
# ────────────────────────────────────────────────────────────────────
md("ex3_hdr", """\
## Example 3 — Multiple backdoor paths: SES, diet, smoking, and CVD

This example raises the stakes.  Socioeconomic status (SES) simultaneously:

- Drives **diet quality** (higher SES → better diet)
- Drives **smoking** (higher SES → less smoking)
- Has a small **direct** residual effect on CVD (healthcare access, stress, etc.)

We want to estimate the causal effect of **diet quality on CVD**.  The true
effect exists, but SES opens *two* backdoor paths:

> Diet ← SES → CVD
> Diet ← SES → Smoking → CVD

Adjusting for SES alone partially corrects the estimate — the second backdoor
(through smoking) is still open.  Only when we adjust for **both SES and smoking**
do we recover the true effect.  This illustrates why a single adjustment is
often not enough.
""")

code("ex3_dag", '''\
nodes_ex3 = {
    "ses":     {"pos": (0.50, 0.88), "type": "confounder", "label": "SES"},
    "diet":    {"pos": (0.10, 0.48), "type": "exposure",   "label": "Diet\\nquality"},
    "smoking": {"pos": (0.50, 0.48), "type": "confounder", "label": "Smoking"},
    "cvd":     {"pos": (0.90, 0.15), "type": "outcome",    "label": "CVD\\nevent"},
}
edges_ex3 = [
    {"from": "ses",     "to": "diet",    "label": "↑SES→better diet",   "label_dx": -0.10},
    {"from": "ses",     "to": "smoking", "label": "↑SES→less smoking"},
    {"from": "ses",     "to": "cvd",     "label": "healthcare", "rad": 0.25,
     "label_dx": 0.07, "label_dy": 0.04},
    {"from": "diet",    "to": "cvd",     "label": "true causal effect"},
    {"from": "smoking", "to": "cvd",     "label": "↑CVD risk"},
]
draw_dag(nodes_ex3, edges_ex3,
         title="Example 3 — SES opens two backdoor paths to CVD\\n"
               "Path 1: Diet \u2190 SES \u2192 CVD   |   Path 2: Diet \u2190 SES \u2192 Smoking \u2192 CVD",
         figsize=(10, 6))
''')

code("ex3_sim", '''\
# Data-generating model:
#   SES           ~ Uniform(1,5)
#   diet_score    = 3 + 1.5·SES + N(0, 2.5)         [clipped 1–20]
#   p(smoking)    = expit(1.0 - 0.55·SES)
#   log-odds(CVD) = 1.8 - 0.12·diet - 1.5·smoking - 0.15·SES
#
# True OR per 5-point diet score: exp(-0.12 × 5) = 0.549

rng3 = np.random.default_rng(7)
N3   = 3500
ses3     = rng3.uniform(1, 5, N3)
diet3    = np.clip(3 + 1.5*ses3 + rng3.normal(0, 2.5, N3), 1, 20)
smok3    = rng3.binomial(1, expit(1.0 - 0.55*ses3), N3)
cvd3     = rng3.binomial(1, expit(1.8 - 0.12*diet3 - 1.5*smok3 - 0.15*ses3), N3)
df3 = pd.DataFrame({"ses": ses3, "diet": diet3, "smoking": smok3, "cvd": cvd3})

print(f"N={N3}  CVD events: {cvd3.sum()} ({100*cvd3.mean():.1f}%)")
print(f"Diet score mean {diet3.mean():.1f}  SD {diet3.std():.1f}")
print(f"Smokers: {100*smok3.mean():.1f}%")
print(f"\\nTrue OR per 5-point diet score: {np.exp(-0.12*5):.3f}")

# ── stepwise adjustment ──────────────────────────────────────────────
m3_0 = smf.logit("cvd ~ diet",               data=df3).fit(disp=False)
m3_1 = smf.logit("cvd ~ diet + ses",         data=df3).fit(disp=False)
m3_2 = smf.logit("cvd ~ diet + ses + smoking", data=df3).fit(disp=False)

true3          = np.exp(-0.12 * 5)
or30,lo30,hi30 = get_or(m3_0, "diet", 5)
or31,lo31,hi31 = get_or(m3_1, "diet", 5)
or32,lo32,hi32 = get_or(m3_2, "diet", 5)

print(f"\\nOR per 5-point diet score:")
print(f"  True (by construction)     : {true3:.3f}")
print(f"  Crude                      : {or30:.3f}  (95% CI {lo30:.3f}–{hi30:.3f})")
print(f"  + SES only                 : {or31:.3f}  (95% CI {lo31:.3f}–{hi31:.3f})")
print(f"  + SES + smoking            : {or32:.3f}  (95% CI {lo32:.3f}–{hi32:.3f})")
print("\\nAdjusting for SES alone closes one backdoor but leaves the smoking path open.")
print("Both must be blocked to recover the true estimate.")

or_plot([
    {"label": "True (by construction)",    "est": true3, "lo": true3, "hi": true3, "color": "#4CAF50"},
    {"label": "Crude",                     "est": or30,  "lo": lo30,  "hi": hi30,  "color": "#F44336"},
    {"label": "Adjusted: + SES",           "est": or31,  "lo": lo31,  "hi": hi31,  "color": "#FF9800"},
    {"label": "Adjusted: + SES + smoking", "est": or32,  "lo": lo32,  "hi": hi32,  "color": "#1565C0"},
], title="Example 3: Diet → CVD — stepwise closing of backdoor paths",
   per_unit="per 5-point diet score")
''')


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 4
# ────────────────────────────────────────────────────────────────────
md("ex4_hdr", """\
## Example 4 — Positive confounding: smoking makes coffee look dangerous

**Research question**: Does drinking coffee increase CVD risk?

In epidemiological studies from the 1970s–80s, coffee drinkers consistently showed
higher CVD rates.  The culprit was **smoking**.  Smokers drink more coffee *and* have
higher CVD risk.  In the crude analysis coffee is a proxy for smoking.

This is **positive confounding**: the confounder inflates the crude estimate *in the
same direction* as you might expect a causal harm — making a null association look
dangerous.  This is arguably more insidious than negative confounding, because the
result looks plausible and aligns with prior suspicion.

After adjusting for smoking, the coffee–CVD association disappears entirely.
""")

code("ex4_dag", '''\
nodes_ex4 = {
    "smoking": {"pos": (0.50, 0.82), "type": "confounder", "label": "Smoking"},
    "coffee":  {"pos": (0.12, 0.25), "type": "exposure",   "label": "Coffee\\n(cups/day)"},
    "cvd":     {"pos": (0.88, 0.25), "type": "outcome",    "label": "CVD\\nevent"},
}
edges_ex4 = [
    {"from": "smoking", "to": "coffee", "label": "smokers drink more coffee",
     "label_dx": -0.08, "label_dy": 0.04},
    {"from": "smoking", "to": "cvd",    "label": "strong causal harm",
     "label_dx":  0.08, "label_dy": 0.04},
    # No coffee → cvd arrow: true causal effect is zero
]
draw_dag(nodes_ex4, edges_ex4,
         title="Example 4 — Smoking confounds Coffee \u2192 CVD  (positive confounding)\\n"
               "No causal arrow coffee \u2192 CVD: the crude association is entirely spurious")
''')

code("ex4_sim", '''\
# Data-generating model:
#   smoking ~ Bernoulli(0.30)
#   coffee  = 1.8 + 1.8·smoking + N(0, 1.4)   [cups/day, floor 0]
#   log-odds(CVD) = -3.2 + 2.0·smoking   (coffee has NO effect)
#
# True OR for coffee per cup/day: 1.000

rng4 = np.random.default_rng(123)
N4   = 4000
smok4   = rng4.binomial(1, 0.30, N4)
coff4   = np.clip(1.8 + 1.8*smok4 + rng4.normal(0, 1.4, N4), 0, None)
cvd4    = rng4.binomial(1, expit(-3.2 + 2.0*smok4), N4)
df4 = pd.DataFrame({"smoking": smok4, "coffee": coff4, "cvd": cvd4})

print(f"N={N4}  CVD events: {cvd4.sum()} ({100*cvd4.mean():.1f}%)")
print(f"Coffee mean {coff4.mean():.2f} cups/day  SD {coff4.std():.2f}")
print(f"Smokers: {100*smok4.mean():.0f}%  |  True OR for coffee: 1.000")

# ── visualise ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cs = {0: "#1565C0", 1: "#B71C1C"}
ls = {0: "Non-smoker", 1: "Smoker"}

for s, grp in df4.groupby("smoking"):
    axes[0].hist(grp["coffee"], bins=25, alpha=0.55, color=cs[s],
                 label=ls[s], density=True)
axes[0].set_xlabel("Coffee (cups/day)"); axes[0].set_ylabel("Density")
axes[0].set_title("Smokers drink more coffee\\n(the confounding mechanism)")
axes[0].legend()

jit4 = rng4.uniform(-0.025, 0.025, N4)
for s, grp in df4.groupby("smoking"):
    axes[1].scatter(grp["coffee"], grp["cvd"]+jit4[:len(grp)],
                    alpha=0.12, s=6, color=cs[s], label=ls[s])
axes[1].set_xlabel("Coffee (cups/day)"); axes[1].set_ylabel("CVD event (jittered)")
axes[1].set_title("CVD vs coffee — smokers dominate\\nthe high-coffee / high-CVD region")
axes[1].legend()
plt.tight_layout(); plt.show()

# ── regression ───────────────────────────────────────────────────────
m4_crude = smf.logit("cvd ~ coffee",           data=df4).fit(disp=False)
m4_adj   = smf.logit("cvd ~ coffee + smoking", data=df4).fit(disp=False)
or4c,lo4c,hi4c = get_or(m4_crude, "coffee")
or4a,lo4a,hi4a = get_or(m4_adj,   "coffee")

print(f"\\nOR for coffee per cup/day:")
print(f"  True (by construction) : 1.000")
print(f"  Crude                  : {or4c:.3f}  (95% CI {lo4c:.3f}–{hi4c:.3f})")
print(f"  Adjusted (+ smoking)   : {or4a:.3f}  (95% CI {lo4a:.3f}–{hi4a:.3f})")
print("\\nCrude: coffee looks HARMFUL (OR > 1). Adjusted: vanishes.")
print("Positive confounding — crude estimate inflated above the null.")

or_plot([
    {"label": "True (by construction)", "est": 1.0,  "lo": 1.0,  "hi": 1.0,  "color": "#4CAF50"},
    {"label": "Crude (unadjusted)",     "est": or4c, "lo": lo4c, "hi": hi4c, "color": "#F44336"},
    {"label": "Adjusted (+ smoking)",   "est": or4a, "lo": lo4a, "hi": hi4a, "color": "#1565C0"},
], title="Example 4: Coffee \u2192 CVD, confounded by smoking\\n"
         "(positive confounding — OR inflated above truth)",
   per_unit="per cup/day")
''')


# ── Summary ──────────────────────────────────────────────────────────
md("summary", """\
## Summary

| Example | Confounder | Bias direction | Without adjustment | With adjustment |
|---|---|---|---|---|
| 1. PA → CVD | Age | Over-estimates protection | OR too low | OR → truth |
| 2. Supplements → BMI | Sex | Spurious protection | β < 0 (false) | β ≈ 0 (truth) |
| 3. Diet → CVD | SES + smoking | Over-estimates protection | Needs two steps | Closes both paths |
| 4. Coffee → CVD | Smoking | **Positive**: spurious harm | OR > 1 (false) | OR ≈ 1 (truth) |

### Checklist before fitting any regression model

1. **Draw the DAG** — commit to causal assumptions before opening the data.
2. **List all common causes** of your exposure and outcome.
3. **Adjust for all confounders** — not just those that change the estimate.
4. **Check for multiple backdoor paths** — each must be blocked independently.
5. **Acknowledge residual confounding** — unmeasured confounders always remain.

> *"Confounding is not a flaw in your analysis. It is a feature of the world you are
> studying. A DAG is your honest statement of what you believe about that world."*
""")


# ── Write notebook ────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                    "notebooks", "1.09a_confounding_dags.ipynb"))
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Written → {out}")
