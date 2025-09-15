#!/usr/bin/env python3
"""
Generate a synthetic but realistic cohort for FB2NEP (N≈25,000).

- Baseline age >= 40
- Baseline date between 2010-01-01 and 2015-12-31
- Follow-up duration ~ Uniform(5,10) years
- Outcomes: CVD_incident, Cancer_incident, with event dates
- Covariates include SES (ABC1/C2DE), IMD, SBP, family history, menopausal status, biomarkers
- Non-linear relations: BMI U-shape for CVD risk, alcohol J-shape, age^2 term for SBP, VitC saturation

Seed: 11088
Outputs: data/synthetic/fb2nep.csv
"""
from __future__ import annotations
import os, math, datetime as dt
import numpy as np
import pandas as pd

SEED = 11088
rng = np.random.default_rng(SEED)
N = 25000

def trunc_normal(mean, sd, low, high, size):
    x = rng.normal(mean, sd, size)
    return np.clip(x, low, high)

def choice(a, p=None, size=None):
    return rng.choice(a, size=size, p=p)

def make_dates(n, start="2010-01-01", end="2015-12-31"):
    start = dt.date.fromisoformat(start)
    end   = dt.date.fromisoformat(end)
    days  = (end - start).days
    offs  = rng.integers(0, days+1, size=n)
    return np.array([start + dt.timedelta(int(d)) for d in offs])

# 1) Core structure
age = trunc_normal(58, 10, 40, 90, N).round().astype(int)
sex = choice(["F","M"], p=[0.52,0.48], size=N)
IMD = choice([1,2,3,4,5], p=[0.22,0.24,0.22,0.18,0.14], size=N)

# SES depends on IMD (more affluent with higher IMD quintile)
ses_probs = {1:0.35, 2:0.45, 3:0.55, 4:0.66, 5:0.75} # P(ABC1)
SES = np.array(["ABC1" if rng.random()<ses_probs[i] else "C2DE" for i in IMD])

# Menopausal status (F only; M=NA)
# Age-patterned: pre (<45), peri (45-53), post (>=54) with noise
menopausal_status = np.array(["NA"]*N, dtype=object)
isF = (sex=="F")
mstat = np.where(age<45, "pre", np.where(age<54, "peri", "post"))
# small noise to allow older pre/peri etc.
flip = rng.random(N) < 0.05
mstat_noisy = np.where(flip, choice(["pre","peri","post"], size=N), mstat)
menopausal_status[isF] = mstat_noisy[isF]

# Smoking
# More never in younger, more former in older; keep ~15% current overall
smk = []
for a in age:
    p_never = np.clip(0.45 + (40-a)/200, 0.25, 0.65)
    p_current = 0.15
    p_former = 1 - p_never - p_current
    smk.append(choice(["never","former","current"], p=[p_never, p_former, p_current]))
smoking = np.array(smk, dtype=object)

# Physical activity depends on IMD
pa_map = {
    1:[0.48,0.39,0.13],
    2:[0.40,0.45,0.15],
    3:[0.33,0.48,0.19],
    4:[0.28,0.51,0.21],
    5:[0.25,0.50,0.25],
}
PA = np.array([choice(["low","moderate","high"], p=pa_map[i]) for i in IMD], dtype=object)

# Family history (independent, with some IMD/age tilt)
fhx_cvd    = (rng.random(N) < (0.20 + 0.002*(age-50)/10)).astype(int)
fhx_cancer = (rng.random(N) < (0.22 + 0.001*(age-50)/10)).astype(int)

# BMI baseline with small age, smoking effects
base_bmi = trunc_normal(27.0, 4.5, 15, 55, N)
age_z = (age - age.mean())/age.std()
smk_current = (smoking=="current").astype(float)
BMI = base_bmi + 0.7*age_z + 0.9*smk_current

# Energy intake (kcal/day) with PA and sex multiplier
pa_mult = np.array([{"low":0.92,"moderate":1.00,"high":1.10}[p] for p in PA])
sex_mult = np.where(sex=="M", 1.06, 1.00)
energy_kcal = np.exp(rng.normal(np.log(1900), 0.25, N)) * pa_mult * sex_mult

# Diet by IMD, SES and energy
imd_inv = 6 - IMD  # 1..5 -> 5..1 deprivation score
fv_base    = 330 - 25*imd_inv + 20*(SES=="ABC1")
red_base   = 70 + 12*imd_inv - 8*(SES=="ABC1")
ssb_base   = 150 + 60*imd_inv - 20*(SES=="ABC1")
fibre_base = 18 + 0.018*fv_base  # fibre tracks FV (scaled)
salt_base  = 6.3 + 0.55*imd_inv - 0.2*(SES=="ABC1")

energy_scale = (energy_kcal/2000)
FV   = rng.normal(fv_base*energy_scale,   60)
RED  = rng.normal(red_base*energy_scale,  30)
SSB  = rng.normal(ssb_base*energy_scale, 120)
FIB  = rng.normal(fibre_base*energy_scale, 5)
SALT = rng.normal(salt_base*energy_scale, 1.1)

# Bounds
FV   = np.clip(FV,   0, 1500)
RED  = np.clip(RED,  0, 500)
SSB  = np.clip(SSB,  0, 2000)
FIB  = np.clip(FIB,  0, 100)
SALT = np.clip(SALT, 2, 20)

# Alcohol (units/week) with modest SES/sex differences, J-shape later in risk
alc_levels = np.array([0, 4, 8, 16, 28])
p_female = np.array([0.38, 0.30, 0.20, 0.09, 0.03])
p_male   = np.array([0.28, 0.30, 0.23, 0.14, 0.05])

alcohol = []
for s in sex:
    p = p_female if s == "F" else p_male
    alcohol.append(rng.choice(alc_levels, p=p))
alcohol = np.array(alcohol, dtype=float)

# Biomarkers: Vit C saturation; Urinary Na linear
vitC_mean = 12 + 0.12*FV - 0.00006*(FV**2)  # saturating
vitC_sd   = np.where(FV<200, 10, 7)
vitC = rng.normal(vitC_mean, vitC_sd)
vitC = np.clip(vitC, 5, 120)

urNa_mean = 25 + 11*SALT
urNa = rng.normal(urNa_mean, 25)
urNa = np.clip(urNa, 10, 250)

# SBP: non-linear in age (quadratic), + salt, + BMI, + smoking, - high PA
SBP = (95 + 0.7*age + 0.03*((age-60)**2) - 1.5*(PA=="high").astype(float)
       + 0.8*SALT + 0.5*BMI + 4*(smoking=="current").astype(float)
       + rng.normal(0,10,N))
SBP = np.clip(SBP, 80, 220)

# Baseline & follow-up times
baseline_date = make_dates(N, "2010-01-01", "2015-12-31")
follow_up_years = rng.uniform(5.0, 10.0, size=N)

# --- Outcomes via exponential time-to-event (piecewise-constant hazard) ---

def simulate_event_time(log_h):
    """Draw exponential event times (years) given log-hazard."""
    rate = np.exp(log_h)
    rate = np.clip(rate, 1e-8, 5.0)  # keep numerics sane
    u = rng.random(len(rate))
    t = -np.log(1 - u) / rate
    return t  # years

def calibrate_intercept(linpred, follow_up_years, target, lo=-12.0, hi=-2.0, iters=25):
    """
    Find intercept b0 so that mean incidence over censoring horizon ~= target.
    Binary search on b0 for stability and speed.
    """
    b_lo, b_hi = lo, hi
    for _ in range(iters):
        b0_try = 0.5 * (b_lo + b_hi)
        t_try = simulate_event_time(b0_try + linpred)
        p_try = (t_try <= follow_up_years).mean()
        if p_try > target:
            # too many events -> decrease hazard -> make intercept more negative
            b_lo = b0_try
        else:
            b_hi = b0_try
    return 0.5 * (b_lo + b_hi)

# Helper terms
male     = (sex == "M").astype(float)
ses_low  = (SES == "C2DE").astype(float)
imd_term = (imd_inv - 3) / 2.0                 # centred IMD (−1..+1)
alc_j    = np.maximum(0, alcohol - 8) * 0.02   # J-shape penalty beyond ~8 units/wk

# --------------------
# CVD linear predictor
# --------------------
lin_cvd = (
    0.020 * age
    + 0.030 * BMI
    + 0.008 * ((BMI - 23) ** 2)               # U-shape
    + 0.70 * (smoking == "current").astype(float)
    + 0.25 * male
    + 0.080 * imd_term
    + 0.050 * ses_low
    + 0.005 * SALT
    + 0.004 * SBP
    - 0.0005 * FV
    + 0.015 * alc_j
    + 0.35 * fhx_cvd
)

# Calibrate to target incidence (e.g., 12%)
b0_cvd = calibrate_intercept(lin_cvd, follow_up_years, target=0.12)
logh_cvd = b0_cvd + lin_cvd
t_cvd = simulate_event_time(logh_cvd)
incident_cvd = (t_cvd <= follow_up_years).astype(int)

# -----------------------
# Cancer linear predictor
# -----------------------
# Thresholded red meat effect: only grams/day above 50 contribute to risk.

red_excess = np.maximum(0, RED - 50)  # RED is red_meat_g_d
lin_ca = (
    0.045 * age
    + 0.015 * BMI
    + 0.55  * (smoking == "current").astype(float)
    + 0.08  * ses_low
    + 0.05  * male
    + 0.35  * fhx_cancer
    + 0.0010 * red_excess          # e.g. +0.10 log-hazard per +100 g/d above 50 g
)

# Calibrate to target incidence (e.g., 10%)
b0_ca = calibrate_intercept(lin_ca, follow_up_years, target=0.10)
logh_ca = b0_ca + lin_ca
t_ca = simulate_event_time(logh_ca)
incident_cancer = (t_ca <= follow_up_years).astype(int)

# Event dates (where incident=1)
def add_years(d: dt.date, years: float) -> dt.date:
    days = int(round(years * 365.25))
    return d + dt.timedelta(days=days)

def make_event_date(bdates, t_years, incident):
    dates = []
    for b, t, inc in zip(bdates, t_years, incident):
        if inc == 1:
            dates.append(add_years(b, float(t)))
        else:
            dates.append("")
    return np.array(dates, dtype=object)

cvd_date = make_event_date(baseline_date, t_cvd, incident_cvd)
ca_date  = make_event_date(baseline_date, t_ca,  incident_cancer)



# Assemble DataFrame
df = pd.DataFrame({
    "id": np.arange(1, N+1),
    "baseline_date": pd.to_datetime(baseline_date).strftime("%Y-%m-%d"),
    "follow_up_years": np.round(follow_up_years, 2),
    "age": age,
    "sex": sex,
    "menopausal_status": menopausal_status,
    "IMD_quintile": IMD,
    "SES_class": SES,
    "smoking_status": smoking,
    "physical_activity": PA,
    "family_history_cvd": fhx_cvd,
    "family_history_cancer": fhx_cancer,
    "BMI": np.round(BMI,1),
    "SBP": np.round(SBP,1),
    "energy_kcal": np.round(energy_kcal, 0),
    "fruit_veg_g_d": np.round(FV,1),
    "red_meat_g_d": np.round(RED,1),
    "ssb_ml_d": np.round(SSB,0),
    "fibre_g_d": np.round(FIB,1),
    "alcohol_units_wk": alcohol,
    "salt_g_d": np.round(SALT,1),
    "plasma_vitC_umol_L": np.round(vitC,1),
    "urinary_sodium_mmol_L": np.round(urNa,1),
    "CVD_incident": incident_cvd.astype(int),
    "CVD_date": cvd_date,
    "Cancer_incident": incident_cancer.astype(int),
    "Cancer_date": ca_date
})

# Inject missingness
# MCAR ~2.5% on selected vars
mcar_cols = ["fruit_veg_g_d","red_meat_g_d","ssb_ml_d","fibre_g_d","salt_g_d",
             "plasma_vitC_umol_L","urinary_sodium_mmol_L","alcohol_units_wk","SBP"]
for c in mcar_cols:
    m = rng.random(N) < 0.025
    df.loc[m, c] = np.nan

# MAR by IMD (1–2) and age >=75
for c in ["fruit_veg_g_d","fibre_g_d","plasma_vitC_umol_L"]:
    m = (rng.random(N) < (0.03 + 0.015*((IMD<=2).astype(float))))
    df.loc[m, c] = np.nan
for c in ["urinary_sodium_mmol_L","salt_g_d","SBP"]:
    m = (rng.random(N) < (0.03 + 0.02*((age>=75).astype(float))))
    df.loc[m, c] = np.nan

# Tiny MNAR
m = (rng.random(N) < (0.01*(alcohol>16)))
df.loc[m, "alcohol_units_wk"] = np.nan
m = (rng.random(N) < (0.005*(BMI>35)))
df.loc[m, "BMI"] = np.nan

# Output
out_dir = os.path.join("data","synthetic")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fb2nep.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {df.shape}")
print(df.head())
