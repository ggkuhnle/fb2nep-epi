#!/usr/bin/env python3
"""
Generate a synthetic but realistic cohort for FB2NEP (N≈25,000).

- Baseline age >= 40
- Baseline date between 2010-01-01 and 2015-12-31
- Follow-up duration ~ Uniform(5,10) years
- Outcomes: CVD_incident, Cancer_incident, with event dates
- Covariates: SES (ABC1/C2DE), IMD, SBP, family history, menopausal status, biomarkers
- Non-linear relations: BMI U-shape for CVD risk, alcohol J-shape, SBP quadratic in age, vitamin C saturation
- Cancer risk includes red meat intake above 50 g/day (thresholded linear effect)
- Sex differences: BMI (men higher mean/SD), smoking (men more current & former; older women less likely ever-smoked),
  SES (women more C2DE at every IMD level), physical activity (men more high-intensity),
  SBP (pre/peri-menopausal women lower by 2–5 mmHg)
- Outcomes: former smoking and physical activity included as predictors in both CVD and cancer models

Seed: 11088
Output: data/synthetic/fb2nep.csv
"""
from __future__ import annotations
import os
import datetime as dt
import numpy as np
import pandas as pd

# -----------------------
# Global parameters
# -----------------------
SEED = 11088
rng = np.random.default_rng(SEED)
N = 25_000

# -----------------------
# Helpers
# -----------------------
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

def simulate_event_time(log_h):
    """Draw exponential event times (years) given log-hazard."""
    rate = np.exp(log_h)
    rate = np.clip(rate, 1e-8, 5.0)  # avoid numeric extremes
    u = rng.random(len(rate))
    return -np.log(1 - u) / rate

def calibrate_intercept(linpred, follow_up_years, target, lo=-12.0, hi=-2.0, iters=25):
    """
    Binary search for intercept b0 so mean incidence by follow_up ~= target.
    NOTE: If simulated incidence is too HIGH, move the UPPER bound down (more negative).
    """
    b_lo, b_hi = lo, hi
    for _ in range(iters):
        b0_try = 0.5 * (b_lo + b_hi)
        rate = np.exp(b0_try + linpred)
        rate = np.clip(rate, 1e-8, 5.0)
        u = rng.random(len(rate))
        t_try = -np.log(1 - u) / rate
        p_try = (t_try <= follow_up_years).mean()
        if p_try > target:
            # too many events -> need *more negative* intercept: tighten the UPPER bound
            b_hi = b0_try
        else:
            # too few events -> need *less negative* intercept: raise the LOWER bound
            b_lo = b0_try
    return 0.5 * (b_lo + b_hi)

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

# -----------------------
# 1) Core structure
# -----------------------
age = trunc_normal(58, 10, 40, 90, N).round().astype(int)
sex = choice(["F","M"], p=[0.52,0.48], size=N)
IMD = choice([1,2,3,4,5], p=[0.22,0.24,0.22,0.18,0.14], size=N)

# SES depends on IMD and sex (women more likely C2DE at every deprivation level)
ses_probs_M = {1: 0.38, 2: 0.49, 3: 0.59, 4: 0.70, 5: 0.79}  # P(ABC1) for men
ses_probs_F = {1: 0.32, 2: 0.41, 3: 0.51, 4: 0.62, 5: 0.71}  # P(ABC1) for women
SES = np.array([
    "ABC1" if rng.random() < (ses_probs_M[i] if s == "M" else ses_probs_F[i]) else "C2DE"
    for i, s in zip(IMD, sex)
], dtype=object)

# Menopausal status (F only; M=NA). Age-patterned with a little noise.
menopausal_status = np.array(["NA"]*N, dtype=object)
isF = (sex == "F")
mstat = np.where(age < 45, "pre", np.where(age < 54, "peri", "post"))
flip = rng.random(N) < 0.05
mstat_noisy = np.where(flip, choice(["pre","peri","post"], size=N), mstat)
menopausal_status[isF] = mstat_noisy[isF]

# Smoking: sex x age interaction
#   Men:   ~22% current at 40, declining with age; older cohorts smoked more (lower p_never)
#   Women: ~13% current at 40, declining with age; older cohorts much less likely to have ever smoked
smk = []
for a, s in zip(age, sex):
    if s == "M":
        p_cur = float(np.clip(0.22 - 0.002 * (a - 40), 0.08, 0.22))
        p_nev = float(np.clip(0.45 - 0.001 * (a - 40), 0.30, 0.45))
    else:
        p_cur = float(np.clip(0.13 - 0.0015 * (a - 40), 0.05, 0.13))
        p_nev = float(np.clip(0.50 + 0.002  * (a - 40), 0.50, 0.70))
    p_for = max(1.0 - p_nev - p_cur, 0.02)
    tot = p_nev + p_for + p_cur
    smk.append(choice(["never", "former", "current"],
                      p=[p_nev/tot, p_for/tot, p_cur/tot]))
smoking = np.array(smk, dtype=object)

# Physical activity depends on IMD and sex (men more likely to report high-intensity activity)
pa_probs = {
    1: {"M": [0.40, 0.42, 0.18], "F": [0.54, 0.36, 0.10]},
    2: {"M": [0.33, 0.47, 0.20], "F": [0.47, 0.42, 0.11]},
    3: {"M": [0.26, 0.50, 0.24], "F": [0.40, 0.46, 0.14]},
    4: {"M": [0.21, 0.52, 0.27], "F": [0.35, 0.49, 0.16]},
    5: {"M": [0.18, 0.52, 0.30], "F": [0.32, 0.48, 0.20]},
}
PA = np.array([choice(["low", "moderate", "high"], p=pa_probs[i][s])
               for i, s in zip(IMD, sex)], dtype=object)

# Family history (with slight age tilt)
fhx_cvd    = (rng.random(N) < (0.20 + 0.002*(age-50)/10)).astype(int)
fhx_cancer = (rng.random(N) < (0.22 + 0.001*(age-50)/10)).astype(int)

# -----------------------
# 2) Anthropometry & energy
# -----------------------
# BMI: sex-specific means (men higher on average; women wider spread reflecting higher obesity prevalence)
bmi_mean = np.where(sex == "M", 28.5, 26.5)
bmi_sd   = np.where(sex == "M", 4.5,  5.0)
base_bmi = np.clip(rng.normal(bmi_mean, bmi_sd, N), 15, 55)
age_z = (age - age.mean()) / age.std()
smk_current = (smoking == "current").astype(float)
BMI = np.clip(base_bmi + 0.7*age_z + 0.9*smk_current, 15, 55)

pa_mult = np.array([{"low":0.92, "moderate":1.00, "high":1.10}[p] for p in PA])
sex_mult = np.where(sex == "M", 1.06, 1.00)
energy_kcal = np.exp(rng.normal(np.log(1900), 0.25, N)) * pa_mult * sex_mult

# -----------------------
# 3) Diet
# -----------------------
imd_inv = 6 - IMD  # 1..5 -> 5..1 deprivation score
fv_base    = 330 - 25*imd_inv + 20*(SES == "ABC1")
red_base   = 70 + 12*imd_inv - 30*(SES == "ABC1")
ssb_base   = 150 + 60*imd_inv - 20*(SES == "ABC1")
fibre_base = 18 + 0.018*fv_base
salt_base  = 6.3 + 0.55*imd_inv - 0.2*(SES == "ABC1")

energy_scale = (energy_kcal/2000)
FV   = rng.normal(fv_base*energy_scale,   60)
FIB  = rng.normal(fibre_base*energy_scale, 5)
SALT = rng.normal(salt_base*energy_scale, 1.1)

# Red meat and SSB: explicitly right-skewed (log-normal-like)

# We treat red_base * energy_scale as a target mean and back-calculate
# the log-normal μ parameter for a chosen σ. This preserves an
# interpretable magnitude while inducing skewness.
red_mean = red_base * energy_scale
sigma_red = 0.75  # larger σ -> more skew; adjust if needed

# Prevent numerical problems for very small means
mu_red = np.log(np.maximum(red_mean, 1e-3)) - 0.5 * sigma_red**2
RED = rng.lognormal(mean=mu_red, sigma=sigma_red)

# SSB can be even more skewed
ssb_mean = ssb_base * energy_scale
sigma_ssb = 0.90
mu_ssb = np.log(np.maximum(ssb_mean, 1e-3)) - 0.5 * sigma_ssb**2
SSB = rng.lognormal(mean=mu_ssb, sigma=sigma_ssb)

FV   = np.clip(FV,   0, 1500)
RED  = np.clip(RED,  0, 500)
SSB  = np.clip(SSB,  0, 2500)
FIB  = np.clip(FIB,  0, 100)
SALT = np.clip(SALT, 2, 20)

# Alcohol (units/week), sex-specific distributions; used in CVD J-shape penalty
alc_levels = np.array([0, 4, 8, 16, 28])
p_female = np.array([0.38, 0.30, 0.20, 0.09, 0.03])
p_male   = np.array([0.28, 0.30, 0.23, 0.14, 0.05])
alcohol = []
for s in sex:
    p = p_female if s == "F" else p_male
    alcohol.append(rng.choice(alc_levels, p=p))
alcohol = np.array(alcohol, dtype=float)

# -----------------------
# 4) Biomarkers & SBP
# -----------------------
# Plasma vitamin C: saturation with FV
vitC_mean = 12 + 0.12*FV - 0.00006*(FV**2)
vitC_sd   = np.where(FV < 200, 10, 7)
vitC = rng.normal(vitC_mean, vitC_sd)
vitC = np.clip(vitC, 5, 120)

# Urinary sodium: linear with salt
urNa_mean = 25 + 11*SALT
urNa = rng.normal(urNa_mean, 25)
urNa = np.clip(urNa, 10, 250)

# SBP: quadratic in age + salt + BMI + smoking - high PA
SBP = (95 + 0.7*age + 0.03*((age-60)**2)
       - 1.5*(PA=="high").astype(float)
       + 1.5*SALT + 0.5*BMI + 4*(smoking=="current").astype(float)
       + rng.normal(0,10,N))
# Sex/menopausal adjustment: pre/peri-menopausal women have lower SBP; post-menopausal approaches male levels
sbp_sex_adj = np.where(
    (sex == "F") & (menopausal_status == "pre"),  -5.0,
    np.where((sex == "F") & (menopausal_status == "peri"), -2.0,
    np.where((sex == "F") & (menopausal_status == "post"),  1.5, 0.0))
)
SBP = np.clip(SBP + sbp_sex_adj, 80, 220)

# -----------------------
# 5) Baseline dates & follow-up
# -----------------------
baseline_date = make_dates(N, "2010-01-01", "2015-12-31")
follow_up_years = rng.uniform(5.0, 10.0, size=N)

# -----------------------
# 6) Outcomes (auto-calibrated intercepts)
# -----------------------
male     = (sex == "M").astype(float)
ses_low  = (SES == "C2DE").astype(float)
imd_term = (imd_inv - 3) / 2.0
alc_j    = np.maximum(0, alcohol - 8) * 0.02  # penalty beyond ~8 units/wk
smk_fmr  = (smoking == "former").astype(float)
pa_cvd   = np.array([{"low": 0.20, "moderate": 0.0, "high": -0.15}[p] for p in PA])

# CVD linear predictor (no intercept)
lin_cvd = (
    0.020*age
    + 0.030*BMI
    + 0.008*((BMI - 23)**2)                     # U-shape
    + 0.70*(smoking == "current").astype(float)
    + 0.30*smk_fmr                              # former smokers: intermediate risk
    + pa_cvd                                    # low PA raises risk, high PA lowers it
    + 0.25*male
    + 0.080*imd_term
    + 0.050*ses_low
    + 0.002*SALT
    + 0.018*SBP
    - 0.0005*FV
    + 0.015*alc_j
    + 0.35*fhx_cvd
)
b0_cvd = calibrate_intercept(lin_cvd, follow_up_years, target=0.12)
logh_cvd = b0_cvd + lin_cvd
t_cvd = simulate_event_time(logh_cvd)
incident_cvd = (t_cvd <= follow_up_years).astype(int)

# Cancer linear predictor (includes red meat >50 g/d)
red_excess = np.maximum(0, RED - 50.0)
pa_ca = np.array([{"low": 0.12, "moderate": 0.0, "high": -0.10}[p] for p in PA])
lin_ca = (
    0.045*age
    + 0.015*BMI
    + 0.55*(smoking == "current").astype(float)
    + 0.20*smk_fmr                                       # former smokers: residual cancer risk
    + pa_ca                                              # physical inactivity raises cancer risk
    + 0.15*ses_low
    + 0.05*male
    + 0.35*fhx_cancer
    + 0.0025*red_excess * ses_low                        # +0.10 per +100 g/d above 50
)
b0_ca = calibrate_intercept(lin_ca, follow_up_years, target=0.10)
logh_ca = b0_ca + lin_ca
t_ca = simulate_event_time(logh_ca)
incident_cancer = (t_ca <= follow_up_years).astype(int)

# Event dates
cvd_date = make_event_date(baseline_date, t_cvd, incident_cvd)
ca_date  = make_event_date(baseline_date, t_ca,  incident_cancer)

# -----------------------
# 7) Assemble DataFrame
# -----------------------
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
    "BMI": np.round(BMI, 1),
    "SBP": np.round(SBP, 1),
    "energy_kcal": np.round(energy_kcal, 0),
    "fruit_veg_g_d": np.round(FV, 1),
    "red_meat_g_d": np.round(RED, 1),
    "ssb_ml_d": np.round(SSB, 0),
    "fibre_g_d": np.round(FIB, 1),
    "alcohol_units_wk": alcohol,
    "salt_g_d": np.round(SALT, 1),
    "plasma_vitC_umol_L": np.round(vitC, 1),
    "urinary_sodium_mmol_L": np.round(urNa, 1),
    "CVD_incident": incident_cvd.astype(int),
    "CVD_date": cvd_date,
    "Cancer_incident": incident_cancer.astype(int),
    "Cancer_date": ca_date
})

# -----------------------
# 8) Missingness (MCAR + MAR + MNAR)
# -----------------------
# MCAR ~5% on selected vars (raised from 2.5% for clearer teaching illustration)
mcar_cols = [
    "fruit_veg_g_d","red_meat_g_d","ssb_ml_d","fibre_g_d","salt_g_d",
    "plasma_vitC_umol_L","urinary_sodium_mmol_L","alcohol_units_wk","SBP",
    "SES_class",
]
for c in mcar_cols:
    m = rng.random(N) < 0.05
    df.loc[m, c] = np.nan

# MAR by IMD (1–2) and age >=75
# Rates raised so that total missingness on key teaching vars reaches ~10-15%
for c in ["fruit_veg_g_d","fibre_g_d","plasma_vitC_umol_L"]:
    m = (rng.random(N) < (0.05 + 0.03*((IMD <= 2).astype(float))))
    df.loc[m, c] = np.nan
for c in ["urinary_sodium_mmol_L","salt_g_d","SBP"]:
    m = (rng.random(N) < (0.05 + 0.04*((age >= 75).astype(float))))
    df.loc[m, c] = np.nan

# MAR: smoking_status missing more often in low-SES participants
m = (rng.random(N) < (0.03 + 0.05*((IMD >= 4).astype(float))))
df.loc[m, "smoking_status"] = np.nan

# MNAR: BMI more likely missing when BMI is high (reluctance to be weighed)
m = (rng.random(N) < (0.05 + 0.08*((df["BMI"] > 35).astype(float))))
df.loc[m, "BMI"] = np.nan

# MNAR: alcohol under-reported at high intake
m = (rng.random(N) < (0.02*(alcohol > 16)))
df.loc[m, "alcohol_units_wk"] = np.nan

# -----------------------
# 9) Write to disk
# -----------------------
out_dir = os.path.join("data","synthetic")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fb2nep.csv")
df.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {df.shape}")
print(df.head())
