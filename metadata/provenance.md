# Provenance (v2) — fb2nep synthetic cohort

**Generator:** `scripts/generate_dataset.py`  
**Seed:** 11088  
**Sample size:** N≈25,000 adults  
**Baseline age:** ≥40 years  
**Period:** Baseline between 2010‑01‑01 and 2015‑12‑31; follow‑up uniformly 5–10 years.

## Design intents
- Teach realistic cohort structure with two endpoints: **CVD** and **Cancer**.  
- Provide both **incident indicators** and **event dates**.  
- Encode **SES** (ABC1/C2DE) and **IMD** gradients.  
- Add clinically relevant covariates: **SBP**, **family history**, **menopausal status**.  
- Include selected **non‑linear associations** (BMI U‑shape, alcohol J‑shape, SBP quadratic in age, vitamin C saturation).  
- Add **red meat intake** as a positive risk factor for Cancer above ~50 g/d.

## Variable generation (summary)
- **Age** ~ truncated Normal(μ=58, σ=10, 40–90). **Sex**: 52% F, 48% M.  
- **IMD_quintile** skewed to 2–4.  
- **SES_class** depends on IMD.  
- **Menopausal status** age‑patterned.  
- **Smoking** ~15% current.  
- **PA**: depends on IMD.  
- **BMI**: Normal(27, 4.5), rises with age/smoking.  
- **Energy**: log‑normal around 1900 kcal, scaled by PA/sex.  
- **Diet**: FV, red meat, SSB, fibre, salt depend on IMD, SES, energy.  
- **Biomarkers**: plasma vitC saturates with FV; urinary Na tracks salt.  
- **SBP**: non‑linear in age plus salt, BMI, smoking, PA.  

## Outcomes and dates
- **CVD**: age, BMI (U‑shape), smoking, sex, IMD, SES, salt, SBP, alcohol J‑shape, FHx CVD. Target 10–15%.  
- **Cancer**: age, BMI, smoking, SES, sex, FHx cancer, **red meat >50 g/d**. Target 8–12%.  
- Event dates: baseline + simulated time for incidents.

## Missingness
- **MCAR:** ~2–3%.  
- **MAR:** +5–8% tied to IMD/age.  
- **MNAR (tiny):** alcohol→alcohol missing; high BMI→BMI missing.

## Validation targets
- Corr(FV, vitC) > 0.45.  
- Corr(salt, urinary Na) > 0.55.  
- SBP vs age Spearman ρ > 0.35.  
- SES/IMD diet gradients as expected.  
- CVD incidence 10–15%; Cancer 8–12%.  
- Red meat higher in cancer cases.  
- Event dates present only if incident=1.

## Notes
This is **teaching data**: tuned for pedagogy, not inference.
