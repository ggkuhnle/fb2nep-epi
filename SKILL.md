# FB2NEP Skills Reference

This file defines reusable workflows and conventions for working on the fb2nep-epi teaching site.
Read this before performing any of the tasks listed below.

---

## Skill 1 — Update notebooks after dataset regeneration

**Use when:** `scripts/generate_dataset.py` has been re-run (new seed or modified generator), and notebook interpretation cells contain hardcoded regression results that no longer match the data.

### Step-by-step workflow

1. **Regenerate the data** (if not already done):
   ```bash
   source venv/bin/activate
   python scripts/generate_dataset.py
   ```

2. **Compute the standard set of reference models** and record their output:
   ```python
   # BMI ~ age + C(sex)  (linear)
   smf.ols("BMI ~ age + C(sex)", data=df).fit()
   # → age coeff, CI, sex coeff, CI, p-value, R²

   # hypertension ~ BMI + age + C(sex)  (logistic; hypertension = SBP ≥ 140)
   smf.logit("hypertension ~ BMI + age + C(sex)", data=df.dropna()).fit()
   # → BMI coeff, OR = exp(coeff), pseudo-R²

   # CVD_incident ~ highPA  (crude)
   # CVD_incident ~ highPA + C(SES_class)  (adjusted)
   # CVD_incident ~ highPA  stratified by SES_class
   ```

3. **Find all hardcoded values** that need updating:
   ```bash
   grep -n "0\.071\|0\.065\|0\.077\|0\.055\|0\.335\|0\.023\|0\.059\|1\.06\|0\.024\|1\.05\|0\.91\|0\.52\|0\.89\|0\.76\|0\.11" \
     notebooks/1.08_regression_modelling_01.ipynb \
     notebooks/1.09_regression_modelling_02.ipynb
   ```

4. **Present a before/after summary** of all values that need changing. **Do not edit any files yet.** Wait for explicit approval.

5. **After approval**, update interpretation markdown cells using `NotebookEdit` with the `cell_id` — do not use `Read` on the whole file (it exceeds token limits). Find cell IDs with:
   ```bash
   grep -n '"id":' notebooks/1.08_regression_modelling_01.ipynb
   ```

6. **Verify execution**:
   ```bash
   source venv/bin/activate
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 \
     --output /tmp/test_1.08.ipynb notebooks/1.08_regression_modelling_01.ipynb
   ```

### Key cell IDs to know

| Notebook | Cell ID | Contains |
|----------|---------|---------|
| 1.08 | `6e2873de` | Linear regression interpretation (age/sex/R²) |
| 1.08 | `46f07457` | Logistic regression output interpretation |
| 1.08 | `47ee75c5` | OR conversion text (exp(coeff) example) |
| 1.09 | `confounders_pa_interpret_fb2nep_7` | PA crude vs adjusted OR interpretation |
| 1.09 | `stratification_pa_interpret_fb2nep_7` | Stratum-specific OR values (ABC1, C2DE) |
| 1.09 | `a2_fv_cancer_interpretation` | Fruit/veg → Cancer confounding interpretation |

---

## Skill 2 — Add or revise a teaching example in a notebook

**Use when:** Adding a new analysis example, replacing an existing one, or revising an explanation cell.

### Rules

1. **Read the pedagogy style guide** at `memory/reference_pedagogy_style.md` before writing any content.
2. **Read the dataset pipeline reference** at `memory/reference_dataset_pipeline.md` to check whether the proposed example produces meaningful results with the current data (section 8: "Teaching examples — what works").
3. Follow the **cell ordering pattern**: intro markdown → code → interpretation markdown → exercise markdown.
4. Every code cell must have a **docstring** at the top.
5. Simulations come **before** FB2NEP examples, clearly labelled "Step 1 — Simulation" / "Step 2 — FB2NEP data".
6. Every new analysis section must include a **public health implication** paragraph.
7. When a new section is added, **increment the notebook version number** in the title cell.

### Never do
- Hardcode numerical results in markdown cells — compute them dynamically in code cells instead.
- Edit a notebook file with `Edit` or `Write` tools (these corrupt the JSON). Always use `NotebookEdit`.
- Read an entire large notebook with `Read` — use `grep '"id":' notebook.ipynb` to find cell IDs, then `sed -n 'X,Yp'` to read specific ranges.

---

## Skill 3 — Navigate and edit large notebooks

**Use when:** A notebook file exceeds the `Read` tool token limit (common in this project).

### Pattern

```bash
# 1. Get all cell IDs and their line numbers
grep -n '"id":' notebooks/1.08_regression_modelling_01.ipynb

# 2. Read a specific range around the cell you need
sed -n '392,460p' notebooks/1.08_regression_modelling_01.ipynb

# 3. Edit using cell_id (never requires reading the full file)
NotebookEdit(notebook_path=..., cell_id="target_id", new_source="...")

# 4. Verify execution after edits
jupyter nbconvert --to notebook --execute --output /tmp/test.ipynb notebooks/target.ipynb
```

---

## Skill 4 — Identify a good teaching example from the FB2NEP data

**Use when:** Choosing which exposure/confounder/outcome combination to use for a new section, or checking whether a proposed example will produce meaningful results.

### Procedure

1. Check `memory/reference_dataset_pipeline.md` section 8 first — it lists confirmed working and known weak examples.
2. If the desired example is not listed, run exploratory models:
   ```python
   source venv/bin/activate && python3 -c "
   import pandas as pd, numpy as np, statsmodels.formula.api as smf
   df = pd.read_csv('data/synthetic/fb2nep.csv')
   # ... model code
   "
   ```
3. A good teaching example needs:
   - A **meaningful effect size** (OR not too close to 1)
   - A **statistically significant crude association** (p < 0.05) — otherwise the confounding/adjustment story is weak
   - A **clear change between crude and adjusted** estimates (≥ 10–15% change in OR) for confounding examples
4. Update section 8 of the dataset reference file with any newly validated examples.

---

## Skill 5 — Plan-then-approve for complex multi-notebook changes

**Use when:** A task involves editing interpretation text in 2 or more notebooks, or any change to content that students read.

### Rule

**Never implement multi-notebook changes without explicit approval.** The CLAUDE.md instruction to "proceed directly" applies to code-level tasks (fixing bugs, running commands). For teaching content:

1. Draft a before/after summary of all proposed text changes.
2. Use `ExitPlanMode` to present the plan and wait for approval.
3. Only proceed after the user confirms.

This overrides the general "proceed directly" instruction from CLAUDE.md for this project.

---

## Quick reference

| Task | Skill to use |
|------|-------------|
| Dataset regenerated, numbers wrong | Skill 1 |
| Add new analysis section to notebook | Skill 2 |
| Notebook too large to read | Skill 3 |
| Need to find a good teaching example | Skill 4 |
| Changing text in 2+ notebooks | Skill 5 |
| Any notebook edit | Always use `NotebookEdit`, never `Edit`/`Write` |
| Style question (tone, formatting, structure) | `memory/reference_pedagogy_style.md` |
| Dataset question (variables, effect sizes) | `memory/reference_dataset_pipeline.md` |
