# FB2NEP Cheat Sheet: Key Python Functions

This cheat sheet explains key Python functions and methods used in the FB2NEP practice notebook, tailored to epidemiological data analysis. Use it to understand the code and focus on interpreting results.

## 1. General Python and NumPy
- **`np.random.normal(0, scale, size)`** (NumPy)
  - **Purpose**: Generates random noise (jitter) for data variability.
  - **Example**: `np.random.normal(0, df['Age'].std() * 0.05, len(df))` adds noise to `Age` (5% of its standard deviation).
  - **Context**: Used to simulate slight variations in continuous variables for practice.

- **`np.log1p(x)`** (NumPy)
  - **Purpose**: Computes log(1 + x) for data transformation (stabilises variance).
  - **Example**: `np.log1p(df['Flavanol_Biomarker'])` transforms biomarker values.
  - **Context**: Used for log-transformation in regression analyses.

- **`np.sqrt(x)`** (NumPy)
  - **Purpose**: Computes square root for data transformation (reduces skewness).
  - **Example**: `np.sqrt(df['Systolic_BP'])` transforms blood pressure.
  - **Context**: Alternative transformation for continuous variables.

## 2. Pandas (Data Handling)
- **`pd.read_csv(file)`**
  - **Purpose**: Loads a CSV file into a DataFrame.
  - **Example**: `pd.read_csv('fb2nep.csv')` loads the dataset.
  - **Context**: Used to import the FB2NEP dataset.

- **`df.head()`**
  - **Purpose**: Displays the first 5 rows of a DataFrame.
  - **Example**: `df.head()` shows a preview of the dataset.
  - **Context**: Helps inspect data structure and column names.

- **`df.isna()`**
  - **Purpose**: Checks for missing values (returns True/False).
  - **Example**: `df.isna().mean()` calculates the proportion of missing values per column.
  - **Context**: Used in the missingness audit.

- **`df.groupby(by)`**
  - **Purpose**: Groups data by a column for summary statistics.
  - **Example**: `df.groupby('Sex')` groups by sex for Table 1 comparisons.
  - **Context**: Enables comparisons by sex, SES, or disease status.

- **`pd.concat(objs, axis)`**
  - **Purpose**: Combines DataFrames or Series (e.g., for tables).
  - **Example**: `pd.concat({c: summarise_series(df[c]) for c in cols}, axis=1)` creates Table 1.
  - **Context**: Used to build summary tables.

- **`df.select_dtypes(include)`**
  - **Purpose**: Selects columns by data type (e.g., numeric).
  - **Example**: `df.select_dtypes(include=[np.number])` gets numeric columns for jitter.
  - **Context**: Identifies continuous variables for transformations.

## 3. Matplotlib (Plotting)
- **`plt.scatter(x, y, alpha)`**
  - **Purpose**: Creates a scatter plot.
  - **Example**: `plt.scatter(df['Flavanol_DD'], df['Flavanol_Biomarker'], alpha=0.6)` plots diet diary vs. biomarker.
  - **Context**: Used for biomarker vs. diet diary comparison.

- **`plt.imshow(data, aspect, interpolation)`**
  - **Purpose**: Displays a matrix (e.g., missingness patterns).
  - **Example**: `plt.imshow(df.isna(), aspect='auto')` shows missing data.
  - **Context**: Visualises missingness matrix.

- **`plt.axhline(y)`**
  - **Purpose**: Adds a horizontal line to a plot.
  - **Example**: `plt.axhline(3, color='red')` marks residual thresholds.
  - **Context**: Used in residual and Bland-Altman plots.

## 4. Statsmodels (Regression)
- **`smf.ols(formula, data).fit()`**
  - **Purpose**: Fits a linear regression model.
  - **Example**: `smf.ols('Systolic_BP ~ Flavanol_Biomarker', df).fit()` regresses BP on nutrient intake.
  - **Context**: Used for nutrient vs. BP analysis.

- **`smf.logit(formula, data).fit(disp=False)`**
  - **Purpose**: Fits a logistic regression model for binary outcomes.
  - **Example**: `smf.logit('CVD_Incidence ~ Flavanol_Biomarker', df).fit()` models disease risk.
  - **Context**: Used for nutrient vs. disease association.

- **`model.summary()`**
  - **Purpose**: Prints regression results (coefficients, p-values, etc.).
  - **Example**: `bp_model.summary()` shows linear regression output.
  - **Context**: Used to interpret model results.

- **`model.params`**, **`model.bse`**
  - **Purpose**: Access regression coefficients and standard errors.
  - **Example**: `np.exp(model.params['Flavanol_Biomarker'])` computes odds ratio.
  - **Context**: Used in change-in-estimate analysis.

## 5. SciPy (Statistics)
- **`stats.chi2_contingency(table)`**
  - **Purpose**: Performs chi-squared test for independence.
  - **Example**: `stats.chi2_contingency(pd.crosstab(miss, df['CVD_Incidence']))` tests if missingness is random.
  - **Context**: Used in missingness audit for categorical variables.

- **`stats.ttest_ind(a, b, equal_var=False)`**
  - **Purpose**: Performs t-test for comparing means.
  - **Example**: `stats.ttest_ind(miss_val, not_miss_val)` compares groups for missingness.
  - **Context**: Used for continuous variables in missingness tests.

## 6. Lifelines (Cox Regression)
- **`CoxPHFitter()`**
  - **Purpose**: Creates a Cox proportional hazards model.
  - **Example**: `CoxPHFitter().fit(df, duration_col='Time_to_Event', event_col='CVD_Incidence')` fits Cox model.
  - **Context**: Used for time-to-event analysis (disease).

- **`model.print_summary()`**
  - **Purpose**: Displays Cox model results (hazard ratios, etc.).
  - **Example**: `cox_model.print_summary()` shows Cox regression output.
  - **Context**: Interprets nutrient vs. disease association.

- **`model.params_`**, **`model.standard_errors_`**
  - **Purpose**: Access hazard ratios and standard errors.
  - **Example**: `np.exp(model.params_['Flavanol_Biomarker'])` computes hazard ratio.
  - **Context**: Used in change-in-estimate for Cox models.

## Tips for Assessment
- **Table 1**: Use `table1(df, cols, by=MAPPING['sex'])` to compare groups. Try `by=MAPPING['outcome']` for disease.
- **Missingness**: Check `missingness_summary(df)` for proportions, then use `test_mar(df, var, group_by)` to test patterns.
- **Regression**: Use `fit_linear_model` for BP, `fit_model` for disease. Toggle `USE_COX` for Cox regression.
- **Transformations**: Set `TRANSFORM='log'` or `'sqrt'` in the mapping cell to experiment.
- **Interpretation**: Look at p-values, coefficients (or OR/HR), and residuals to draw conclusions.