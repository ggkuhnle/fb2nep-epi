#!/usr/bin/env python3
"""
Helper functions for simple table summaries and plots.

Currently provides:

- proportion_table: counts and proportions for one categorical variable.
- representation_table: compare sample proportions to Census (or other)
  reference proportions and compute representation ratios.
- plot_representation: bar plot of representation ratios.
- draw_sample_mean_bmi / simulate_sampling_distribution: simple tools
  for illustrating sampling variation of the mean.
- compare_two_sources: side-by-side comparison of category distributions.
- make_table1: very simple 'Table 1' of baseline characteristics by group.
- log_transform / z_score: basic transformations often used before modelling.
- plot_hist_pair: helper to show the effect of a transformation.
- ensure_lifelines: import (and if necessary install) the lifelines package.
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Simple tables and representation
# ---------------------------------------------------------------------


def proportion_table(
    data: pd.DataFrame,
    column: str,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Return counts and proportions for one categorical column.

    Parameters
    ----------
    data : pandas.DataFrame
        The data frame that contains the variable of interest.
    column : str
        The name of the column (variable) to be tabulated.
    dropna : bool, default False
        If False (the default), missing values (NaN) are treated as a
        separate category. This is often useful in epidemiology, as it
        shows how much missing data there is. If True, missing values
        are excluded from the counts and proportions.

    Returns
    -------
    pandas.DataFrame
        A data frame indexed by category with two columns:
        - "count": number of observations in this category;
        - "proportion": fraction of all observations in this category.
    """
    # value_counts gives the frequency of each category.
    # dropna controls whether NaN is treated as its own category.
    counts = data[column].value_counts(dropna=dropna)

    # value_counts(normalize=True) gives the proportion (i.e. counts / total).
    props = data[column].value_counts(normalize=True, dropna=dropna)

    # Combine counts and proportions into a single table.
    table = pd.DataFrame(
        {
            "count": counts,
            "proportion": props,
        }
    )

    return table


def representation_table(
    sample_tab: pd.DataFrame,
    census_tab: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    """
    Merge sample and Census proportions and compute representation ratios.

    Parameters
    ----------
    sample_tab : pandas.DataFrame
        Table with columns [key, 'proportion'] from the sample
        (for example, NHANES). This would typically come from
        proportion_table(...).reset_index().
    census_tab : pandas.DataFrame
        Table with columns [key, 'census_prop'] from Census or other
        reference source.
    key : str
        Column name that identifies the categories (for example, 'sex').

    Returns
    -------
    pandas.DataFrame
        Data frame with one row per category, containing:
        - key
        - sample_prop
        - census_prop
        - representation_ratio = sample_prop / census_prop

    Notes
    -----
    - A representation_ratio of 1.0 means perfect agreement between the
      sample and the reference.
    - Values > 1.0 indicate over-representation in the sample.
    - Values < 1.0 indicate under-representation in the sample.
    """
    # Merge the two tables so that the same categories (rows) line up.
    # how="outer" ensures that categories that appear in only one table
    # still appear in the result (with NaN for the missing side).
    merged = sample_tab.merge(
        census_tab,
        on=key,
        how="outer",
        validate="one_to_one",
    )

    # Rename the sample proportion column for clarity.
    merged = merged.rename(columns={"proportion": "sample_prop"})

    # Compute the representation ratio; if census_prop is missing or zero,
    # the result will be NaN or inf.
    merged["representation_ratio"] = merged["sample_prop"] / merged["census_prop"]

    return merged


def plot_representation(
    df: pd.DataFrame,
    category_col: str,
    title: str,
) -> None:
    """
    Plot representation ratios for one categorical variable.

    A horizontal line at 1.0 indicates perfect agreement between
    sample (for example, NHANES) and reference (for example, Census).

    Parameters
    ----------
    df : pandas.DataFrame
        Table with columns [category_col, 'representation_ratio'].
    category_col : str
        Name of the column that contains category labels (for example, 'sex').
    title : str
        Title of the plot.

    Returns
    -------
    None
        The function creates and shows a matplotlib figure.
    """
    # Work on a copy so that the original data frame is not modified.
    df = df.copy().sort_values("representation_ratio")

    plt.figure(figsize=(6, 4))

    # Convert category labels to string so that unusual types (e.g. integers,
    # categories) are handled safely on the x-axis.
    plt.bar(df[category_col].astype(str), df["representation_ratio"])

    # Add a reference line at 1.0 (perfect representation).
    plt.axhline(1.0, linestyle="--")

    plt.ylabel("Representation ratio (sample / reference)")
    plt.title(title)

    # Rotate x-axis labels to avoid overlap if there are many categories.
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 2. Sampling and simulation helpers
# ---------------------------------------------------------------------


def draw_sample_mean_bmi(
    data: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> float:
    """
    Draw a simple random sample of size n and return its mean BMI.

    Parameters
    ----------
    data : pandas.DataFrame
        Data set from which we draw the sample. It must contain a
        numeric column called 'bmi'.
    n : int
        Desired sample size (number of rows to draw).
    rng : numpy.random.Generator
        Random number generator (created earlier with a fixed seed).

    Returns
    -------
    float
        The mean BMI in the sampled individuals.

    Notes
    -----
    - Sampling is done *without replacement*: once a row is selected,
      it cannot be selected again in the same sample.
    - This mimics the usual situation where each person in a study
      can only be recruited once.
    """
    # Convert the row index of the data frame to a NumPy array so that
    # rng.choice can select row positions directly.
    index_values = data.index.to_numpy()

    # Choose n distinct indices (replace=False ensures "without replacement").
    sample_indices = rng.choice(index_values, size=n, replace=False)

    # Select those rows and calculate the mean of the 'bmi' column.
    sample_mean = data.loc[sample_indices, "bmi"].mean()

    return sample_mean


def simulate_sampling_distribution(
    data: pd.DataFrame,
    n: int,
    n_sim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a sampling distribution of the mean BMI.

    Parameters
    ----------
    data : pandas.DataFrame
        Data set that plays the role of the population (here: NHANES).
    n : int
        Sample size for each simulated study.
    n_sim : int
        Number of repeated samples to draw.
    rng : numpy.random.Generator
        Random number generator used for all simulations.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of length n_sim containing the mean BMI
        from each simulated sample.

    Notes
    -----
    - Each simulation uses a fresh random sample of size n.
    - The distribution of these means approximates the *sampling
      distribution* of the mean BMI for studies of size n.
    """
    # Pre-allocate an array to store the mean from each simulated sample.
    means = np.empty(n_sim)

    # Repeat the sampling process n_sim times.
    for i in range(n_sim):
        means[i] = draw_sample_mean_bmi(data, n, rng)

    return means


def compare_two_sources(
    ref: pd.DataFrame,
    study: pd.DataFrame,
    column: str,
    ref_label: str,
    study_label: str,
) -> pd.DataFrame:
    """
    Compare category distributions between a reference dataset and a study dataset.

    Parameters
    ----------
    ref : pandas.DataFrame
        Reference dataset (for example, NHANES or Census margins).
    study : pandas.DataFrame
        Study dataset you want to compare (for example, FB2NEP cohort).
    column : str
        Name of the categorical variable to compare
        (for example, 'sex' or 'age_group').
    ref_label : str
        Short label used to name the reference columns
        (for example, 'nhanes' or 'census').
    study_label : str
        Short label used to name the study columns
        (for example, 'fb2nep').

    Returns
    -------
    pandas.DataFrame
        A table indexed by category with four columns:
        - '<ref_label>_count'  : counts in the reference dataset
        - '<ref_label>_prop'   : proportions in the reference dataset
        - '<study_label>_count': counts in the study dataset
        - '<study_label>_prop' : proportions in the study dataset

    Notes
    -----
    - Internally this uses `proportion_table`, which by default keeps
      missing values as a separate category. This makes it clear if
      one source has more missing data than the other.
    - An outer join is used so that categories present in only one
      of the datasets still appear in the final table.
    """
    # Tabulate counts and proportions in the reference dataset.
    ref_tab = proportion_table(ref, column).rename(
        columns={
            "count": f"{ref_label}_count",
            "proportion": f"{ref_label}_prop",
        }
    )

    # Tabulate counts and proportions in the study dataset.
    study_tab = proportion_table(study, column).rename(
        columns={
            "count": f"{study_label}_count",
            "proportion": f"{study_label}_prop",
        }
    )

    # Merge the two tables so that categories line up.
    # `how="outer"` ensures that if a category exists in only one
    # dataset, it still appears in the combined table.
    merged = ref_tab.merge(
        study_tab,
        left_index=True,
        right_index=True,
        how="outer",
    )

    return merged


# ---------------------------------------------------------------------
# 3. 'Table 1' helper
# ---------------------------------------------------------------------


def make_table1(
    data: pd.DataFrame,
    group: str,
    continuous: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """
    Create a simple 'Table 1' of baseline characteristics by group.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing all variables.
    group : str
        Name of the grouping variable (for example, 'sex', 'treatment').
        Each level of this variable will become a column in the table.
    continuous : list of str
        List of variable names to summarise as continuous variables.
        For each group, these will be shown as "mean ± SD".
    categorical : list of str
        List of variable names to summarise as categorical variables.
        For each group, these will be shown as "category: count (percent%)"
        for all observed categories.

    Returns
    -------
    pandas.DataFrame
        A table where:
        - rows are variables (continuous + categorical),
        - columns are groups (levels of the `group` variable),
        - cells are formatted strings ready to display in a notebook.
    """
    # Find all observed (non-missing) group levels, e.g. 'M' and 'F' for sex.
    # These will become the columns of the final table.
    groups = data[group].dropna().unique()

    # Dictionary to hold one column (summary) per group.
    # Keys: group levels; values: dict of {variable_name: formatted_string}.
    table: dict[str, dict[str, str]] = {}

    # Loop over each group level (for example 'M', 'F').
    for g in groups:
        # Subset the data frame to the current group only.
        df_g = data[data[group] == g]

        # Temporary dictionary to store all summaries for this group.
        col_dict: dict[str, str] = {}

        # 1) Summarise continuous variables as mean ± SD.
        for v in continuous:
            # Only proceed if the variable actually exists in the data frame.
            if v in data.columns:
                # Compute mean and standard deviation for this group.
                # By default, pandas ignores missing values (NaN).
                m = df_g[v].mean()
                s = df_g[v].std()

                # Format as "mean ± SD" with one decimal place.
                col_dict[v] = f"{m:.1f} ± {s:.1f}"

        # 2) Summarise categorical variables as counts and percentages.
        for v in categorical:
            # Only proceed if the variable exists in the data frame.
            if v in data.columns:
                # Count how often each category occurs in this group.
                # `normalize=False` returns absolute counts (not proportions).
                vc = df_g[v].value_counts(normalize=False)

                # Total number of participants in this group.
                total = len(df_g)

                # Build a string such as:
                # "never: 500 (40.0%); former: 600 (48.0%); current: 150 (12.0%)".
                col_dict[v] = "; ".join(
                    [
                        f"{cat}: {count} ({count / total * 100:.1f}%)"
                        for cat, count in vc.items()
                    ]
                )

        # Store the summary for this group as one column in the table.
        table[g] = col_dict

    # Convert the dictionary-of-dictionaries to a DataFrame:
    # - outer keys (groups) → columns
    # - inner keys (variable names) → rows.
    return pd.DataFrame(table)


# ---------------------------------------------------------------------
# 4. Simple transformations and plots
# ---------------------------------------------------------------------


def log_transform(x: pd.Series, constant: float = 0.0) -> pd.Series:
    """
    Apply a log transformation with optional constant.

    Parameters
    ----------
    x : pd.Series
        Variable to be transformed.
    constant : float, default 0.0
        Constant to be added before taking the logarithm.

    Returns
    -------
    pd.Series
        Log-transformed values.

    Notes
    -----
    We add a small constant if the variable can take the value 0, because
    log(0) is undefined. The constant changes the scale slightly but keeps
    the ordering of values.
    """
    return np.log(x + constant)


def plot_hist_pair(
    original: pd.Series,
    transformed: pd.Series,
    original_label: str,
    transformed_label: str,
) -> None:
    """
    Plot histograms of original and transformed data side by side.

    This function is intended as a simple visual aid when comparing the
    distribution of a variable before and after a transformation
    (for example, raw intake vs log-transformed intake, or raw values vs
    Box–Cox transformed values).

    Parameters
    ----------
    original : pd.Series
        Original values (for example, raw dietary intake or biomarker
        concentrations). Missing values (NaN) are removed before plotting.
    transformed : pd.Series
        Transformed values corresponding to the same variable (for example,
        log- or Box–Cox-transformed values). Missing values (NaN) are
        removed before plotting.
    original_label : str
        Label for the original variable. This is used as the x-axis label
        for the left-hand histogram.
    transformed_label : str
        Label for the transformed variable. This is used as the x-axis label
        for the right-hand histogram.

    Returns
    -------
    None
        The function creates a matplotlib figure and displays it. It does not
        return any object.
    """
    # ------------------------------------------------------------------
    # 1. Remove missing values
    # ------------------------------------------------------------------
    # Histograms cannot handle NaN values directly. We therefore drop any
    # missing values from both series before plotting. This affects only
    # the visualisation, not the underlying data frame.
    o = original.dropna()
    t = transformed.dropna()

    # ------------------------------------------------------------------
    # 2. Create a new figure and define its size
    # ------------------------------------------------------------------
    # We use a 1 × 2 layout (two panels next to each other). The figure
    # size can be adjusted if necessary, but 10 × 4 inches works well for
    # most notebook displays.
    plt.figure(figsize=(10, 4))

    # ------------------------------------------------------------------
    # 3. Left panel: histogram of the original data
    # ------------------------------------------------------------------
    plt.subplot(1, 2, 1)          # First subplot: 1 row, 2 columns, position 1
    o.hist(bins=30)               # Use 30 bins as a reasonable default
    plt.xlabel(original_label)    # Label the x-axis with the original variable
    plt.ylabel("Number of participants")
    plt.title("Original scale")

    # ------------------------------------------------------------------
    # 4. Right panel: histogram of the transformed data
    # ------------------------------------------------------------------
    plt.subplot(1, 2, 2)          # Second subplot: position 2
    t.hist(bins=30)
    plt.xlabel(transformed_label)  # Label the x-axis with the transformed variable
    plt.ylabel("Number of participants")
    plt.title("Transformed scale")

    # ------------------------------------------------------------------
    # 5. Adjust layout and display the figure
    # ------------------------------------------------------------------
    # tight_layout() reduces overlap between axis labels and titles.
    plt.tight_layout()
    plt.show()


def z_score(x: pd.Series) -> pd.Series:
    """
    Return the z-score of a variable.

    Parameters
    ----------
    x : pd.Series
        Variable to standardise.

    Returns
    -------
    pd.Series
        Standardised variable with mean 0 and standard deviation 1.

    Notes
    -----
    The function subtracts the mean and divides by the standard deviation.
    """
    return (x - x.mean()) / x.std()


# ---------------------------------------------------------------------
# 5. Package helper
# ---------------------------------------------------------------------


def ensure_lifelines():
    """
    Import the 'lifelines' package, installing it first if necessary.

    This helper is mainly intended for teaching notebooks, where students
    may run the code in different environments (for example, Google Colab
    or a local Jupyter installation).

    Returns
    -------
    module
        The imported lifelines module.

    Behaviour
    ---------
    - If lifelines is already installed, it is imported and returned.
    - If lifelines is not installed, the function attempts to install it
      using `pip` and then imports it.
    """
    try:
        import lifelines  # noqa: F401  (imported for its side effect)
    except ImportError:
        print("The 'lifelines' package is not installed. Installing it now...")
        # Use the current Python interpreter to run 'pip install lifelines'.
        # This is more robust than assuming a particular 'pip' command.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lifelines"])
        print("Installation complete. Importing 'lifelines'...")
        import lifelines  # noqa: F401

    # Return the module so that it can be used by the caller.
    return lifelines

def summarise_logit_coef(model, var_name, label=None, prefix=None):
    """
    Create a small summary for one coefficient from a fitted logistic regression model.

    This helper is intended for use with statsmodels Logit results. It extracts:

    - The log-odds coefficient (beta).
    - The 95 % confidence interval for beta.
    - The p-value.
    - The odds ratio (OR = exp(beta)).
    - The 95 % confidence interval for the odds ratio.

    Parameters
    ----------
    model : statsmodels.discrete.discrete_model.BinaryResults
        Fitted logistic regression model (for example, from smf.logit(...).fit()).
    var_name : str
        Name of the coefficient / parameter in model.params (for example, "high_red").
    label : str, optional
        Human-readable label for this coefficient (for example, "Crude model").
        If None, var_name is used.
    prefix : str, optional
        Optional prefix to add to the returned index keys (for example, "crude_").
        This can be useful when combining summaries from several models.

    Returns
    -------
    pandas.Series
        A one-dimensional object with the following fields:

        - "label"          : label for this coefficient / model.
        - "var_name"       : name of the coefficient in the model.
        - "beta"           : log-odds coefficient.
        - "ci_lower"       : lower 95 % confidence limit for beta.
        - "ci_upper"       : upper 95 % confidence limit for beta.
        - "p_value"        : p-value for the coefficient.
        - "OR"             : odds ratio = exp(beta).
        - "OR_ci_lower"    : lower 95 % confidence limit for the odds ratio.
        - "OR_ci_upper"    : upper 95 % confidence limit for the odds ratio.

    Notes
    -----
    This function does not print anything. It is designed so that several
    Series objects can be combined into a DataFrame, for example:

        rows = []
        rows.append(summarise_logit_coef(m_crude, "high_red", label="Crude"))
        rows.append(summarise_logit_coef(m_adj, "high_red", label="Adjusted"))
        summary = pd.DataFrame(rows)

    """

    if label is None:
        label = var_name

    # Extract coefficient, confidence interval and p-value from the model
    beta = float(model.params[var_name])
    ci_lo, ci_hi = model.conf_int().loc[var_name]
    p_val = float(model.pvalues[var_name])

    # Convert to odds ratio scale
    OR = float(np.exp(beta))
    OR_lo = float(np.exp(ci_lo))
    OR_hi = float(np.exp(ci_hi))

    data = {
        "label": label,
        "var_name": var_name,
        "beta": beta,
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "p_value": p_val,
        "OR": OR,
        "OR_ci_lower": OR_lo,
        "OR_ci_upper": OR_hi,
    }

    # Optionally add a prefix to each key (except label and var_name)
    if prefix:
        prefixed = {}
        for key, value in data.items():
            if key in ("label", "var_name"):
                prefixed[key] = value
            else:
                prefixed[f"{prefix}{key}"] = value
        data = prefixed

    return pd.Series(data)

def stan_summary_table(fit, param_order=None, ci=0.95):
    """
    Create a tidy summary table for a CmdStanPy MCMC fit.

    This helper produces a compact table with posterior means, standard deviations,
    and credible intervals, formatted in a structure similar to the other summary
    tables used in FB2NEP (for example, complete-case and imputation outputs).

    Parameters
    ----------
    fit : cmdstanpy.CmdStanMCMC
        The fitted Stan model returned by CmdStanPy.
    param_order : list of str, optional
        If provided, parameters will appear in this order. Useful for placing
        regression coefficients (e.g. intercept, BMI, age, sex) in a consistent
        layout for teaching.
    ci : float, default 0.95
        The posterior credible interval width. The table will show the lower and
        upper bounds of the equal-tailed interval.

    Returns
    -------
    pandas.DataFrame
        A tidy table with columns:
        - 'mean'
        - 'sd'
        - 'lower'
        - 'upper'

    Notes
    -----
    - Internally uses CmdStanPy's ``summary()`` method.
    - By default, parameters appear in the Stan model order unless
      ``param_order`` is explicitly supplied.
    - Designed for use in teaching notebooks so that Bayesian results match the
      layout of frequentist results from ``statsmodels``.
    """

    import pandas as pd

    # Extract raw summary from CmdStanPy
    summ = fit.summary()

    # Identify credible interval column names from CmdStanPy's conventions
    ci_low = f"{int((1-ci)/2 * 100)}%"
    ci_high = f"{int((1 + ci)/2 * 100)}%"

    # Build the core table
    tbl = pd.DataFrame({
        "mean": summ["Mean"],
        "sd": summ["SD"],
        "lower": summ[ci_low],
        "upper": summ[ci_high],
    })

    # Reorder if a parameter order is provided
    if param_order is not None:
        tbl = tbl.loc[param_order]

    return tbl
