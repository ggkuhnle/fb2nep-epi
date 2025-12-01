#!/usr/bin/env python3
"""
Helper functions for simple table summaries and plots.

Currently provides:

- proportion_table: counts and proportions for one categorical variable.
- representation_table: compare sample proportions to Census (or other)
  reference proportions and compute representation ratios.
- plot_representation: bar plot of representation ratios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    counts = data[column].value_counts(dropna=dropna)
    props = data[column].value_counts(normalize=True, dropna=dropna)

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
        (for example, NHANES).
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
    """
    merged = sample_tab.merge(
        census_tab,
        on=key,
        how="outer",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"proportion": "sample_prop"})
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
    """
    df = df.copy().sort_values("representation_ratio")

    plt.figure(figsize=(6, 4))
    plt.bar(df[category_col].astype(str), df["representation_ratio"])
    plt.axhline(1.0, linestyle="--")
    plt.ylabel("Representation ratio (sample / reference)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


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
    # Preallocate an array to store the mean from each simulated sample.
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
        - '<ref_label>_count' : counts in the reference dataset
        - '<ref_label>_prop'  : proportions in the reference dataset
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
