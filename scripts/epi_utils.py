"""
epi_utils.py - Utility functions for Epidemiology Teaching Course

This module contains the core functions for calculating DALYs, QALYs,
and running policy simulations. Import this module in the Colab notebooks.

Usage:
    # In Colab, first upload or fetch this file, then:
    from epi_utils import *

Author: University of Reading, Nutrition and Food Science
"""

import numpy as np
import pandas as pd

# =============================================================================
# LIFE TABLES AND REFERENCE DATA
# =============================================================================

# GBD 2019 Reference Life Table (abridged)
LIFE_TABLE = pd.DataFrame({
    'age': [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'life_expectancy': [88.9, 88.0, 84.0, 79.0, 74.1, 69.1, 64.2, 59.2, 54.3, 49.4, 44.5, 
                        39.7, 35.0, 30.4, 25.9, 21.6, 17.5, 13.7, 10.3, 7.4, 5.0]
})

# GBD 2019 Disability Weights for selected conditions
GBD_DISABILITY_WEIGHTS = pd.DataFrame({
    'condition': [
        'Mild anaemia',
        'Moderate anaemia',
        'Severe anaemia',
        'Moderate hearing loss',
        'Moderate depression',
        'Severe low back pain',
        'Blindness',
        'Moderate heart failure',
        'Severe dementia',
        'Untreated spinal cord injury (below neck)',
        'Terminal cancer with severe pain',
        'Type 2 diabetes without complications',
        'Type 2 diabetes with diabetic foot',
        'Obesity (class III, BMI >= 40)',
        'Iron deficiency'
    ],
    'disability_weight': [
        0.004, 0.052, 0.149, 0.027, 0.145, 0.325, 0.187, 0.072, 0.449,
        0.589, 0.569, 0.015, 0.133, 0.086, 0.004
    ],
    'category': [
        'Nutritional deficiency', 'Nutritional deficiency', 'Nutritional deficiency',
        'Sensory', 'Mental health', 'Musculoskeletal', 'Sensory', 'Cardiovascular',
        'Neurological', 'Injury', 'Cancer', 'Metabolic', 'Metabolic', 'Metabolic',
        'Nutritional deficiency'
    ]
})


# =============================================================================
# DALY CALCULATION FUNCTIONS
# =============================================================================

def get_life_expectancy(age, life_table=None):
    """
    Get remaining life expectancy at a given age using linear interpolation.
    
    Parameters
    ----------
    age : float
        Age in years
    life_table : pd.DataFrame, optional
        Life table with 'age' and 'life_expectancy' columns.
        Uses GBD 2019 reference table if not provided.
    
    Returns
    -------
    float
        Remaining life expectancy in years
    """
    if life_table is None:
        life_table = LIFE_TABLE
    return np.interp(age, life_table['age'], life_table['life_expectancy'])


def calculate_yll(deaths_by_age, life_table=None):
    """
    Calculate Years of Life Lost from mortality data.
    
    Parameters
    ----------
    deaths_by_age : dict
        Dictionary mapping age at death to number of deaths
        Example: {45: 50, 55: 200, 65: 500}
    life_table : pd.DataFrame, optional
        Life table to use. Defaults to GBD 2019 reference.
    
    Returns
    -------
    tuple
        (total_yll, breakdown_df) where breakdown_df shows YLL by age
    
    Example
    -------
    >>> deaths = {45: 50, 55: 200, 65: 500}
    >>> total_yll, breakdown = calculate_yll(deaths)
    >>> print(f"Total YLL: {total_yll:,.0f}")
    """
    if life_table is None:
        life_table = LIFE_TABLE
    
    results = []
    total_yll = 0
    
    for age, n_deaths in deaths_by_age.items():
        le = get_life_expectancy(age, life_table)
        yll = n_deaths * le
        total_yll += yll
        results.append({
            'age_at_death': age,
            'n_deaths': n_deaths,
            'life_expectancy': round(le, 1),
            'yll': round(yll, 1)
        })
    
    return total_yll, pd.DataFrame(results)


def calculate_yld(prevalence, disability_weight, duration=1):
    """
    Calculate Years Lived with Disability.
    
    Parameters
    ----------
    prevalence : int
        Number of prevalent cases (or incidence if using duration)
    disability_weight : float
        Disability weight (0-1), where 0 = perfect health, 1 = death
    duration : float, optional
        Average duration in years. Default=1 for prevalence-based calculation.
    
    Returns
    -------
    float
        Years Lived with Disability
    
    Example
    -------
    >>> yld = calculate_yld(prevalence=100000, disability_weight=0.15)
    >>> print(f"YLD: {yld:,.0f}")
    """
    return prevalence * disability_weight * duration


def calculate_dalys(deaths_by_age, prevalence, disability_weight, 
                    condition_name="Condition", verbose=True):
    """
    Calculate total DALYs for a condition (YLL + YLD).
    
    Parameters
    ----------
    deaths_by_age : dict
        Dictionary mapping age at death to number of deaths
    prevalence : int
        Number of prevalent cases
    disability_weight : float
        Disability weight (0-1)
    condition_name : str, optional
        Name for display purposes
    verbose : bool, optional
        If True, print summary. Default True.
    
    Returns
    -------
    dict
        Dictionary with 'yll', 'yld', and 'dalys' values
    
    Example
    -------
    >>> result = calculate_dalys(
    ...     deaths_by_age={55: 200, 65: 500},
    ...     prevalence=50000,
    ...     disability_weight=0.072,
    ...     condition_name="Heart Disease"
    ... )
    """
    yll, _ = calculate_yll(deaths_by_age)
    yld = calculate_yld(prevalence, disability_weight)
    dalys = yll + yld
    
    if verbose:
        print(f"\n{condition_name} - DALY Calculation")
        print("=" * 50)
        print(f"YLL (Years of Life Lost):          {yll:>12,.0f}")
        print(f"YLD (Years Lived with Disability): {yld:>12,.0f}")
        print("-" * 50)
        print(f"Total DALYs:                       {dalys:>12,.0f}")
        print(f"\nYLL proportion: {yll/dalys*100:.1f}%")
        print(f"YLD proportion: {yld/dalys*100:.1f}%")
    
    return {'yll': yll, 'yld': yld, 'dalys': dalys}


# =============================================================================
# QALY CALCULATION FUNCTIONS
# =============================================================================

def calculate_qalys_gained(intervention_effect, population, duration, 
                           dw_before, dw_after):
    """
    Calculate QALYs gained from an intervention.
    
    Parameters
    ----------
    intervention_effect : float
        Proportion of population that benefits (0-1)
    population : int
        Size of target population
    duration : float
        Years of benefit
    dw_before : float
        Disability weight before intervention
    dw_after : float
        Disability weight after intervention
    
    Returns
    -------
    tuple
        (qalys_gained, utility_gain_per_person)
    
    Example
    -------
    >>> qalys, utility = calculate_qalys_gained(
    ...     intervention_effect=0.95,
    ...     population=10000,
    ...     duration=15,
    ...     dw_before=0.187,  # blindness
    ...     dw_after=0.003    # corrected vision
    ... )
    """
    utility_before = 1 - dw_before
    utility_after = 1 - dw_after
    utility_gain = utility_after - utility_before
    
    people_benefiting = population * intervention_effect
    qalys = people_benefiting * utility_gain * duration
    
    return qalys, utility_gain


# =============================================================================
# POLICY SIMULATION FUNCTIONS
# =============================================================================

# Default intervention data for policy simulation
INTERVENTIONS = {
    'salt_reduction': {
        'name': 'Salt Reduction Campaign',
        'description': 'Population-wide campaign: reformulation agreements with industry, public awareness, labelling',
        'base_cost_per_daly': 1500,
        'max_capacity_millions': 15,
        'diminishing_factor': 1.8,
        'equity_distribution': [0.25, 0.23, 0.20, 0.17, 0.15],
        'time_to_impact_years': 2,
        'uncertainty_range': 0.3,
        'evidence_quality': 'High',
        'category': 'Population'
    },
    'folic_acid': {
        'name': 'Folic Acid Fortification',
        'description': 'Mandatory fortification of flour to prevent neural tube defects',
        'base_cost_per_daly': 2000,
        'max_capacity_millions': 5,
        'diminishing_factor': 1.2,
        'equity_distribution': [0.22, 0.21, 0.20, 0.19, 0.18],
        'time_to_impact_years': 1,
        'uncertainty_range': 0.2,
        'evidence_quality': 'High',
        'category': 'Population'
    },
    'weight_management': {
        'name': 'Adult Weight Management',
        'description': 'Structured programmes: behavioural support, dietary counselling, physical activity',
        'base_cost_per_daly': 8000,
        'max_capacity_millions': 25,
        'diminishing_factor': 2.5,
        'equity_distribution': [0.15, 0.18, 0.22, 0.23, 0.22],
        'time_to_impact_years': 3,
        'uncertainty_range': 0.4,
        'evidence_quality': 'Moderate',
        'category': 'Individual'
    },
    'diabetes_prevention': {
        'name': 'NHS Diabetes Prevention Programme',
        'description': 'Intensive lifestyle intervention for people with pre-diabetes',
        'base_cost_per_daly': 6000,
        'max_capacity_millions': 20,
        'diminishing_factor': 2.0,
        'equity_distribution': [0.18, 0.20, 0.21, 0.21, 0.20],
        'time_to_impact_years': 4,
        'uncertainty_range': 0.35,
        'evidence_quality': 'High',
        'category': 'Targeted'
    },
    'school_meals': {
        'name': 'School Food Standards Enhancement',
        'description': 'Improved nutritional standards, free school meal expansion, food education',
        'base_cost_per_daly': 12000,
        'max_capacity_millions': 30,
        'diminishing_factor': 1.5,
        'equity_distribution': [0.30, 0.25, 0.20, 0.15, 0.10],
        'time_to_impact_years': 15,
        'uncertainty_range': 0.5,
        'evidence_quality': 'Moderate',
        'category': 'Population'
    },
    'smoking_cessation': {
        'name': 'Smoking Cessation Services',
        'description': 'NHS stop smoking services with pharmacotherapy',
        'base_cost_per_daly': 3500,
        'max_capacity_millions': 20,
        'diminishing_factor': 2.2,
        'equity_distribution': [0.28, 0.24, 0.20, 0.16, 0.12],
        'time_to_impact_years': 5,
        'uncertainty_range': 0.25,
        'evidence_quality': 'High',
        'category': 'Individual'
    },
    'hypertension_screening': {
        'name': 'Hypertension Detection & Treatment',
        'description': 'Community screening, GP follow-up, medication adherence support',
        'base_cost_per_daly': 5000,
        'max_capacity_millions': 25,
        'diminishing_factor': 1.8,
        'equity_distribution': [0.20, 0.20, 0.20, 0.20, 0.20],
        'time_to_impact_years': 3,
        'uncertainty_range': 0.2,
        'evidence_quality': 'High',
        'category': 'Clinical'
    },
    'sdil_extension': {
        'name': 'Sugar Tax Extension',
        'description': 'Extend SDIL to confectionery and other high-sugar products',
        'base_cost_per_daly': 800,
        'max_capacity_millions': 8,
        'diminishing_factor': 1.3,
        'equity_distribution': [0.24, 0.23, 0.20, 0.18, 0.15],
        'time_to_impact_years': 3,
        'uncertainty_range': 0.45,
        'evidence_quality': 'Moderate',
        'category': 'Population'
    }
}


def calculate_dalys_averted(spend_millions, intervention):
    """
    Calculate DALYs averted for a given spend, accounting for diminishing returns.
    
    Uses numerical integration over spend increments.
    
    Parameters
    ----------
    spend_millions : float
        Budget allocated in millions of pounds
    intervention : dict
        Intervention dictionary with keys: base_cost_per_daly, 
        max_capacity_millions, diminishing_factor
    
    Returns
    -------
    float
        Total DALYs averted
    """
    if spend_millions <= 0:
        return 0
    
    # Cap at maximum capacity
    spend_millions = min(spend_millions, intervention['max_capacity_millions'])
    
    base_cost = intervention['base_cost_per_daly']
    alpha = intervention['diminishing_factor']
    
    # Integrate in £0.1M increments
    increments = np.arange(0.1, spend_millions + 0.1, 0.1)
    total_dalys = 0
    
    for x in increments:
        # Cost per DALY at this level of spend
        marginal_cost = base_cost * (x ** (alpha - 1))
        # DALYs from this £0.1M increment
        dalys_increment = (0.1 * 1_000_000) / marginal_cost
        total_dalys += dalys_increment
    
    return total_dalys


def calculate_marginal_cost_per_daly(spend_millions, intervention):
    """
    Calculate the marginal cost per DALY at a given spend level.
    
    Parameters
    ----------
    spend_millions : float
        Current spend level in millions
    intervention : dict
        Intervention dictionary
    
    Returns
    -------
    float
        Marginal cost per DALY in pounds
    """
    if spend_millions <= 0:
        return intervention['base_cost_per_daly']
    
    base_cost = intervention['base_cost_per_daly']
    alpha = intervention['diminishing_factor']
    
    return base_cost * (spend_millions ** (alpha - 1))


def calculate_equity_dalys(total_dalys, equity_distribution):
    """
    Distribute DALYs across deprivation quintiles.
    
    Parameters
    ----------
    total_dalys : float
        Total DALYs to distribute
    equity_distribution : list
        List of 5 proportions for Q1-Q5 (must sum to 1)
    
    Returns
    -------
    dict
        DALYs for each quintile {'Q1': ..., 'Q2': ..., etc}
    """
    return {
        f'Q{i+1}': total_dalys * equity_distribution[i]
        for i in range(5)
    }


def evaluate_strategy(allocations, interventions=None):
    """
    Evaluate a budget allocation strategy.
    
    Parameters
    ----------
    allocations : dict
        Dictionary mapping intervention keys to spend in millions
        Example: {'salt_reduction': 10, 'folic_acid': 5, ...}
    interventions : dict, optional
        Intervention definitions. Uses default INTERVENTIONS if not provided.
    
    Returns
    -------
    dict
        Dictionary with keys: total_dalys, total_spend, cost_per_daly,
        q1_share, equity_totals
    
    Example
    -------
    >>> allocations = {'salt_reduction': 10, 'folic_acid': 5}
    >>> results = evaluate_strategy(allocations)
    >>> print(f"DALYs averted: {results['total_dalys']:,.0f}")
    """
    if interventions is None:
        interventions = INTERVENTIONS
    
    total_dalys = 0
    equity_totals = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0, 'Q5': 0}
    total_spend = sum(allocations.values())
    
    for key, spend in allocations.items():
        if key not in interventions:
            continue
            
        intervention = interventions[key]
        dalys = calculate_dalys_averted(spend, intervention)
        total_dalys += dalys
        
        equity_dalys = calculate_equity_dalys(dalys, intervention['equity_distribution'])
        for q, val in equity_dalys.items():
            equity_totals[q] += val
    
    return {
        'total_dalys': total_dalys,
        'total_spend': total_spend,
        'cost_per_daly': (total_spend * 1_000_000) / total_dalys if total_dalys > 0 else 0,
        'q1_share': equity_totals['Q1'] / total_dalys * 100 if total_dalys > 0 else 0,
        'equity_totals': equity_totals
    }


def evaluate_strategy_with_uncertainty(allocations, n_samples=1000, 
                                       interventions=None, seed=None):
    """
    Evaluate strategy using Monte Carlo simulation to capture uncertainty.
    
    Parameters
    ----------
    allocations : dict
        Budget allocation dictionary
    n_samples : int, optional
        Number of Monte Carlo samples. Default 1000.
    interventions : dict, optional
        Intervention definitions
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        Array of total DALYs from each simulation run
    """
    if interventions is None:
        interventions = INTERVENTIONS
    
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    for _ in range(n_samples):
        total_dalys = 0
        
        for key, spend in allocations.items():
            if spend <= 0 or key not in interventions:
                continue
                
            intervention = interventions[key]
            
            # Sample from uncertainty range
            uncertainty = intervention['uncertainty_range']
            multiplier = np.random.uniform(1 - uncertainty, 1 + uncertainty)
            
            dalys = calculate_dalys_averted(spend, intervention) * multiplier
            total_dalys += dalys
        
        results.append(total_dalys)
    
    return np.array(results)


def evaluate_with_equity_weights(allocations, equity_weights, interventions=None):
    """
    Evaluate strategy with equity-weighted DALYs.
    
    Parameters
    ----------
    allocations : dict
        Budget allocation dictionary
    equity_weights : dict
        Weights for each quintile, e.g. {'Q1': 2.0, 'Q2': 1.5, ...}
    interventions : dict, optional
        Intervention definitions
    
    Returns
    -------
    float
        Total equity-weighted DALYs
    
    Example
    -------
    >>> weights = {'Q1': 2.0, 'Q2': 1.5, 'Q3': 1.0, 'Q4': 0.75, 'Q5': 0.5}
    >>> weighted_dalys = evaluate_with_equity_weights(allocations, weights)
    """
    if interventions is None:
        interventions = INTERVENTIONS
    
    total_weighted_dalys = 0
    
    for key, spend in allocations.items():
        if key not in interventions:
            continue
            
        intervention = interventions[key]
        dalys = calculate_dalys_averted(spend, intervention)
        
        equity_dalys = calculate_equity_dalys(dalys, intervention['equity_distribution'])
        
        for q, val in equity_dalys.items():
            total_weighted_dalys += val * equity_weights[q]
    
    return total_weighted_dalys


# =============================================================================
# PREDEFINED STRATEGIES FOR COMPARISON
# =============================================================================

STRATEGIES = {
    'Maximum Efficiency': {
        'description': 'Allocate purely based on cost-effectiveness, ignoring equity',
        'allocations': {
            'salt_reduction': 10,
            'folic_acid': 5,
            'sdil_extension': 8,
            'smoking_cessation': 12,
            'hypertension_screening': 15,
            'diabetes_prevention': 0,
            'weight_management': 0,
            'school_meals': 0
        }
    },
    'Equity Focus': {
        'description': 'Prioritise interventions that benefit deprived populations',
        'allocations': {
            'salt_reduction': 8,
            'folic_acid': 5,
            'sdil_extension': 5,
            'smoking_cessation': 15,
            'hypertension_screening': 0,
            'diabetes_prevention': 0,
            'weight_management': 0,
            'school_meals': 17
        }
    },
    'Balanced Portfolio': {
        'description': 'Diversified approach across intervention types',
        'allocations': {
            'salt_reduction': 8,
            'folic_acid': 5,
            'sdil_extension': 5,
            'smoking_cessation': 8,
            'hypertension_screening': 10,
            'diabetes_prevention': 7,
            'weight_management': 0,
            'school_meals': 7
        }
    }
}


# =============================================================================
# DISABILITY WEIGHT EXERCISE DATA
# =============================================================================

EXERCISE_CONDITIONS = [
    {
        'name': 'Mild anaemia',
        'description': 'Feels slightly tired and weak at times. Some difficulty with physical activities.',
        'gbd_weight': 0.004
    },
    {
        'name': 'Moderate hearing loss',
        'description': 'Has difficulty hearing conversations in noisy environments. May need to ask people to repeat themselves.',
        'gbd_weight': 0.027
    },
    {
        'name': 'Moderate depression',
        'description': 'Feels sad and has lost interest in usual activities. Has difficulty sleeping and concentrating.',
        'gbd_weight': 0.145
    },
    {
        'name': 'Severe low back pain',
        'description': 'Has severe, constant pain in the lower back. Cannot do most daily activities and has difficulty sleeping.',
        'gbd_weight': 0.325
    },
    {
        'name': 'Complete blindness',
        'description': 'Cannot see at all. Needs assistance with many daily activities.',
        'gbd_weight': 0.187
    },
    {
        'name': 'Severe dementia',
        'description': 'Cannot remember recent events or recognise close family members. Needs constant supervision.',
        'gbd_weight': 0.449
    },
    {
        'name': 'Type 2 diabetes (controlled)',
        'description': 'Must monitor diet and take daily medication. No major symptoms if well-managed.',
        'gbd_weight': 0.015
    },
    {
        'name': 'Obesity (BMI ≥ 40)',
        'description': 'Has significant excess weight causing difficulty with physical activities and daily tasks.',
        'gbd_weight': 0.086
    }
]


# =============================================================================
# HELPER FUNCTIONS FOR NOTEBOOKS
# =============================================================================

def get_intervention_summary(interventions=None):
    """
    Get a summary DataFrame of all interventions.
    
    Returns
    -------
    pd.DataFrame
        Summary table of interventions
    """
    if interventions is None:
        interventions = INTERVENTIONS
    
    summary_data = []
    for key, v in interventions.items():
        summary_data.append({
            'Key': key,
            'Intervention': v['name'],
            'Category': v['category'],
            'Base £/DALY': f"£{v['base_cost_per_daly']:,}",
            'Max Budget (£M)': v['max_capacity_millions'],
            'Time to Impact': f"{v['time_to_impact_years']} years",
            'Evidence': v['evidence_quality'],
            'Uncertainty': f"±{v['uncertainty_range']*100:.0f}%"
        })
    
    return pd.DataFrame(summary_data)


def print_strategy_comparison(strategies=None, interventions=None):
    """
    Print a comparison of predefined strategies.
    """
    if strategies is None:
        strategies = STRATEGIES
    if interventions is None:
        interventions = INTERVENTIONS
    
    print("STRATEGY COMPARISON")
    print("=" * 80)
    
    for name, strategy in strategies.items():
        metrics = evaluate_strategy(strategy['allocations'], interventions)
        print(f"\n{name.upper()}")
        print(f"  {strategy['description']}")
        print(f"  DALYs averted: {metrics['total_dalys']:,.0f}")
        print(f"  Cost per DALY: £{metrics['cost_per_daly']:,.0f}")
        print(f"  Share to most deprived (Q1): {metrics['q1_share']:.1f}%")


# =============================================================================
# HEALTH INEQUALITY FUNCTIONS
# =============================================================================

def calculate_sii(data, outcome_col, pop_share_col='population_share'):
    """
    Calculate Slope Index of Inequality.
    
    Assumes data is ordered from most to least deprived.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with outcome and population share columns
    outcome_col : str
        Name of outcome column
    pop_share_col : str
        Name of population share column
    
    Returns
    -------
    dict
        Dictionary with 'sii', 'intercept', 'r_squared', 'p_value'
    """
    from scipy import stats
    
    data = data.copy()
    data['cum_pop'] = data[pop_share_col].cumsum()
    data['cum_pop_lag'] = data['cum_pop'].shift(1, fill_value=0)
    data['ridit'] = (data['cum_pop'] + data['cum_pop_lag']) / 2
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data['ridit'], data[outcome_col]
    )
    
    return {
        'sii': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'data_with_ridit': data
    }


def calculate_rii(data, outcome_col, pop_share_col='population_share'):
    """
    Calculate Relative Index of Inequality.
    
    RII = SII / mean outcome
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with outcome and population share columns
    outcome_col : str
        Name of outcome column
    pop_share_col : str
        Name of population share column
    
    Returns
    -------
    float
        Relative Index of Inequality
    """
    sii_result = calculate_sii(data, outcome_col, pop_share_col)
    mean_outcome = data[outcome_col].mean()
    return sii_result['sii'] / mean_outcome


def calculate_concentration_index(data, outcome_col, pop_share_col='population_share'):
    """
    Calculate the Concentration Index for health inequality.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data ordered from most to least deprived
    outcome_col : str
        Name of outcome column
    pop_share_col : str
        Name of population share column
    
    Returns
    -------
    float
        Concentration Index (ranges from -1 to +1)
    """
    data = data.copy()
    
    total_outcome = (data[outcome_col] * data[pop_share_col]).sum()
    data['outcome_share'] = (data[outcome_col] * data[pop_share_col]) / total_outcome
    
    data['cum_pop'] = data[pop_share_col].cumsum()
    data['cum_outcome'] = data['outcome_share'].cumsum()
    
    cum_pop = np.concatenate([[0], data['cum_pop'].values])
    cum_outcome = np.concatenate([[0], data['cum_outcome'].values])
    
    area = np.trapz(cum_outcome, cum_pop)
    ci = 1 - 2 * area
    
    return ci


def plot_concentration_curve(data, outcome_col, pop_share_col='population_share', ax=None):
    """
    Plot a concentration curve for a health outcome.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data ordered from most to least deprived
    outcome_col : str
        Name of outcome column
    pop_share_col : str
        Name of population share column
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    float
        Concentration Index
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    data = data.copy()
    
    total_outcome = (data[outcome_col] * data[pop_share_col]).sum()
    data['outcome_share'] = (data[outcome_col] * data[pop_share_col]) / total_outcome
    
    data['cum_pop'] = data[pop_share_col].cumsum()
    data['cum_outcome'] = data['outcome_share'].cumsum()
    
    cum_pop = np.concatenate([[0], data['cum_pop'].values])
    cum_outcome = np.concatenate([[0], data['cum_outcome'].values])
    
    ax.plot([0, 1], [0, 1], 'k--', label='Line of equality')
    ax.plot(cum_pop, cum_outcome, 'b-', linewidth=2, marker='o', label='Concentration curve')
    ax.fill_between(cum_pop, cum_pop, cum_outcome, alpha=0.3)
    
    ax.set_xlabel('Cumulative % of population (most to least deprived)')
    ax.set_ylabel('Cumulative % of health outcome')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    area = np.trapz(cum_outcome, cum_pop)
    ci = 1 - 2 * area
    
    return ci


# Example data for inequality exercises
INEQUALITY_EXAMPLE_DATA = {
    'life_expectancy': pd.DataFrame({
        'quintile': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'population_share': [0.20, 0.20, 0.20, 0.20, 0.20],
        'life_expectancy_male': [74.0, 77.2, 79.1, 80.5, 83.2],
        'life_expectancy_female': [78.8, 81.2, 82.8, 84.0, 86.3]
    }),
    'obesity': pd.DataFrame({
        'quintile': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        'population_share': [0.20, 0.20, 0.20, 0.20, 0.20],
        'obesity_prevalence': [0.34, 0.30, 0.27, 0.24, 0.20]
    })
}


# =============================================================================
# SALT/SUGAR HEALTH IMPACT FUNCTIONS
# =============================================================================

def estimate_bp_change_from_sodium(salt_reduction_g, effect_per_g=1.2):
    """
    Estimate systolic blood pressure change from salt reduction.
    
    Parameters
    ----------
    salt_reduction_g : float
        Reduction in salt intake (g/day)
    effect_per_g : float
        mmHg SBP reduction per g salt reduction (default 1.2)
    
    Returns
    -------
    float
        Estimated SBP reduction (mmHg)
    """
    return salt_reduction_g * effect_per_g


def estimate_cvd_risk_change(sbp_reduction_mmhg, outcome='stroke'):
    """
    Estimate relative risk reduction for CVD outcomes from BP reduction.
    
    Parameters
    ----------
    sbp_reduction_mmhg : float
        Reduction in systolic blood pressure (mmHg)
    outcome : str
        'stroke' or 'chd'
    
    Returns
    -------
    float
        Relative risk reduction (as proportion)
    """
    rr_per_2mmhg = {'stroke': 0.10, 'chd': 0.07}
    return (sbp_reduction_mmhg / 2) * rr_per_2mmhg.get(outcome, 0.07)


def calculate_deaths_averted(population, baseline_rate_per_100k, 
                             risk_reduction, years=1):
    """
    Calculate deaths averted from a risk reduction.
    
    Parameters
    ----------
    population : int
        Population size
    baseline_rate_per_100k : float
        Baseline mortality rate per 100,000 per year
    risk_reduction : float
        Relative risk reduction (as proportion)
    years : int
        Number of years
    
    Returns
    -------
    float
        Estimated deaths averted
    """
    baseline_deaths = (population / 100_000) * baseline_rate_per_100k * years
    return baseline_deaths * risk_reduction


# =============================================================================
# HEALTH INEQUALITIES FUNCTIONS (WEEK 3)
# =============================================================================

def calculate_sii(outcome_by_quintile, population_by_quintile=None):
    """
    Calculate the Slope Index of Inequality (SII).
    
    The SII represents the absolute difference in health between the 
    most and least deprived, accounting for the entire distribution.
    
    Parameters
    ----------
    outcome_by_quintile : list or array
        Health outcome values for quintiles 1-5 (Q1=most deprived)
    population_by_quintile : list or array, optional
        Population in each quintile. If None, assumes equal populations.
    
    Returns
    -------
    dict
        Dictionary with 'sii', 'intercept', 'r_squared', and 'midpoints'
    
    Example
    -------
    >>> mortality = [450, 380, 320, 280, 220]  # per 100,000
    >>> result = calculate_sii(mortality)
    >>> print(f"SII: {result['sii']:.1f} per 100,000")
    """
    outcome = np.array(outcome_by_quintile)
    n_groups = len(outcome)
    
    if population_by_quintile is None:
        population_by_quintile = np.ones(n_groups) / n_groups
    else:
        population_by_quintile = np.array(population_by_quintile)
        population_by_quintile = population_by_quintile / population_by_quintile.sum()
    
    # Calculate cumulative population midpoints (ridit scores)
    cumsum = np.cumsum(population_by_quintile)
    midpoints = cumsum - population_by_quintile / 2
    
    # Weighted linear regression
    weights = population_by_quintile
    x_mean = np.average(midpoints, weights=weights)
    y_mean = np.average(outcome, weights=weights)
    
    numerator = np.sum(weights * (midpoints - x_mean) * (outcome - y_mean))
    denominator = np.sum(weights * (midpoints - x_mean) ** 2)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # R-squared
    y_pred = intercept + slope * midpoints
    ss_res = np.sum(weights * (outcome - y_pred) ** 2)
    ss_tot = np.sum(weights * (outcome - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'sii': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'midpoints': midpoints
    }


def calculate_rii(outcome_by_quintile, population_by_quintile=None):
    """
    Calculate the Relative Index of Inequality (RII).
    
    The RII is the ratio of the predicted outcome at the most deprived end
    to the predicted outcome at the least deprived end.
    
    Parameters
    ----------
    outcome_by_quintile : list or array
        Health outcome values for quintiles 1-5 (Q1=most deprived)
    population_by_quintile : list or array, optional
        Population in each quintile.
    
    Returns
    -------
    dict
        Dictionary with 'rii', 'sii', and regression details
    """
    sii_result = calculate_sii(outcome_by_quintile, population_by_quintile)
    
    # Predicted values at extremes (0 and 1 on the ridit scale)
    y_at_0 = sii_result['intercept']  # Most deprived end
    y_at_1 = sii_result['intercept'] + sii_result['sii']  # Least deprived end
    
    # RII is ratio of extremes
    rii = y_at_0 / y_at_1 if y_at_1 != 0 else np.inf
    
    return {
        'rii': rii,
        'sii': sii_result['sii'],
        'intercept': sii_result['intercept'],
        'r_squared': sii_result['r_squared'],
        'predicted_most_deprived': y_at_0,
        'predicted_least_deprived': y_at_1
    }


def calculate_concentration_index(outcome_by_quintile, population_by_quintile=None):
    """
    Calculate the Concentration Index for health inequality.
    
    The concentration index ranges from -1 to +1:
    - Negative values: outcome concentrated among the deprived
    - Zero: no inequality
    - Positive values: outcome concentrated among the affluent
    
    Parameters
    ----------
    outcome_by_quintile : list or array
        Health outcome for each quintile (Q1=most deprived)
    population_by_quintile : list or array, optional
        Population weights
    
    Returns
    -------
    float
        Concentration index
    """
    outcome = np.array(outcome_by_quintile)
    n = len(outcome)
    
    if population_by_quintile is None:
        pop = np.ones(n) / n
    else:
        pop = np.array(population_by_quintile)
        pop = pop / pop.sum()
    
    # Fractional rank (ridit)
    cumsum = np.cumsum(pop)
    rank = cumsum - pop / 2
    
    # Mean outcome
    mu = np.sum(pop * outcome)
    
    # Concentration index
    ci = (2 / mu) * np.sum(pop * outcome * rank) - 1
    
    return ci


def calculate_rate_ratio(rate_deprived, rate_affluent):
    """
    Calculate simple rate ratio between most and least deprived.
    
    Parameters
    ----------
    rate_deprived : float
        Rate in most deprived group
    rate_affluent : float
        Rate in least deprived group
    
    Returns
    -------
    float
        Rate ratio (relative inequality)
    """
    return rate_deprived / rate_affluent if rate_affluent != 0 else np.inf


def calculate_rate_difference(rate_deprived, rate_affluent):
    """
    Calculate simple rate difference between most and least deprived.
    
    Parameters
    ----------
    rate_deprived : float
        Rate in most deprived group
    rate_affluent : float
        Rate in least deprived group
    
    Returns
    -------
    float
        Rate difference (absolute inequality)
    """
    return rate_deprived - rate_affluent


def calculate_par(outcome_by_quintile, population_by_quintile=None, reference='best'):
    """
    Calculate Population Attributable Risk.
    
    How much of the total burden could be eliminated if all groups
    had the same rate as the reference group?
    
    Parameters
    ----------
    outcome_by_quintile : list or array
        Outcome rates by quintile
    population_by_quintile : list or array, optional
        Population in each quintile
    reference : str or int
        'best' (lowest rate), 'worst' (highest), or quintile index (0-4)
    
    Returns
    -------
    dict
        Dictionary with 'par', 'par_percent', 'reference_rate', 'overall_rate'
    """
    outcome = np.array(outcome_by_quintile)
    
    if population_by_quintile is None:
        pop = np.ones(len(outcome)) / len(outcome)
    else:
        pop = np.array(population_by_quintile)
        pop = pop / pop.sum()
    
    overall_rate = np.sum(pop * outcome)
    
    if reference == 'best':
        ref_rate = np.min(outcome)
    elif reference == 'worst':
        ref_rate = np.max(outcome)
    else:
        ref_rate = outcome[reference]
    
    par = overall_rate - ref_rate
    par_percent = (par / overall_rate) * 100 if overall_rate > 0 else 0
    
    return {
        'par': par,
        'par_percent': par_percent,
        'reference_rate': ref_rate,
        'overall_rate': overall_rate
    }


# Example data for health inequalities exercises
INEQUALITY_EXAMPLE_DATA = {
    'cvd_mortality': {
        'description': 'CVD mortality rate per 100,000 (age-standardised)',
        'quintiles': [285, 245, 210, 175, 140],
        'year': 2019
    },
    'obesity_prevalence': {
        'description': 'Adult obesity prevalence (%)',
        'quintiles': [34.2, 30.1, 26.5, 22.8, 18.5],
        'year': 2019
    },
    'fruit_veg_consumption': {
        'description': 'Adults meeting 5-a-day (%)',
        'quintiles': [42, 48, 54, 61, 72],
        'year': 2019
    },
    'diabetes_prevalence': {
        'description': 'Type 2 diabetes prevalence (%)',
        'quintiles': [9.2, 7.8, 6.5, 5.4, 4.2],
        'year': 2019
    },
    'life_expectancy_male': {
        'description': 'Male life expectancy at birth (years)',
        'quintiles': [74.1, 76.8, 78.9, 80.5, 83.2],
        'year': 2019
    },
    'life_expectancy_female': {
        'description': 'Female life expectancy at birth (years)',
        'quintiles': [78.6, 80.8, 82.5, 83.9, 86.1],
        'year': 2019
    }
}


# =============================================================================
# SALT REDUCTION MODELLING FUNCTIONS (WEEK 4)
# =============================================================================

def estimate_bp_reduction_from_salt(salt_reduction_g, age_group='adult'):
    """
    Estimate blood pressure reduction from salt intake reduction.
    
    Based on meta-analyses of salt reduction trials.
    
    Parameters
    ----------
    salt_reduction_g : float
        Reduction in daily salt intake (grams)
    age_group : str
        'adult' or 'elderly' (>60 years)
    
    Returns
    -------
    dict
        Estimated systolic and diastolic BP reductions (mmHg)
    """
    # Approximate effect sizes from He & MacGregor meta-analyses
    if age_group == 'elderly':
        sbp_per_gram = 1.2  # Greater effect in elderly
        dbp_per_gram = 0.6
    else:
        sbp_per_gram = 0.9
        dbp_per_gram = 0.45
    
    return {
        'systolic_reduction': salt_reduction_g * sbp_per_gram,
        'diastolic_reduction': salt_reduction_g * dbp_per_gram
    }


def estimate_cvd_reduction_from_bp(sbp_reduction, population_size, 
                                    baseline_cvd_rate=0.01, age_group='adult'):
    """
    Estimate CVD events prevented from blood pressure reduction.
    
    Uses risk reduction estimates from meta-analyses.
    
    Parameters
    ----------
    sbp_reduction : float
        Systolic blood pressure reduction (mmHg)
    population_size : int
        Size of population
    baseline_cvd_rate : float
        Annual CVD event rate (default 1%)
    age_group : str
        'adult' or 'elderly'
    
    Returns
    -------
    dict
        Estimated CVD events prevented, strokes prevented, CHD prevented
    """
    # Risk reduction per mmHg SBP (from Law et al., BMJ 2009)
    # ~2% reduction in CHD, ~3% reduction in stroke per mmHg
    chd_rr_per_mmhg = 0.02
    stroke_rr_per_mmhg = 0.03
    
    # Baseline split: assume 40% CHD, 30% stroke, 30% other CVD
    baseline_chd = baseline_cvd_rate * 0.4
    baseline_stroke = baseline_cvd_rate * 0.3
    
    chd_prevented = population_size * baseline_chd * (1 - (1 - chd_rr_per_mmhg) ** sbp_reduction)
    stroke_prevented = population_size * baseline_stroke * (1 - (1 - stroke_rr_per_mmhg) ** sbp_reduction)
    
    total_prevented = chd_prevented + stroke_prevented
    
    return {
        'total_cvd_prevented': total_prevented,
        'chd_prevented': chd_prevented,
        'stroke_prevented': stroke_prevented,
        'relative_risk_reduction': 1 - ((1 - chd_rr_per_mmhg) ** sbp_reduction)
    }


def model_salt_reduction_impact(salt_change_g, population_size, years=10,
                                 baseline_salt_g=8.0, target_salt_g=6.0):
    """
    Model the health impact of a population salt reduction programme.
    
    Parameters
    ----------
    salt_change_g : float
        Achieved reduction in daily salt intake (grams)
    population_size : int
        Adult population size
    years : int
        Time horizon for modelling
    baseline_salt_g : float
        Baseline average salt intake
    target_salt_g : float
        Target salt intake
    
    Returns
    -------
    dict
        Comprehensive results including CVD events prevented, deaths averted,
        DALYs averted, and cost savings
    """
    # BP reduction
    bp_effect = estimate_bp_reduction_from_salt(salt_change_g)
    
    # CVD prevention (annual)
    cvd_effect = estimate_cvd_reduction_from_bp(
        bp_effect['systolic_reduction'], 
        population_size
    )
    
    # Scale to time horizon
    cvd_events_prevented = cvd_effect['total_cvd_prevented'] * years
    
    # Estimate deaths (assume 20% case fatality for CVD events)
    deaths_averted = cvd_events_prevented * 0.20
    
    # DALYs (rough estimate: 10 YLL per death, 2 YLD per non-fatal event)
    dalys_averted = deaths_averted * 10 + (cvd_events_prevented - deaths_averted) * 2
    
    # Cost savings (NHS costs per CVD event ~£5,000)
    cost_savings = cvd_events_prevented * 5000
    
    return {
        'salt_reduction_g': salt_change_g,
        'sbp_reduction_mmhg': bp_effect['systolic_reduction'],
        'cvd_events_prevented': cvd_events_prevented,
        'deaths_averted': deaths_averted,
        'dalys_averted': dalys_averted,
        'cost_savings_gbp': cost_savings,
        'years': years,
        'population': population_size
    }


# UK Salt Reduction Programme data
UK_SALT_DATA = {
    'urinary_sodium': {
        'years': [2006, 2008, 2011, 2014, 2019],
        'salt_g_per_day': [9.5, 9.0, 8.1, 8.0, 8.4],
        'sample_sizes': [1658, 1714, 547, 689, 648],
        'notes': 'National Diet and Nutrition Survey 24h urine collection'
    },
    'food_categories': {
        'bread': {'2006': 1.23, '2011': 0.98, 'change_pct': -20},
        'breakfast_cereals': {'2006': 1.02, '2011': 0.78, 'change_pct': -24},
        'biscuits': {'2006': 0.68, '2011': 0.53, 'change_pct': -22},
        'ready_meals': {'2006': 1.45, '2011': 1.18, 'change_pct': -19},
        'crisps': {'2006': 1.5, '2011': 1.3, 'change_pct': -13}
    },
    'targets': {
        '2006_target': 6.0,
        '2015_target': 6.0,
        '2024_target': 6.0,
        'who_target': 5.0
    }
}


# =============================================================================
# SUGAR REDUCTION / SDIL FUNCTIONS (WEEK 5)
# =============================================================================

def estimate_calorie_reduction_from_sugar(sugar_reduction_g):
    """
    Estimate calorie reduction from reduced sugar intake.
    
    Parameters
    ----------
    sugar_reduction_g : float
        Reduction in daily free sugar intake (grams)
    
    Returns
    -------
    float
        Calorie reduction (kcal)
    """
    return sugar_reduction_g * 4  # 4 kcal per gram of sugar


def estimate_weight_change_from_calories(calorie_change, duration_weeks):
    """
    Estimate weight change from sustained calorie change.
    
    Uses the simplified "3500 kcal = 1 lb" rule with adaptation.
    
    Parameters
    ----------
    calorie_change : float
        Daily calorie change (negative = deficit)
    duration_weeks : int
        Duration of change
    
    Returns
    -------
    dict
        Estimated weight change and metabolic adaptation
    """
    # Hall et al. model suggests ~7700 kcal per kg initially
    # But with metabolic adaptation, actual change is less
    
    total_deficit = calorie_change * duration_weeks * 7
    
    # Initial weight change (first 3 months)
    if duration_weeks <= 12:
        weight_change_kg = total_deficit / 7700
        adaptation_factor = 1.0
    else:
        # Metabolic adaptation reduces effect over time
        initial_change = (calorie_change * 12 * 7) / 7700
        remaining_weeks = duration_weeks - 12
        adapted_change = (calorie_change * remaining_weeks * 7) / 9500  # Less efficient
        weight_change_kg = initial_change + adapted_change
        adaptation_factor = 0.8
    
    return {
        'weight_change_kg': weight_change_kg,
        'adaptation_factor': adaptation_factor,
        'total_calorie_deficit': total_deficit
    }


def model_sdil_impact(price_increase_pct, baseline_consumption_ml, 
                      price_elasticity=-0.8, population_size=1000000):
    """
    Model the impact of a soft drinks levy on consumption and health.
    
    Parameters
    ----------
    price_increase_pct : float
        Price increase from levy (%)
    baseline_consumption_ml : float
        Baseline daily consumption (ml)
    price_elasticity : float
        Price elasticity of demand (default -0.8)
    population_size : int
        Population size
    
    Returns
    -------
    dict
        Estimated consumption change, calorie reduction, and health impact
    """
    # Consumption change
    consumption_change_pct = price_elasticity * price_increase_pct
    new_consumption = baseline_consumption_ml * (1 + consumption_change_pct / 100)
    ml_reduction = baseline_consumption_ml - new_consumption
    
    # Sugar reduction (assume 10.6g per 100ml for full-sugar drinks)
    sugar_reduction_g = (ml_reduction / 100) * 10.6
    
    # Calorie reduction
    calorie_reduction = estimate_calorie_reduction_from_sugar(sugar_reduction_g)
    
    # Weight change over 1 year
    weight_effect = estimate_weight_change_from_calories(-calorie_reduction, 52)
    
    # Population impact
    total_sugar_reduction = sugar_reduction_g * population_size * 365 / 1000  # kg
    
    return {
        'consumption_change_pct': consumption_change_pct,
        'ml_reduction_per_day': ml_reduction,
        'sugar_reduction_g_per_day': sugar_reduction_g,
        'calorie_reduction_per_day': calorie_reduction,
        'weight_change_kg_per_year': weight_effect['weight_change_kg'],
        'population_sugar_reduction_tonnes': total_sugar_reduction / 1000,
        'population_size': population_size
    }


def estimate_reformulation_impact(products_reformulated_pct, 
                                   avg_sugar_reduction_pct,
                                   market_share_pct=100):
    """
    Estimate impact of product reformulation on sugar intake.
    
    Parameters
    ----------
    products_reformulated_pct : float
        Percentage of products reformulated
    avg_sugar_reduction_pct : float
        Average sugar reduction in reformulated products
    market_share_pct : float
        Market share of reformulated products
    
    Returns
    -------
    dict
        Estimated overall sugar reduction
    """
    # Effective reduction
    effective_reduction = (products_reformulated_pct / 100) * \
                         (avg_sugar_reduction_pct / 100) * \
                         (market_share_pct / 100)
    
    return {
        'effective_reduction_pct': effective_reduction * 100,
        'products_reformulated_pct': products_reformulated_pct,
        'avg_sugar_reduction_pct': avg_sugar_reduction_pct,
        'market_share_pct': market_share_pct
    }


# UK SDIL and sugar reduction data
UK_SUGAR_DATA = {
    'sdil_timeline': {
        '2016_03': 'SDIL announced',
        '2018_04': 'SDIL implemented',
        '2018_rates': {'high_tier': 0.24, 'low_tier': 0.18},  # per litre
        'thresholds': {'high': 8, 'low': 5}  # g sugar per 100ml
    },
    'reformulation': {
        'drinks_above_5g': {'2015': 49, '2018': 15, '2019': 11},  # % of products
        'avg_sugar_per_100ml': {'2015': 4.4, '2019': 2.9},
        'levy_liable_volume_pct': {'2015': 49, '2019': 15}
    },
    'purchase_data': {
        'ssb_ml_per_hh_per_week': {
            '2014': 1892,
            '2015': 1778,
            '2016': 1666,
            '2017': 1480,
            '2018': 1295,
            '2019': 1180
        },
        'source': 'Kantar Worldpanel'
    },
    'limitations': [
        'No biomarker for sugar intake',
        'Purchase data ≠ consumption',
        'Secular trends may confound',
        'Reformulation vs switching unclear',
        'Health outcomes take years to manifest'
    ]
}


# =============================================================================
# GENERAL EVALUATION FRAMEWORK
# =============================================================================

def evaluate_intervention_evidence(intervention_name, has_biomarker=True,
                                    rct_evidence=True, time_series_data=True,
                                    control_group=False, long_term_followup=False):
    """
    Score the strength of evidence for an intervention.
    
    Parameters
    ----------
    intervention_name : str
        Name of the intervention
    has_biomarker : bool
        Whether there's an objective biomarker
    rct_evidence : bool
        Whether RCT evidence exists
    time_series_data : bool
        Whether interrupted time series data available
    control_group : bool
        Whether there's a valid control/comparison group
    long_term_followup : bool
        Whether long-term health outcomes measured
    
    Returns
    -------
    dict
        Evidence quality assessment
    """
    score = 0
    strengths = []
    limitations = []
    
    if has_biomarker:
        score += 2
        strengths.append("Objective biomarker available")
    else:
        limitations.append("No objective biomarker - relies on self-report or purchase data")
    
    if rct_evidence:
        score += 2
        strengths.append("RCT evidence for mechanism")
    else:
        limitations.append("Limited experimental evidence")
    
    if time_series_data:
        score += 1
        strengths.append("Time series data available")
    
    if control_group:
        score += 2
        strengths.append("Valid control group for comparison")
    else:
        limitations.append("No control group - secular trends may confound")
    
    if long_term_followup:
        score += 2
        strengths.append("Long-term health outcomes measured")
    else:
        limitations.append("Health outcomes modelled, not directly observed")
    
    # Quality rating
    if score >= 7:
        quality = "High"
    elif score >= 4:
        quality = "Moderate"
    else:
        quality = "Low"
    
    return {
        'intervention': intervention_name,
        'score': score,
        'max_score': 9,
        'quality': quality,
        'strengths': strengths,
        'limitations': limitations
    }

def fit_sii_rii(group: pd.DataFrame) -> dict:
    """
    Fit:
    - SII via weighted linear regression of obesity_prev on ridit
    - RII via weighted linear regression of log(obesity_prev) on ridit

    Weighting uses population_share (derived from weighted bases).
    """
    x = group["ridit"].to_numpy()
    w = group["population_share"].to_numpy()

    y = group["obesity_prev"].to_numpy()
    sii_slope, sii_intercept = np.polyfit(x, y, 1, w=w)

    # Log model for RII (guard against zeros)
    y_log = np.log(y)
    rii_slope, rii_intercept = np.polyfit(x, y_log, 1, w=w)
    rii = float(np.exp(rii_slope))

    return {
        "sii": float(sii_slope),
        "sii_intercept": float(sii_intercept),
        "rii": rii,
        "rii_slope": float(rii_slope),
        "rii_intercept": float(rii_intercept),
        "data": group.copy()
    }