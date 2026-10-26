import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2_contingency


# function to implement a permutation chi2 test - as more robust with lower counts
def chi2_permutation(
        df, 
        group_col, 
        outcome_col,
        alpha=0.01,
        n_perm=5000,
        random_state=2000,
        verbose=True,
        use_tqdm=True
): 

    # create working copy of the dataset
    working = df[[group_col, outcome_col]].dropna().copy()

    # create contingency table
    observed = pd.crosstab(working[group_col], working[outcome_col])

    # return initial chi2 value, degrees of freedom, and expected counts
    chi2, _, dof, expected = chi2_contingency(observed, correction=False) 

    # initialise values for shuffling/permutation
    rng = np.random.default_rng(random_state)
    groups = working[group_col].to_numpy()
    outcomes = working[outcome_col].to_numpy()
    group_levels = observed.index
    outcome_levels = observed.columns
    null_distribution = np.empty(n_perm)

    # sub-function to implement the shuffling
    def chi2_shuffle(outcomes, groups=groups, outcome_level=outcome_levels, group_levels=group_levels):

        # shuffle the outcome
        shuffled_outcomes = rng.permutation(outcomes)

        # get contingency table of the shuffled outcome
        perm_table = pd.crosstab(groups, shuffled_outcomes).reindex(
            index=group_levels,
            columns=outcome_levels,
            fill_value=0
        )

        # return the p value
        p_value = chi2_contingency(perm_table, correction=False)[0]

        return p_value

    # perform all permutations
    if use_tqdm: 
        for i in tqdm(range(n_perm), desc="Processing {n_perm:,} permutations..."):
                null_distribution[i] = chi2_shuffle(outcomes)

    else: 
        for i in range(n_perm):
            null_distribution[i] = chi2_shuffle(outcomes)  

    p_value = (1 + np.sum(null_distribution >= chi2)) / (n_perm + 1)

    if verbose: 
        print(f"Calculated chi2: {chi2}")
        print(f"Calculated P value: {p_value}")
        if p_value < alpha:
            print(f"Reject null hypothesis at {alpha} confidence level!")
        else: 
            print(f"Cannot reject null hypothesis at {alpha} confidence level!")

    return chi2, p_value, dof, expected, null_distribution, observed



# function to calculate the effect size. using Cramer's V
def cramers_v(contingency, chi2): 

    table = np.array(contingency)
    n = table.sum()
    rows, columns = table.shape

    dof = min(rows -1, columns - 1)    

    denom = n * dof

    # avoid divide by zero
    if denom == 0:
        return np.nan

    return np.sqrt(chi2 / denom)



# function to return the effect size as table, based on the dof
def effect_size(cv, dof): 

    # set the standard values, per Cohen
    thresholds = np.asarray([0.1, 0.3, 0.5])

    # adjust the thresholds with the degrees of freedom
    thresholds_adjusted = thresholds / np.sqrt(dof)

    # unpack the values
    minor, moderate, large = thresholds_adjusted

    # compare the cv value and get labelled result
    if cv < minor: 
        es = "minor"
    elif cv < moderate: 
        es = "minor to moderate"
    elif cv < large:
        es = "moderate to large"
    else: 
        es = "large"

    return es, thresholds_adjusted
