import pandas as pd
import numpy as np
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
    working = df[group_col, outcome_col].dropne().copy()

    # create contingency table
    observed = pd.crosstab(working[outcome_col], working[outcome_col])

    # return initial chi2 value, degrees of freedom, and expected counts
    chi2, _, dof, expected = chi2_contingency(observed, correction=False) 

    # initialise values for shuffling/permutation
    rng = np.random.default_rng(random_state)
    groups = working(group_col).to_numpy()
    outcomes = working(outcome_col).to_numpy()
    group_levels = observed.index
    outcome_levels = observed.columns
    null_distribution = np.empty(n_perm)

    # sub-function to implement the shuffling
    def chi2_shuffle(outcomes, groups)