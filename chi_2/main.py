import pandas as pd

from chi_2 import chi2_permutation, cramers_v


# read in the example data
admissions = pd.read_csv("hospital_readmissions_3000.csv")

# inspect the data
print(len(admissions))
print(admissions.head())

# perform permutation chi2 test of independence and return an answer
chi2, p_value, dof, expected, null_distribution, observed = chi2_permutation(admissions, "Post_Discharge_Care_Flag", "Readmitted_30Days", n_perm=100)


# return the effect size
cv = cramers_v(observed, )