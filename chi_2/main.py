import pandas as pd
import numpy as np

from chi_2 import chi2_permutation, cramers_v, effect_size


# read in the example data
admissions = pd.read_csv("hospital_readmissions_3000.csv")

admissions["Care_Service_Type"] = np.where(admissions["Care_Service_Type"].isna(), "None", admissions["Care_Service_Type"])

# inspect the data
print(len(admissions))
print(admissions.head())


# set alpha - significance level
alpha = 0.01

# perform permutation chi2 test of independence and return an answer
chi2, p_value, dof, expected, null_distribution, observed = chi2_permutation(admissions, "Post_Discharge_Care_Flag", "Readmitted_30Days", n_perm=100, alpha=alpha)


# return the effect size
cv = cramers_v(observed, chi2)
print(round(cv, 4))

# compare the effect size against adjusted threshold
print(effect_size(cv=cv, dof=dof))


# compare post-discarhge care types against patients without post discharge care


# iterate through the post-care types and perform the permutation comparison
cols = ["Care_Type", "Chi2", "P_Value", "DoF", "Cramers_V", "Effect_Size"]
res = pd.DataFrame(columns=cols)
for care_type in admissions["Care_Service_Type"].unique():

    if care_type != 'None':

        print(f"Implementing analysis for {care_type}")
        df = admissions.query("Care_Service_Type == @care_type or Care_Service_Type == 'None'").copy()

        # perform the test
        chi2, p_value, dof, expected, null_distribution, observed = chi2_permutation(df, 
                                                                                     "Post_Discharge_Care_Flag", 
                                                                                     "Readmitted_30Days", 
                                                                                     n_perm=100, 
                                                                                     alpha=alpha,
                                                                                    verbose=False)

        # return cramer's v and effect size
        cv = cramers_v(observed, chi2)
        es = effect_size(cv, dof)

        # create the row to insert into the results 
        row = [care_type, chi2, p_value, dof, cv, es]
        res.loc[len(res)] = row

    else: 
        pass

# save down the result
res.to_csv("result.csv", index=False)