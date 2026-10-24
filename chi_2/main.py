import pandas as pd


# read in the example data
admissions = pd.read_csv("hospital_readmissions_3000.csv")

print(len(admissions))
print(admissions.head())