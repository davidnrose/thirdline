import pandas as pd
from src.beta import run_beta


# read in the mock data
data = pd.read_csv("data/ot_sample_results.txt", sep=",", header=0)

# function to summarise the data
def summarise_data(data, x_col, n_col): 
    x, n = data[x_col].sum(), data[n_col].sum()
    ot_summary = [x, n]
    print(ot_summary)
    return ot_summary

ot_summary = summarise_data(data, "unfair", "sample_size")

# set prior, Jeffrey's prior
prior = [0.5, 0.5]

# implement 
result = run_beta(ot_summary, prior, ci_level=0.9, threshold=0.05, n_sims=2_000_000)
print(result)