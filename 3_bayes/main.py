import pandas as pd


# read in the mock data
data = pd.read_csv("data/ot_sample_results.txt", sep=",", header=0)

print(data)