import pandas as pd

df = pd.read_csv("data/TSB/metrics/final_labeled_dataset_converted.csv")

df["key"] = df["dataset"].astype(str) + "/" + df["filename"].astype(str)

dup = df[df["key"].duplicated(keep=False)]

print("중복 개수:", len(dup))
print(dup)