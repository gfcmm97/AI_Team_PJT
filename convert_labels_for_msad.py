import pandas as pd

input_path = "/home/ygang/Documents/anomaly/MSAD/data/TSB/metrics/final_labeled_dataset.csv"
output_path = "/home/ygang/Documents/anomaly/MSAD/data/TSB/metrics/final_labeled_dataset_converted.csv"

df = pd.read_csv(input_path)

def convert_filename(fn):
    # 예: S01R02E0.test.csv@4.out → S01R02E0.csv
    if ".test" in fn:
        base = fn.split(".test")[0]
        return base + ".csv"
    else:
        return fn   # 혹시 다른 타입 있으면 그대로

df["filename"] = df["filename"].apply(convert_filename)

df.to_csv(output_path, index=False)
print("변환 완료:", output_path)