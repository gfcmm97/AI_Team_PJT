import os
import pandas as pd

base = "data/TSB/data/NAB_KAGGLE"

for fname in os.listdir(base):
    if fname.endswith(".out"):
        path = os.path.join(base, fname)
        try:
            df = pd.read_csv(path, header=None)
            if df.shape[1] != 2:
                print(f"[REMOVE] {fname} (cols={df.shape[1]})")
                os.remove(path)
        except Exception as e:
            print(f"[ERROR REMOVE] {fname}: {e}")
            os.remove(path)

print("=== Cleanup done ===")