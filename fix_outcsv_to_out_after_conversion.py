import os

TARGET = "/home/ygang/Documents/anomaly/MSAD/data/TSB/data/NAB_KAGGLE"

print("=== Fixing .out.csv → .out ===")
count = 0

for root, dirs, files in os.walk(TARGET):

    for f in files:
        if f.endswith(".out.csv"):
            old = os.path.join(root, f)
            new = os.path.join(root, f.replace(".out.csv", ".out"))
            os.rename(old, new)
            print(f"[RENAME] {old} → {new}")
            count += 1

print(f"=== DONE. Fixed {count} files ===")