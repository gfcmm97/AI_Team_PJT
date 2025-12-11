import os
import pandas as pd

ROOT = "/home/ygang/Documents/anomaly/MSAD/data"
WINDOW_SIZES = [32, 64, 128, 256, 512, 768, 1024, 1536]

def check_folder(path):
    print(f"\n=== Checking {path} ===")
    if not os.path.exists(path):
        print("❌ Folder does NOT exist")
        return False
    print("✔ Folder exists")
    return True

def check_subdatasets(window_path):
    subfolders = [f for f in os.listdir(window_path) if os.path.isdir(os.path.join(window_path, f))]
    print("→ Found subdatasets:", subfolders)
    
    required = ["Daphnet", "PTBXL", "NAB_KAGGLE"]
    ok = True
    for r in required:
        if r not in subfolders:
            print(f"❌ Missing subdataset folder: {r}")
            ok = False
        else:
            print(f"✔ {r} exists")
    return ok

def check_sample_csv(window_path):
    # 찾기: 하나의 dataset에서 임의의 CSV 파일
    for ds in os.listdir(window_path):
        ds_path = os.path.join(window_path, ds)
        if not os.path.isdir(ds_path):
            continue

        csv_list = [f for f in os.listdir(ds_path) if f.endswith(".csv")]
        if len(csv_list) == 0:
            print(f"❌ No CSV found in {ds}")
            continue

        sample_csv = os.path.join(ds_path, csv_list[0])
        print(f"→ Sample CSV: {sample_csv}")

        try:
            df = pd.read_csv(sample_csv)
        except Exception as e:
            print("❌ Failed to read CSV:", e)
            return False

        # label column 체크
        if "label" not in df.columns:
            print("❌ Missing 'label' column")
            return False
        print("✔ label column exists")

        # val_x column 체크
        val_cols = [c for c in df.columns if c.startswith("val_")]
        if len(val_cols) == 0:
            print("❌ No val_x columns found")
            return False
        print(f"✔ val_x columns OK: {len(val_cols)} cols")

        return True

    print("❌ No datasets found inside window folder")
    return False

def main():
    for w in WINDOW_SIZES:
        path = os.path.join(ROOT, f"TSB_{w}")
        print("\n====================================================")
        print(f"Checking window size: {w}")
        print("====================================================")

        if not check_folder(path):
            continue

        check_subdatasets(path)
        check_sample_csv(path)

if __name__ == "__main__":
    main()