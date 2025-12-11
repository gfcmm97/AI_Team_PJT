import os

NAB_PATH = "/home/ygang/Documents/anomaly/MSAD/data/TSB/data/NAB_KAGGLE"

def main():
    print("=== Adding .out.csv suffix to NAB_KAGGLE files ===")
    for fname in os.listdir(NAB_PATH):
        if not fname.endswith(".csv"):
            continue
        if fname.endswith(".out.csv"):
            # 이미 처리된 파일
            continue

        old_path = os.path.join(NAB_PATH, fname)
        new_name = fname.replace(".csv", ".out.csv")
        new_path = os.path.join(NAB_PATH, new_name)

        print(f"Renaming: {fname} → {new_name}")
        os.rename(old_path, new_path)

    print("=== DONE ===")

if __name__ == "__main__":
    main()