import os
import re

NAB_PATH = "/home/ygang/Documents/anomaly/MSAD/data/TSB/data/NAB_KAGGLE"

def clean_name(fname):
    """
    NAB_KAGGLE 파일명에서 prefix 제거 규칙:
    - artificialNoAnomaly_
    - artificialWithAnomaly_
    - realAdExchange_
    - realAWSCloudwatch_
    - realKnownCause_
    - realTweets_
    - realTraffic_
    - realRogueAgent_
    """
    prefixes = [
        "artificialNoAnomaly_",
        "artificialWithAnomaly_",
        "realAdExchange_",
        "realAWSCloudwatch_",
        "realKnownCause_",
        "realTweets_",
        "realTraffic_",
        "realRogueAgent_"
    ]
    for p in prefixes:
        if fname.startswith(p):
            return fname[len(p):]   # prefix 삭제 후 반환
    return fname

def main():
    print("=== Renaming NAB_KAGGLE filenames to TSB format ===")
    for fname in os.listdir(NAB_PATH):
        if not fname.endswith(".csv"):
            continue

        new_name = clean_name(fname)
        old_path = os.path.join(NAB_PATH, fname)
        new_path = os.path.join(NAB_PATH, new_name)

        if old_path != new_path:
            print(f"Renaming: {fname} → {new_name}")
            os.rename(old_path, new_path)

    print("=== Done. ===")

if __name__ == "__main__":
    main()