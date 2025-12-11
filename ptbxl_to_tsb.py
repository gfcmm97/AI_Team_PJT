import os
import wfdb
import pandas as pd
from tqdm import tqdm

PTBXL_ROOT = "/home/ygang/Documents/anomaly/MSAD/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
SAVE_DIR = "/home/ygang/Documents/anomaly/MSAD/data/TSB/data/PTBXL"

os.makedirs(SAVE_DIR, exist_ok=True)

records_file = os.path.join(PTBXL_ROOT, "RECORDS")
print("Reading RECORDS...")

clean_records = []
with open(records_file, "r") as f:
    for raw_line in f:
        line = raw_line.strip()

        # 1) invalid: two paths are stuck together (contains both records100 & records500)
        if "records100" in line and "records500" in line:
            print(f"[SKIP] corrupted RECORDS line: {line}")
            continue

        # 2) basic validity check
        if not (line.startswith("records100") or line.startswith("records500")):
            print(f"[SKIP] invalid RECORDS line: {line}")
            continue

        # 3) ensure either .dat or .hea exists
        dat_path = os.path.join(PTBXL_ROOT, line + ".dat")
        hea_path = os.path.join(PTBXL_ROOT, line + ".hea")

        if not (os.path.exists(dat_path) and os.path.exists(hea_path)):
            print(f"[SKIP] missing file pair: {line}")
            continue

        clean_records.append(line)

print("Valid records:", len(clean_records))

# --- Convert ---
for rp in tqdm(clean_records):
    full_path = os.path.join(PTBXL_ROOT, rp)

    try:
        record = wfdb.rdrecord(full_path)
        data = record.p_signal

        # Lead II
        lead_signal = data[:, 1]

        file_id = os.path.basename(rp)
        numeric_id = file_id.split("_")[0]
        save_path = os.path.join(SAVE_DIR, f"{numeric_id}.csv")

        df = pd.DataFrame({"value": lead_signal})
        df.to_csv(save_path, index=False)

    except Exception as e:
        print(f"[ERROR] {rp}: {e}")
        continue

print("=== PTB-XL → TSB 변환 완료 ===")