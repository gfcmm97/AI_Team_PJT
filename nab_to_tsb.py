import os
import pandas as pd

# 원본 NAB 데이터 최상위 폴더
SRC_ROOT = "/home/ygang/Documents/anomaly/MSAD/data/NAB"

# 변환 후 저장 폴더
DST = "/home/ygang/Documents/anomaly/MSAD/data/TSB/data/NAB_KAGGLE"
os.makedirs(DST, exist_ok=True)

print("=== NAB → TSB 변환 시작 ===")

total = 0

# NAB 폴더 내부의 모든 서브폴더 순회
for root, dirs, files in os.walk(SRC_ROOT):

    for fname in files:
        if not fname.endswith(".csv"):
            continue

        src_path = os.path.join(root, fname)

        # __MACOSX 등 숨김파일 제거
        if "/__MACOSX" in src_path:
            continue

        try:
            df = pd.read_csv(src_path)

            # 값이 있는 numeric column 자동 탐지
            value_col = None
            for col in df.columns:
                # 날짜(datetime)은 object, 값(value)는 float/int
                if df[col].dtype != object:
                    value_col = col
                    break

            if value_col is None:
                print(f"[SKIP] {fname}: numeric column 없음")
                continue

            values = df[value_col].astype(float).values
            labels = [0] * len(values)

            # 저장될 파일명: 원본 파일명에서 .csv → .out
            new_name = fname.replace(".csv", ".out")
            dst_path = os.path.join(DST, new_name)

            # 최종 TSB 형식 저장
            out = pd.DataFrame({
                0: values,
                1: labels
            })

            out.to_csv(dst_path, index=False, header=False)
            print(f"[OK] {fname} → {new_name}")
            total += 1

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

print(f"=== NAB → TSB 변환 완료 (총 {total}개 변환됨) ===")