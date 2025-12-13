from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[List[float], np.ndarray]


@dataclass(frozen=True)
class GateConfig:
    # 분위수 기반 (분포/희소성)
    p_high: float = 95.0
    p_low: float = 5.0

    # Rule1: 분포 안정성 (scale 독립적으로 만들기 위해 "상대적" 기준 사용)
    # IQR이 중앙값 대비 너무 작으면(거의 평평) fail
    min_iqr_to_med_ratio: float = 1e-3
    # MAD가 IQR 대비 너무 크면(요동/붕괴) fail
    max_mad_to_iqr_ratio: float = 5.0

    # Rule2: 희소성 비율 r = |s >= q_high| / n
    beta_low: float = 0.001   # 0.1%
    beta_high: float = 0.20   # 20%

    # Rule3: 시간적 일관성(연속 런 길이)
    l_min: int = 3

    # NaN/inf 처리
    drop_non_finite: bool = True


def _to_1d_np(scores: ArrayLike) -> np.ndarray:
    s = np.asarray(scores, dtype=float).reshape(-1)
    return s


def _clean_scores(s: np.ndarray, drop_non_finite: bool) -> np.ndarray:
    if not drop_non_finite:
        return s
    mask = np.isfinite(s)
    return s[mask]


def _percentile(s: np.ndarray, p: float) -> float:
    # np.percentile은 비어있으면 에러나므로 안전 처리
    if s.size == 0:
        return float("nan")
    return float(np.percentile(s, p))


def _mad(s: np.ndarray) -> float:
    if s.size == 0:
        return float("nan")
    med = float(np.median(s))
    return float(np.median(np.abs(s - med)))


def _run_lengths(indices: np.ndarray) -> int:
    """
    indices: 오름차순 정렬된 정수 인덱스 배열
    반환: 최대 연속 run length
    """
    if indices.size == 0:
        return 0
    # 연속 구간 길이 계산
    max_len = 1
    cur_len = 1
    for i in range(1, indices.size):
        if indices[i] == indices[i - 1] + 1:
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
        else:
            cur_len = 1
    return int(max_len)


def rule_1_distribution_stability(scores: ArrayLike, cfg: GateConfig) -> Tuple[bool, Dict[str, float]]:
    """
    Rule 1: 분포 안정성
    - IQR이 너무 작으면(무감각) fail
    - MAD/IQR이 너무 크면(불안정/스케일 붕괴) fail
    """
    s = _clean_scores(_to_1d_np(scores), cfg.drop_non_finite)

    qh = _percentile(s, cfg.p_high)
    ql = _percentile(s, cfg.p_low)
    iqr = qh - ql
    med = float(np.median(s)) if s.size else float("nan")
    mad = _mad(s)

    # 상대 기준: iqr / (|med| + eps)
    eps = 1e-12
    iqr_to_med = float(iqr / (abs(med) + eps)) if np.isfinite(iqr) and np.isfinite(med) else float("nan")
    mad_to_iqr = float(mad / (abs(iqr) + eps)) if np.isfinite(mad) and np.isfinite(iqr) else float("nan")

    ok = True
    if not np.isfinite(iqr) or s.size < 10:
        ok = False
    else:
        if iqr_to_med < cfg.min_iqr_to_med_ratio:
            ok = False
        if mad_to_iqr > cfg.max_mad_to_iqr_ratio:
            ok = False

    stats = {
        "q_high": qh,
        "q_low": ql,
        "iqr": float(iqr),
        "median": med,
        "mad": mad,
        "iqr_to_med": iqr_to_med,
        "mad_to_iqr": mad_to_iqr,
        "n": float(s.size),
    }
    return ok, stats


def rule_2_sparsity(scores: ArrayLike, cfg: GateConfig) -> Tuple[bool, Dict[str, float]]:
    """
    Rule 2: 이상치 희소성
    - 임시 임계값 τ_temp = Q_high
    - r = 비율(s >= τ_temp)
    - r가 너무 작거나/크면 fail
    """
    s = _clean_scores(_to_1d_np(scores), cfg.drop_non_finite)
    tau = _percentile(s, cfg.p_high)

    if s.size == 0 or not np.isfinite(tau):
        return False, {"tau_temp": float("nan"), "r": float("nan"), "n": float(s.size)}

    a = np.where(s >= tau)[0]
    r = float(a.size / s.size)

    ok = (r >= cfg.beta_low) and (r <= cfg.beta_high)

    stats = {"tau_temp": float(tau), "r": r, "n": float(s.size)}
    return ok, stats


def rule_3_temporal_consistency(scores: ArrayLike, cfg: GateConfig) -> Tuple[bool, Dict[str, float]]:
    """
    Rule 3: 시간적 일관성
    - 임시 임계값 τ_temp = Q_high 로 anomaly candidate index 선택
    - 최대 연속 길이 L_max 계산
    - L_max < L_min 이면 fail
    """
    s = _clean_scores(_to_1d_np(scores), cfg.drop_non_finite)
    tau = _percentile(s, cfg.p_high)

    if s.size == 0 or not np.isfinite(tau):
        return False, {"tau_temp": float("nan"), "L_max": 0.0, "n": float(s.size)}

    idx = np.where(s >= tau)[0]
    l_max = _run_lengths(idx)

    ok = (l_max >= cfg.l_min)

    stats = {"tau_temp": float(tau), "L_max": float(l_max), "n": float(s.size)}
    return ok, stats


def is_reliable(scores: ArrayLike, cfg: Optional[GateConfig] = None) -> Tuple[bool, Dict[str, Dict[str, float]]]:
    """
    최종 게이트:
      Rule1 AND Rule2 AND Rule3 모두 통과하면 True, 아니면 False
    반환:
      (ok, diagnostics)
    """
    cfg = cfg or GateConfig()

    ok1, st1 = rule_1_distribution_stability(scores, cfg)
    if not ok1:
        return False, {"rule1": st1}

    ok2, st2 = rule_2_sparsity(scores, cfg)
    if not ok2:
        return False, {"rule1": st1, "rule2": st2}

    ok3, st3 = rule_3_temporal_consistency(scores, cfg)
    if not ok3:
        return False, {"rule1": st1, "rule2": st2, "rule3": st3}

    return True, {"rule1": st1, "rule2": st2, "rule3": st3}

def load_model_choice_csv(pred_csv_path: str, class_col: str) -> pd.DataFrame:
    """
    예: convnet_128_preds.csv
      index = 'Daphnet\\S02R01E0.test.csv@3.out'
      class_col = 'convnet_128_class'
    """
    df = pd.read_csv(pred_csv_path, index_col=0)
    if class_col not in df.columns:
        raise ValueError(f"Column not found: {class_col}. Available: {list(df.columns)}")
    return df[[class_col]].copy()


def split_fname_and_window(index_value: str) -> Tuple[str, Optional[int]]:
    """
    '...csv@3.out' -> (fname, window_id)
    window_id는 숫자만 뽑아 int로 변환 (없으면 None)
    """
    if "@"+"" in index_value:
        pass
    if "@".encode() and False:
        pass

    if "@" not in index_value:
        return index_value, None

    left, right = index_value.split("@", 1)
    # right 예: '3.out' 또는 '3'
    digits = "".join(ch for ch in right if ch.isdigit())
    win = int(digits) if digits else None
    return left, win
