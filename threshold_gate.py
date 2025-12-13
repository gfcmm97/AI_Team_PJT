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

    # 후보 랭킹 시 confidence 반영 여부
    use_confidence: bool = True
    # 랭킹 점수에서 빈도 vs confidence 가중치
    w_freq: float = 1.0
    w_conf: float = 0.25

    # 순차 시도 시 최대 몇 개 detector까지 시도할지 (앙상블 회피용)
    max_trials: int = 3


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

from typing import Callable, Any

# ---------- 6개 선택기 결과 로딩 ----------

def load_selector_preds(pred_csv_path: str, class_col: str, conf_col: Optional[str] = None) -> pd.DataFrame:
    """
    shared_results의 *_preds.csv 로딩
    index: '...@win.out'
    class_col: 예 'convnet_128_class'
    conf_col: 예 'convnet_128_inf' (없어도 됨)
    """
    df = pd.read_csv(pred_csv_path, index_col=0)
    cols = [class_col]
    if conf_col and conf_col in df.columns:
        cols.append(conf_col)
    if class_col not in df.columns:
        raise ValueError(f"Column not found: {class_col}. Available: {list(df.columns)}")
    return df[cols].copy()


def default_selector_sources(shared_dir: str = r".\shared_results") -> Dict[str, Dict[str, str]]:
    return {
        "convnet_128": {
            "path": rf"{shared_dir}\convnet_128_preds.csv",
            "class_col": "convnet_128_class",
            "conf_col": "convnet_128_inf",
        },
        "resnet_128": {
            "path": rf"{shared_dir}\resnet_128_preds.csv",
            "class_col": "resnet_128_class",
            "conf_col": "resnet_128_inf",
        },
        "sit_stem_512": {
            "path": rf"{shared_dir}\sit_stem_512_preds.csv",
            "class_col": "sit_stem_512_class",
            "conf_col": "sit_stem_512_inf",
        },
        "sit_768": {
            "path": rf"{shared_dir}\sit_stem_ReLU_768_preds.csv",
            "class_col": "sit_768_class",
            "conf_col": "sit_768_inf",
        },
        "rocket_128": {
            "path": rf"{shared_dir}\rocket_128_preds.csv",
            "class_col": "rocket_128_class",
            "conf_col": "rocket_128_inf",
        },
        "knn_1024": {
            "path": rf"{shared_dir}\knn_1024_preds.csv",
            "class_col": "knn_1024_class",
            "conf_col": "knn_1024_inf",
        },
    }


def load_all_selectors(sources: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, meta in sources.items():
        out[name] = load_selector_preds(meta["path"], meta["class_col"], meta.get("conf_col"))
    return out


# ---------- 후보 랭킹(개선점 반영) ----------

def rank_detectors_for_key(
    selectors: Dict[str, pd.DataFrame],
    key: str,
    sources: Dict[str, Dict[str, str]],
    cfg: GateConfig,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    key(예: 'Daphnet\\S02R01E0.test.csv@3.out')에 대해
    6개 선택기가 추천한 detector들을 모아 우선순위로 정렬.
    점수 = w_freq * (빈도/6) + w_conf * (평균 confidence)
    반환: [(detector, score, meta), ...] 내림차순
    """
    votes: List[str] = []
    confs: List[float] = []

    # detector별 집계
    det_freq: Dict[str, int] = {}
    det_conf_sum: Dict[str, float] = {}
    det_conf_cnt: Dict[str, int] = {}

    for sel_name, df in selectors.items():
        if key not in df.index:
            continue

        class_col = sources[sel_name]["class_col"]
        det = str(df.loc[key, class_col])
        votes.append(det)
        det_freq[det] = det_freq.get(det, 0) + 1

        conf_col = sources[sel_name].get("conf_col")
        if cfg.use_confidence and conf_col and conf_col in df.columns:
            try:
                c = float(df.loc[key, conf_col])
                if np.isfinite(c):
                    det_conf_sum[det] = det_conf_sum.get(det, 0.0) + c
                    det_conf_cnt[det] = det_conf_cnt.get(det, 0) + 1
            except Exception:
                pass

    total = max(1, len(votes))
    ranked = []
    for det, f in det_freq.items():
        freq_norm = f / total
        conf_avg = 0.0
        if cfg.use_confidence and det_conf_cnt.get(det, 0) > 0:
            conf_avg = det_conf_sum[det] / det_conf_cnt[det]

        score = cfg.w_freq * freq_norm + cfg.w_conf * conf_avg
        ranked.append((det, float(score), {
            "freq": f,
            "freq_norm": float(freq_norm),
            "conf_avg": float(conf_avg),
        }))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ---------- 순차 시도 게이트(앙상블 최소화) ----------

DetectorRunner = Callable[[str, str], np.ndarray]
# signature: run_detector(detector_name, key) -> 1D score array

EnsembleRunner = Callable[[List[Tuple[str, np.ndarray]], str], np.ndarray]
# signature: run_ensemble([(det, score), ...], key) -> 1D ensemble score array


def choose_detector_with_gate(
    key: str,
    selectors: Dict[str, pd.DataFrame],
    sources: Dict[str, Dict[str, str]],
    run_detector: DetectorRunner,
    cfg: Optional[GateConfig] = None,
    run_ensemble: Optional[EnsembleRunner] = None,
) -> Dict[str, Any]:
    """
    핵심 함수.
    - 6개 선택기 추천 -> detector 우선순위 리스트 생성
    - 상위 max_trials개까지 순차 실행
      - score 생성 -> is_reliable(score) 통과하면 즉시 채택
    - 전부 실패하면:
      - run_ensemble 있으면 앙상블 score 생성 후 반환
      - 없으면 실패로 반환

    반환 예:
    {
      "key": key,
      "picked_mode": "single" or "ensemble" or "fail",
      "picked_detector": "IFOREST",
      "trials": [
          {"detector":"IFOREST","ok":False,"diag":...},
          {"detector":"PCA","ok":True,"diag":...},
      ],
      "ranked": [...],
    }
    """
    cfg = cfg or GateConfig()

    ranked = rank_detectors_for_key(selectors, key, sources, cfg)
    trials = []
    tried_scores: List[Tuple[str, np.ndarray]] = []

    # 후보가 아예 없으면 fail
    if len(ranked) == 0:
        return {
            "key": key,
            "picked_mode": "fail",
            "picked_detector": None,
            "ranked": [],
            "trials": [],
            "reason": "no_selector_votes_for_key",
        }

    # 순차 시도
    for det, det_score, meta in ranked[: cfg.max_trials]:
        scores = run_detector(det, key)  # 반드시 1D score 반환해야 함
        ok, diag = is_reliable(scores, cfg)

        trials.append({
            "detector": det,
            "rank_score": det_score,
            "vote_meta": meta,
            "ok": ok,
            "diag": diag,
        })
        tried_scores.append((det, _to_1d_np(scores)))

        if ok:
            return {
                "key": key,
                "picked_mode": "single",
                "picked_detector": det,
                "ranked": ranked,
                "trials": trials,
            }

    # 여기까지 왔으면: top-N 후보 전부 fail
    if run_ensemble is not None:
        ens_scores = run_ensemble(tried_scores, key)
        return {
            "key": key,
            "picked_mode": "ensemble",
            "picked_detector": None,
            "ranked": ranked,
            "trials": trials,
            "ensemble_used_detectors": [d for d, _ in tried_scores],
            "ensemble_scores_shape": int(_to_1d_np(ens_scores).size),
        }

    return {
        "key": key,
        "picked_mode": "fail",
        "picked_detector": None,
        "ranked": ranked,
        "trials": trials,
        "reason": "all_trials_failed_no_ensemble_runner",
    }
