"""다중 라벨 평가 지표, 태그별 임계값 조정, 키워드 룰 베이스라인."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


def precision_at_1(Y_true: np.ndarray, P: np.ndarray) -> float:
    """정답 태그가 1개 이상인 기사에서 확률 1위 태그가 정답에 들어있는 비율."""
    has = Y_true.sum(1) > 0
    if not has.any():
        return float("nan")
    top = P[has].argmax(1)
    return float(Y_true[has][np.arange(has.sum()), top].mean())


def multilabel_report(Y_true, Y_pred, P=None, prefix="") -> dict:
    Y_true, Y_pred = np.asarray(Y_true), np.asarray(Y_pred)
    p, r, f, _ = precision_recall_fscore_support(Y_true, Y_pred, average="micro", zero_division=0)
    out = {
        "micro_f1": f, "micro_precision": p, "micro_recall": r,
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
        # '관련 테마 없음'을 정확히 맞혔는지 (정답도 예측도 비어 있음)
        "none_accuracy": float(((Y_true.sum(1) == 0) == (Y_pred.sum(1) == 0)).mean()),
    }
    if P is not None:
        out["precision_at_1"] = precision_at_1(Y_true, np.asarray(P))
    return {prefix + k: round(float(v), 4) for k, v in out.items()}


def per_tag_report(Y_true, Y_pred, tags) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(Y_true, Y_pred, average=None, zero_division=0)
    return pd.DataFrame({"tag": tags, "precision": p, "recall": r, "f1": f, "support": s}).round(3)


def tune_global_threshold(Y_true, P, grid=None) -> float:
    """모든 태그에 같은 임계값을 쓸 때 Micro F1이 최대인 값 (동점이면 높은 값 = 정밀도 우선)."""
    Y_true, P = np.asarray(Y_true), np.asarray(P)
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
    scores = [f1_score(Y_true, P >= t, average="micro", zero_division=0) for t in grid]
    best = max(scores)
    return float(max(t for t, s in zip(grid, scores) if s == best))


def tune_thresholds(Y_true, P, grid=None, fallback=None, min_pos: int = 5) -> np.ndarray:
    """태그별로 F1이 최대가 되는 임계값. 검증셋에서 양성이 min_pos개 미만인 태그는 fallback을 쓴다.
    fallback을 주지 않으면 같은 검증셋에서 정한 공통 임계값(tune_global_threshold)을 쓴다.
    골드 검증셋은 100건이라 태그당 양성이 0~14개뿐이어서, 양성 2~4개로 태그별 최적화를 하면 과적합된다.
    동점이면 더 높은 임계값(=정밀도 우선)을 고른다."""
    Y_true, P = np.asarray(Y_true), np.asarray(P)
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
    if fallback is None:
        fallback = tune_global_threshold(Y_true, P, grid)
    th = np.broadcast_to(np.asarray(fallback, dtype=float), (P.shape[1],)).copy()
    for j in range(P.shape[1]):
        if Y_true[:, j].sum() < min_pos:
            continue
        scores = [f1_score(Y_true[:, j], P[:, j] >= t, zero_division=0) for t in grid]
        best = max(scores)
        th[j] = max(t for t, s in zip(grid, scores) if s == best)
    return th


def apply_thresholds(P, th) -> np.ndarray:
    P = np.asarray(P)
    th = np.broadcast_to(np.asarray(th, dtype=float), (P.shape[1],))  # 스칼라 임계값도 허용
    return (P >= th[None, :]).astype(int)


class KeywordRuleBaseline:
    """시드 키워드가 n개 이상 나오면 태그 부여. 확률 대신 키워드 개수를 점수로 낸다."""

    def __init__(self, keywords: dict[str, list[str]], n: int = 2):
        self.keywords, self.n = keywords, n
        self.tags = list(keywords)
        self.sets = [set(w.upper() if w.isascii() else w for w in keywords[t]) for t in self.tags]

    def scores(self, token_docs) -> np.ndarray:
        S = np.zeros((len(token_docs), len(self.tags)))
        for i, doc in enumerate(token_docs):
            toks = doc.split()
            for j, s in enumerate(self.sets):
                S[i, j] = sum(t in s for t in toks)
        return S

    def predict(self, token_docs) -> np.ndarray:
        return (self.scores(token_docs) >= self.n).astype(int)
