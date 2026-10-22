"""모델 구성과 '약한 라벨 검증 + 골드셋' 동시 평가 헬퍼."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .evaluate import apply_thresholds, multilabel_report, tune_thresholds
from .features import make_features


def as_frame(df: pd.DataFrame, tokens_col="tokens", text_col="text") -> pd.DataFrame:
    """모델 입력 DataFrame: tokens(형태소), text(원문·마스킹 문자열), press, log_len."""
    return pd.DataFrame({"tokens": df[tokens_col].values, "text": df[text_col].values,
                         "press": df["press"].fillna("unknown").values, "log_len": df["log_len"].values})


def lr_pipeline(C=1.0, penalty="l2", feature_kw=None, n_jobs=-1) -> Pipeline:
    """n_jobs: 태그별 이진 분류기 병렬 수. GridSearchCV 안에서는 1로 두고 바깥에서 병렬화한다(중첩 병렬 시 크래시)."""
    feature_kw = feature_kw or {}
    lr = LogisticRegression(C=C, penalty=penalty, solver="liblinear", class_weight="balanced", max_iter=2000)
    return Pipeline([("features", make_features(**feature_kw)), ("clf", OneVsRestClassifier(lr, n_jobs=n_jobs))])


def predict_proba(model, X) -> np.ndarray:
    P = model.predict_proba(X)
    return np.asarray(P.toarray() if hasattr(P, "toarray") else P)


class GoldEvaluator:
    """학습된 모델(또는 점수 함수)을 약한 검증셋과 골드셋(검증 100으로 임계값 → 테스트 200 평가)에서 함께 평가."""

    def __init__(self, Y_weak_valid, Y_gold_valid, Y_gold_test):
        self.Yw, self.Ygv, self.Ygt = map(np.asarray, (Y_weak_valid, Y_gold_valid, Y_gold_test))
        self.rows = []

    def evaluate(self, name, P_weak, P_gold_valid, P_gold_test, weak_th=None, note=""):
        weak_th = np.full(self.Yw.shape[1], 0.5) if weak_th is None else weak_th
        th = tune_thresholds(self.Ygv, P_gold_valid)  # 공통 임계값 + 양성 5개 이상 태그만 개별 조정
        row = {"model": name, "note": note}
        row.update(multilabel_report(self.Yw, apply_thresholds(P_weak, weak_th), P_weak, prefix="weak_"))
        row.update(multilabel_report(self.Ygt, apply_thresholds(P_gold_test, th), P_gold_test, prefix="gold_"))
        row.update(multilabel_report(self.Ygt, apply_thresholds(P_gold_test, weak_th), prefix="gold@0.5_"))
        self.rows.append(row)
        return th, row

    def table(self, cols=None):
        cols = cols or ["model", "note", "weak_micro_f1", "weak_macro_f1", "gold_micro_f1", "gold_macro_f1",
                        "gold_micro_precision", "gold_micro_recall", "gold_precision_at_1", "gold_hamming_loss",
                        "gold_none_accuracy", "gold@0.5_micro_f1"]
        return pd.DataFrame(self.rows)[cols]


class SavedTagger:
    """04에서 저장한 모델(lr/rf/xgb)을 같은 인터페이스로 쓰기 위한 래퍼."""

    def __init__(self, path):
        import joblib
        self.kind = path.stem
        self.obj = joblib.load(path)
        self.tags, self.thresholds = self.obj["tags"], np.asarray(self.obj["thresholds"])

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        o = self.obj
        if self.kind == "lr":
            return predict_proba(o["model"], X)
        Fm = o["features"].transform(X).tocsr()
        if self.kind == "rf":
            return predict_proba(o["model"], o["svd"].transform(Fm))
        S = Fm[:, o["keep"]]
        return np.column_stack([m.predict_proba(S)[:, 1] for m in o["models"]])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return apply_thresholds(self.proba(X), self.thresholds)

    def tag_lists(self, X: pd.DataFrame) -> list[list[str]]:
        return [[t for t, v in zip(self.tags, row) if v] for row in self.predict(X)]


# ---------------------------------------------------------------- 하이퍼파라미터 탐색 (캐시)
# 탐색은 수십 분~1시간 걸린다. 결과를 outputs/models/*_search.joblib에 저장해 두고, 노트북을 다시 실행할 때는
# 저장된 탐색 결과를 불러온다. 데이터나 탐색 공간을 바꾸면 캐시 파일을 지우고 `python scripts/tune.py`로 다시 돌린다.
FEAT_DEFAULT = dict(word=True, char=True, keyword=True, meta=False)


def _cached(path, fit_fn):
    import joblib
    from pathlib import Path
    if path is not None and Path(path).exists():
        return joblib.load(path)
    obj = fit_fn()
    if path is not None:
        joblib.dump(obj, path)
    return obj


def search_lr(X, Y, cache=None, verbose=0):
    """LR: GridSearchCV — C × 규제(L1/L2) × 단어 n-gram, TimeSeriesSplit(5). 학습 점수도 남겨 과적합(gap)을 본다."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    grid = {"clf__estimator__C": [0.1, 1, 10], "clf__estimator__penalty": ["l1", "l2"],
            "features__word__ngram_range": [(1, 1), (1, 2)]}

    def fit():
        gs = GridSearchCV(lr_pipeline(feature_kw=FEAT_DEFAULT, n_jobs=1), grid, cv=TimeSeriesSplit(n_splits=5),
                          scoring="f1_micro", n_jobs=12, return_train_score=True, verbose=verbose)
        return gs.fit(X, Y)
    return _cached(cache, fit)


def search_rf(D, Y, cache=None, verbose=0, n_iter=12):
    """RF(SVD 300차원 입력): RandomizedSearchCV 12회, TimeSeriesSplit(3).
    1차 실행에서 깊은 트리·큰 표본 비율 조합과 학습 점수 계산(return_train_score)까지 넣으니 2시간이 넘어
    표본 비율(max_samples)을 낮추고 학습 점수 계산을 뺐다. 2차 실행은 시간 제약으로 30회 → 12회로 줄였다. 과적합은 최종 모델의 학습/검증 점수 차이로 본다."""
    from scipy.stats import randint
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    space = {"estimator__n_estimators": randint(100, 250), "estimator__max_depth": [20, 40, None],
             "estimator__min_samples_leaf": randint(2, 10), "estimator__max_features": ["sqrt", 0.1],
             "estimator__max_samples": [0.3, 0.5]}
    base = OneVsRestClassifier(RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=42))

    def fit():
        rs = RandomizedSearchCV(base, space, n_iter=n_iter, cv=TimeSeriesSplit(3), scoring="f1_micro",
                                random_state=42, n_jobs=1, verbose=verbose)
        return rs.fit(D, Y)
    return _cached(cache, fit)


def search_xgb(S, Y, cache=None, verbose=0, n_iter=15):
    """XGBoost(chi² 상위 3천 개 희소 피처): RandomizedSearchCV 15회(시간 제약으로 30회에서 축소), TimeSeriesSplit(3)."""
    from scipy.stats import loguniform, randint, uniform
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from xgboost import XGBClassifier
    pos = Y.mean(0)
    space = {"estimator__n_estimators": randint(100, 400), "estimator__max_depth": randint(3, 9),
             "estimator__learning_rate": loguniform(0.03, 0.3), "estimator__min_child_weight": randint(1, 8),
             "estimator__subsample": uniform(0.6, 0.4), "estimator__colsample_bytree": uniform(0.3, 0.5),
             "estimator__reg_lambda": loguniform(0.5, 10)}
    base = OneVsRestClassifier(XGBClassifier(tree_method="hist", max_bin=32, n_jobs=16, eval_metric="logloss",
                                             scale_pos_weight=float(np.median((1 - pos) / pos)), random_state=42))

    def fit():
        rs = RandomizedSearchCV(base, space, n_iter=n_iter, cv=TimeSeriesSplit(3), scoring="f1_micro",
                                random_state=42, n_jobs=1, verbose=verbose)
        return rs.fit(S, Y)
    return _cached(cache, fit)


def tree_features(Xtr, Ytr, word_ngram, others=(), n_svd=300, n_chi2=3000):
    """RF·XGBoost 입력: (학습셋에 맞춘) TF-IDF+키워드 피처 → RF용 SVD 300차원, XGBoost용 chi² 상위 3천 개."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_selection import chi2
    feat = make_features(**FEAT_DEFAULT, word_ngram=word_ngram)
    Ftr = feat.fit_transform(Xtr).tocsr()
    Fo = [feat.transform(X).tocsr() for X in others]
    svd = TruncatedSVD(n_components=n_svd, random_state=42).fit(Ftr)
    chi = np.nan_to_num(np.vstack([chi2(Ftr, Ytr[:, j])[0] for j in range(Ytr.shape[1])])).max(0)
    keep = np.argsort(-chi)[:n_chi2]
    return {"feat": feat, "svd": svd, "keep": keep, "F": [Ftr] + Fo,
            "D": [svd.transform(M) for M in [Ftr] + Fo], "S": [M[:, keep] for M in [Ftr] + Fo]}
