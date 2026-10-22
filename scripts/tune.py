"""04_modeling의 하이퍼파라미터 탐색(LR·RF·XGBoost)을 진행 로그와 함께 미리 돌려 캐시에 저장한다.
노트북은 같은 함수(src.modeling.search_*)로 캐시를 불러 쓰므로 결과가 같다.

    python scripts/tune.py            # 없는 캐시만 계산
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from joblib import parallel_config

from src.config import OUTPUTS, PROCESSED
from src.dataset import label_matrix
from src.modeling import as_frame, search_lr, search_rf, search_xgb, tree_features

if __name__ == "__main__":
    M = OUTPUTS / "models"
    M.mkdir(exist_ok=True)
    with parallel_config(max_nbytes=None):
        train = pd.read_pickle(PROCESSED / "weak_train.pkl")
        Xtr, Ytr = as_frame(train), label_matrix(train.tag_list)
        t = time.time()
        lr = search_lr(Xtr, Ytr, cache=M / "lr_search.joblib", verbose=1)
        print(f"LR {time.time() - t:.0f}s", lr.best_params_, round(lr.best_score_, 4), flush=True)
        tf = tree_features(Xtr, Ytr, lr.best_params_["features__word__ngram_range"])
        t = time.time()
        rf = search_rf(tf["D"][0], Ytr, cache=M / "rf_search.joblib", verbose=2)
        print(f"RF {time.time() - t:.0f}s", rf.best_params_, round(rf.best_score_, 4), flush=True)
        t = time.time()
        xgb = search_xgb(tf["S"][0], Ytr, cache=M / "xgb_search.joblib", verbose=2)
        print(f"XGB {time.time() - t:.0f}s", xgb.best_params_, round(xgb.best_score_, 4), flush=True)
