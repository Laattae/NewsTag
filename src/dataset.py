"""약한 라벨 데이터셋 구성, 중복 제거, 시간순 분할."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import TAGS


def article_key(df: pd.DataFrame) -> pd.Series:
    """기사 식별자: 원문 링크(없으면 네이버 링크). 국내 언론사 다수가 `articleView.html?idxno=123` 형식이라
    쿼리스트링은 남기고, 추적용 파라미터(utm_*)와 #앵커만 지운다."""
    url = df["originallink"].fillna("").where(df["originallink"].fillna("") != "", df["link"])
    url = url.str.replace(r"#.*$", "", regex=True).str.replace(r"[?&]utm_[^&]*", "", regex=True)
    return url.str.replace(r"^https?://(www\.|m\.)?", "", regex=True).str.rstrip("/")


def build_weak_labels(raw: pd.DataFrame, title_filter: bool = True) -> pd.DataFrame:
    """D2 원본 → 기사 단위 약한 라벨.
    1) (선택) 제목에 종목명이 있는 행만 남김  2) 같은 URL은 합치고 태그는 합집합."""
    df = raw.copy()
    if title_filter:
        df = df[df["title_has_name"].astype(bool)]
    df["key"] = article_key(df)
    df["tag_list"] = df["tags"].str.split("|")
    agg = df.groupby("key").agg(
        title=("title", "first"), description=("description", "first"),
        pub_date=("pub_date", "min"), press=("press", "first"),
        stocks=("stock", lambda s: "|".join(sorted(set(s)))),
        tag_list=("tag_list", lambda s: sorted(set(t for ts in s for t in ts))),
    ).reset_index()
    agg["pub_date"] = pd.to_datetime(agg["pub_date"])
    agg["n_tags"] = agg["tag_list"].str.len()
    return agg.sort_values("pub_date").reset_index(drop=True)


def label_matrix(tag_lists, tags=TAGS) -> np.ndarray:
    idx = {t: i for i, t in enumerate(tags)}
    Y = np.zeros((len(tag_lists), len(tags)), dtype=int)
    for r, ts in enumerate(tag_lists):
        for t in ts:
            if t in idx:
                Y[r, idx[t]] = 1
    return Y


_NORM_RE = re.compile(r"[^가-힣A-Za-z0-9]")


def title_signature(title: str, n: int = 25) -> str:
    """거의 같은 제목(통신사 재전송, [속보]/[단독] 머리말 차이) 판별용 정규화 키."""
    t = re.sub(r"^\s*(\[[^\]]{1,10}\]|【[^】]{1,10}】)\s*", "", str(title))
    return _NORM_RE.sub("", t)[:n]


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """같은 URL + 정규화 제목이 같은 기사는 하나만 (가장 이른 기사 유지)."""
    df = df.sort_values("pub_date")
    df = df.drop_duplicates("key")
    sig = df["title"].map(title_signature)
    return df[~sig.duplicated()].reset_index(drop=True)


def cap_per_tag(df: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """태그별 기사 수 상한. 희귀 태그 기사를 먼저 확보한 뒤 흔한 태그를 채워 불균형을 줄인다."""
    rng = np.random.default_rng(seed)
    counts = pd.Series([t for ts in df.tag_list for t in ts]).value_counts()
    keep = set()
    got = dict.fromkeys(counts.index, 0)
    for tag in counts.index[::-1]:  # 희귀한 태그부터
        cand = df.index[df.tag_list.map(lambda ts: tag in ts)].difference(list(keep))
        need = max(cap - got[tag], 0)
        pick = rng.choice(cand, size=min(need, len(cand)), replace=False) if need else []
        for i in pick:
            keep.add(i)
            for t in df.at[i, "tag_list"]:
                got[t] += 1
    return df.loc[sorted(keep)].sort_values("pub_date").reset_index(drop=True)


def time_split(df: pd.DataFrame, frac: float = 0.8):
    """날짜순 앞 80% 학습, 뒤 20% 검증 (무작위 분할 금지: 같은 사건 기사가 양쪽에 들어가는 것을 막음)."""
    df = df.sort_values("pub_date").reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def load_gold(path=None) -> pd.DataFrame:
    """골드셋(표본 텍스트 + 사람이 단 tags) → 모델 입력 열(tokens, text, press, log_len)까지 갖춘 DataFrame."""
    from .config import GOLD
    from .text import tokenize_many
    g = pd.read_csv(path or GOLD / "gold_labels.csv")
    g["tag_list"] = g["tags"].fillna("").map(lambda s: [t.strip() for t in s.split("|") if t.strip()])
    bad = {t for ts in g.tag_list for t in ts} - set(TAGS)
    assert not bad, f"태그 체계에 없는 라벨: {bad}"
    from .masking import default_masker
    g["text"] = g["text_raw"].map(default_masker().mask)  # 학습과 같은 마스킹 (경영자 이름, 남은 계열사명 등)
    g["tokens"] = tokenize_many(g.text.tolist())
    g["log_len"] = np.log1p(g.text.str.len())
    g["pub_date"] = pd.to_datetime(g.pub_date)
    return g
