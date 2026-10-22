"""데이터 수집기.

- D1: KRX 상장사 목록(업종 포함) + 태그 매핑표(tag_map.csv)
- D2: 대표 종목별 뉴스 (네이버 검색 Open API, 약한 라벨)
- D3: 종목명 없는 경제 뉴스 후보 (네이버 검색 Open API, 경제·산업 키워드)
- D5: 태그별 대표 종목 일별 종가 (FinanceDataReader) → 태그 지수

네이버 금융·뉴스 페이지는 robots.txt가 전체 크롤링을 막고 있어(2026-10 확인) 공식 검색 API(NAVER API HUB)를 쓴다.
API는 제목·요약(본문 앞 1~2문장)·원문 링크·발행시각을 주며 쿼리당 최대 1,100건(start≤1000)까지 받는다.
"""
from __future__ import annotations

import html
import os
import re
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import pandas as pd
import requests

from .config import ALIASES, EXTRA_TAGS, NAVER_REQUEST_SLEEP, RAW, ROOT, TAG_STOCKS

# 네이버 검색 API는 NAVER API HUB(네이버 클라우드)로 이관됐다. Developers 키(openapi.naver.com)는
# 신규 발급이 막혔으므로 HUB 엔드포인트를 쓴다. 키: NCP 콘솔 > NAVER API HUB에서 발급한 Client ID/Secret.
API_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"

# D3 검색어: 일반 거시·정책 키워드와 산업 동향 키워드를 섞는다.
# 산업 키워드만 쓰면 골드셋에서 '없음' 비율이 실제보다 낮아지므로 거시 키워드를 절반 이상 둔다.
D3_QUERIES_MACRO = [
    "기준금리", "금리 인하", "한국은행 통화정책", "환율 급등", "원달러 환율", "소비자물가", "물가 상승",
    "수출 증가", "수출 감소", "무역수지", "경상수지", "관세 부과", "미국 관세", "통상 협상", "보호무역",
    "경제성장률", "내수 부진", "고용 동향", "가계부채", "재정 정책", "추경 예산", "세법 개정", "기획재정부",
    "최저임금", "국제유가", "공급망", "중국 경기", "미국 연준", "FOMC", "경기 침체", "소비 심리",
    "규제 완화", "정부 지원책", "산업통상자원부", "공정거래위원회", "부동산 정책", "주택 공급", "전세 시장",
]
D3_QUERIES_INDUSTRY = [
    "HBM 수요", "메모리 가격", "반도체 수출", "파운드리", "전기차 수요", "배터리 업계", "리튬 가격",
    "완성차 업계", "자동차 수출", "선박 수주", "조선업계", "철강업계", "철강 관세", "구리 가격",
    "원전 수출", "SMR", "전력망", "변압기 수요", "방산 수출", "국방 예산", "신약 개발", "바이오 업계",
    "약가 인하", "게임 업계", "플랫폼 규제", "인공지능 투자", "은행권", "증권업계", "보험업계",
    "건설업계", "분양 시장", "PF 부실", "정유업계", "석유화학 구조조정", "정제마진", "항공업계",
    "항공 여객", "해외여행 수요", "방한 관광객", "면세점", "K뷰티 수출", "화장품 수출", "K팝", "엔터 업계",
    "OTT 시장", "콘텐츠 산업",
]


# ---------------------------------------------------------------- D1
def load_krx_listing(refresh: bool = False) -> pd.DataFrame:
    """KRX 전 종목 목록(업종 포함). 한 번 받은 뒤에는 CSV를 재사용한다."""
    path = RAW / "krx_listing_desc.csv"
    if refresh or not path.exists():
        import FinanceDataReader as fdr
        fdr.StockListing("KRX-DESC").to_csv(path, index=False, encoding="utf-8-sig")
    return pd.read_csv(path, dtype={"Code": str})


def build_tag_map(listing: pd.DataFrame) -> pd.DataFrame:
    """태그-대표종목 매핑표. 한 종목이 여러 태그에 속하면 행이 여러 개 생긴다."""
    rows = []
    for tag, stocks in TAG_STOCKS.items():
        for name in stocks:
            rows.append({"tag": tag, "name": name})
    for name, extra in EXTRA_TAGS.items():
        rows += [{"tag": t, "name": name} for t in extra]
    tm = pd.DataFrame(rows).drop_duplicates()
    info = listing[["Code", "Name", "Market", "Industry"]].rename(columns={"Code": "code", "Name": "name"})
    tm = tm.merge(info, on="name", how="left")
    missing = tm[tm.code.isna()]
    if len(missing):
        print("[경고] KRX 목록에 없는 종목:", missing.name.tolist())
    tm = tm.dropna(subset=["code"])
    tm["aliases"] = tm.name.map(lambda n: "|".join(ALIASES.get(n, [])))
    return tm.reset_index(drop=True)


# ---------------------------------------------------------------- 네이버 검색 API
def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


class NaverNewsAPI:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        _load_env()
        self.id = client_id or os.environ.get("NAVER_CLIENT_ID")
        self.secret = client_secret or os.environ.get("NAVER_CLIENT_SECRET")
        if not (self.id and self.secret):
            raise RuntimeError(".env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET을 넣어주세요.")
        self.session = requests.Session()
        self.session.headers.update({"X-NCP-APIGW-API-KEY-ID": self.id, "X-NCP-APIGW-API-KEY": self.secret})
        self.calls = 0

    def search(self, query: str, start: int = 1, sort: str = "date", display: int = 100) -> list[dict]:
        for attempt in range(5):
            r = self.session.get(API_URL, params={"query": query, "display": display,
                                                  "start": start, "sort": sort, "format": "json"},
                                 timeout=10)
            self.calls += 1
            time.sleep(NAVER_REQUEST_SLEEP)
            if r.status_code == 200:
                return r.json().get("items", [])
            if r.status_code == 429:  # 속도 제한: 잠깐 쉬고 재시도
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 400:  # start 범위 초과 등
                return []
            r.raise_for_status()
        return []

    def search_all(self, query: str, sorts=("date", "sim"), max_start: int = 1000) -> pd.DataFrame:
        """쿼리 하나를 정렬 방식별로 끝까지(최대 1,100건) 받는다."""
        out = []
        for sort in sorts:
            for start in range(1, max_start + 1, 100):
                items = self.search(query, start=start, sort=sort)
                for it in items:
                    it["sort"] = sort
                out += items
                if len(items) < 100:
                    break
        return normalize_items(pd.DataFrame(out), query)


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    return html.unescape(_TAG_RE.sub("", s or "")).strip()


def normalize_items(df: pd.DataFrame, query: str) -> pd.DataFrame:
    cols = ["query", "title", "description", "originallink", "link", "pub_date", "press", "sort"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df["query"] = query
    df["title"] = df["title"].map(strip_html)
    df["description"] = df["description"].map(strip_html)
    df["pub_date"] = df["pubDate"].map(lambda s: parsedate_to_datetime(s).strftime("%Y-%m-%d %H:%M"))
    src = df["originallink"].where(df["originallink"].astype(bool), df["link"])
    df["press"] = src.map(lambda u: urlparse(u).netloc.removeprefix("www.").removeprefix("m."))
    return df[cols]


# ---------------------------------------------------------------- D2
def title_has_name(title: str, names: list[str]) -> bool:
    """제목에 종목명/약칭이 들어있는지. 영문 약칭은 단어 경계로, 한글은 앞 글자가 한글이 아니어야 한다."""
    for n in names:
        if re.fullmatch(r"[A-Za-z0-9 .&\-]+", n):
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", title):
                return True
        elif re.search(rf"(?<![가-힣]){re.escape(n)}", title):
            return True
    return False


def collect_stock_news(api: NaverNewsAPI, tag_map: pd.DataFrame, out_path=None,
                       sorts=("date", "sim")) -> pd.DataFrame:
    """D2: 대표 종목별 뉴스. 제목 필터 전 원본(title_has_name 플래그 포함)을 저장한다."""
    out_path = out_path or RAW / "d2_stock_news_raw.csv"
    done = set()
    frames = []
    if os.path.exists(out_path):  # 중단 후 이어받기
        prev = pd.read_csv(out_path, dtype={"code": str})
        frames.append(prev)
        done = set(prev.code.unique())
    stocks = tag_map.groupby(["code", "name"]).agg(tags=("tag", lambda s: "|".join(sorted(set(s)))),
                                                    aliases=("aliases", "first")).reset_index()
    for i, row in enumerate(stocks.itertuples(), 1):
        if row.code in done:
            continue
        names = [row.name] + [a for a in (row.aliases or "").split("|") if a]
        # 2글자 이하 영문 약칭(SM, YG 등)은 검색어로 쓰지 않는다 (노이즈가 너무 큼)
        queries = [row.name] + [a for a in names[1:] if len(a) >= 3 and a != row.name][:1]
        df = pd.concat([api.search_all(q, sorts=sorts) for q in queries], ignore_index=True)
        df["code"], df["stock"], df["tags"] = row.code, row.name, row.tags
        df["title_has_name"] = df.title.map(lambda t: title_has_name(t, names))
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{i}/{len(stocks)}] {row.name}: {len(df)}건 (제목 포함 {df.title_has_name.sum()}) "
              f"누적 API 호출 {api.calls}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- D3
def collect_general_news(api: NaverNewsAPI, queries=None, out_path=None) -> pd.DataFrame:
    out_path = out_path or RAW / "d3_general_news_raw.csv"
    queries = queries or (D3_QUERIES_MACRO + D3_QUERIES_INDUSTRY)
    frames, done = [], set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        frames.append(prev)
        done = set(prev["query"].unique())
    for i, q in enumerate(queries, 1):
        if q in done:
            continue
        df = api.search_all(q)
        df["query_group"] = "macro" if q in D3_QUERIES_MACRO else "industry"
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{i}/{len(queries)}] {q}: {len(df)}건, 누적 API 호출 {api.calls}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- D5
def collect_prices(tag_map: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """태그 대표 종목의 일별 종가(long format)."""
    import FinanceDataReader as fdr
    rows = []
    for code, name in tag_map[["code", "name"]].drop_duplicates().itertuples(index=False):
        try:
            p = fdr.DataReader(code, start, end)[["Close"]].reset_index()
        except Exception as e:  # 상장폐지·거래정지 등
            print("[가격 실패]", name, e)
            continue
        p["code"], p["name"] = code, name
        rows.append(p)
        time.sleep(0.2)
    return pd.concat(rows, ignore_index=True).rename(columns={"Date": "date", "Close": "close"})


def build_tag_index(prices: pd.DataFrame, tag_map: pd.DataFrame) -> pd.DataFrame:
    """대표 종목 동일가중 일간 수익률로 태그 지수를 만든다 (KRX 업종 지수 API가 로그인 필요해 대체)."""
    px = prices.pivot_table(index="date", columns="code", values="close").sort_index()
    ret = px.pct_change(fill_method=None)
    # 가격제한폭(±30%)을 넘는 변동은 액면분할·병합 등 수정주가 미반영으로 보고 0으로 둔다
    ret = ret.mask(ret.abs() > 0.3, 0.0)
    out = {}
    for tag, g in tag_map.groupby("tag"):
        codes = [c for c in g.code if c in ret.columns]
        out[tag] = ret[codes].mean(axis=1)
    idx = pd.DataFrame(out)
    return (1 + idx.fillna(0)).cumprod()
