# %% [markdown]
# # 01. 데이터 수집 (D1 · D2 · D3 · D5)
#
# | ID | 데이터 | 방법 |
# |---|---|---|
# | D1 | 태그별 대표 종목 + KRX 전 종목 목록 | `FinanceDataReader` KRX-DESC, 직접 만든 `tag_map.csv` |
# | D2 | 대표 종목 뉴스 (약한 라벨) | 네이버 검색 API(NAVER API HUB) — 종목명 검색 |
# | D3 | 종목명 없는 경제 뉴스 후보 | 네이버 검색 API — 거시·산업 키워드 검색 |
# | D5 | 태그 지수 | 대표 종목 일별 종가(FinanceDataReader) → 동일가중 지수 |
#
# **수집 경로를 명세에서 바꾼 이유**
# - `finance.naver.com`, `news.naver.com`, `m.stock.naver.com`의 robots.txt가 모두 `Disallow: /`이고,
#   명세가 가정한 종목 뉴스 페이지(`item/news_news.naver`)는 410(Gone)을 반환한다 (2026-10-20 확인).
# - 그래서 공식 **네이버 검색 Open API**를 쓴다. 2026년에 NAVER API HUB(네이버 클라우드)로 이관되어
#   엔드포인트가 `naverapihub.apigw.ntruss.com/search/v1/news`, 헤더가 `X-NCP-APIGW-API-KEY(-ID)`로 바뀌었다.
# - API는 **제목 + 요약(본문 앞 1~2문장)**, 원문 링크, 발행시각을 준다. 본문 전체는 받지 않으므로
#   명세의 입력 "제목+본문 앞 3~5문장"은 "제목+요약"으로 대체된다. 원문은 재배포하지 않고 분석용으로만 쓴다.
# - KRX 업종 지수(pykrx)는 2026년 현재 KRX 로그인이 필요해, 대표 종목 동일가중 지수로 대체했다(D5).
#
# 수집은 오래 걸리므로 터미널에서 `python scripts/collect.py d2|d3`로 돌렸다. 아래 `RUN_COLLECT=True`면 노트북에서 직접 받는다.

# %%
import sys
sys.path.insert(0, "..")
import pandas as pd
from src.config import RAW, TAG_STOCKS, EXCLUDED_CONGLOMERATES
from src import crawler

RUN_COLLECT = False
pd.set_option("display.max_colwidth", 80)

# %% [markdown]
# ## D1. 태그 체계와 대표 종목
# 태그 15개 × 대표 종목 8~14개. 명세 표를 출발점으로, KRX 상장명(2026-10 기준)과 업종(Industry)을 보고 확정했다.
# - 사명 변경 반영: LIG넥스원→LIG디펜스앤에어로스페이스, 엔씨소프트→NC, 티웨이항공→트리니티항공,
#   HDC현대산업개발→IPARK현대산업개발, GC녹십자→녹십자 (옛 이름은 약칭으로 남겨 제목 필터·마스킹에 사용)
# - HD현대미포는 HD현대중공업에 합병되어 제외
# - 다산업 대기업(삼성전자, LG전자, SK, LG화학 등)은 라벨 노이즈가 커서 제외
# - 두 산업에 걸친 종목은 태그 2개: 풍산(철강·비철+방산), POSCO홀딩스·SK이노베이션(+2차전지)

# %%
listing = crawler.load_krx_listing(refresh=RUN_COLLECT)
tag_map = crawler.build_tag_map(listing)
tag_map.to_csv(RAW / "tag_map.csv", index=False, encoding="utf-8-sig")
print("KRX 상장사:", len(listing), "| 대표 종목(행):", len(tag_map), "| 고유 종목:", tag_map.code.nunique())
print("제외한 대기업:", EXCLUDED_CONGLOMERATES)
tag_map.groupby("tag").agg(n=("name", "size"), stocks=("name", ", ".join))

# %% [markdown]
# 대표 종목의 KRX 표준산업분류(Industry)를 보면 태그와 대체로 일치한다. 일치하지 않는 종목(예: 지주회사로
# 분류된 고려아연·POSCO홀딩스)은 실제 사업 내용을 보고 유지했다.

# %%
tag_map.groupby("tag")["Industry"].agg(lambda s: s.value_counts().head(3).to_dict())

# %% [markdown]
# ## D2 · D3. 네이버 검색 API
# - D2: 종목마다 `종목명`(+3글자 이상 약칭 1개)으로 검색, 정렬 `date`/`sim` 각각 최대 1,000건
# - 제목에 종목명·약칭이 있는지(`title_has_name`)를 같이 저장 → 02에서 약한 라벨 규칙 적용, E5에서 필터 효과 비교
# - D3: 거시 키워드 38개 + 산업 키워드 46개. 산업 키워드만 쓰면 '관련 테마 없음' 기사가 너무 적어진다.

# %%
if RUN_COLLECT:
    api = crawler.NaverNewsAPI()
    crawler.collect_stock_news(api, tag_map)
    crawler.collect_general_news(api)

d2_raw = pd.read_csv(RAW / "d2_stock_news_raw.csv", dtype={"code": str})
d3_raw = pd.read_csv(RAW / "d3_general_news_raw.csv")
print(f"D2 원본 {len(d2_raw):,}행 (종목 {d2_raw.code.nunique()}개), 제목 필터 통과 {d2_raw.title_has_name.mean():.1%}")
print(f"D3 원본 {len(d3_raw):,}행 (검색어 {d3_raw['query'].nunique()}개)")
d2_raw.sample(5, random_state=0)[["stock", "tags", "title", "title_has_name", "pub_date", "press"]]

# %%
per_stock = d2_raw.groupby(["stock", "tags"]).agg(rows=("title", "size"), title_hit=("title_has_name", "mean"),
                                                  oldest=("pub_date", "min")).reset_index()
per_stock.sort_values("title_hit").head(10)

# %% [markdown]
# 제목 포함 비율이 낮은 종목은 종목명이 일반 단어와 겹치거나(검색 결과에 다른 뜻이 섞임),
# 시황 나열 기사에만 이름이 언급되는 경우다. 이런 기사가 약한 라벨 노이즈의 주된 원천이라 제목 필터로 거른다.

# %%
d2_raw["pub_date"] = pd.to_datetime(d2_raw["pub_date"])
d3_raw["pub_date"] = pd.to_datetime(d3_raw["pub_date"])
print("D2 기간:", d2_raw.pub_date.min().date(), "~", d2_raw.pub_date.max().date())
print("D3 기간:", d3_raw.pub_date.min().date(), "~", d3_raw.pub_date.max().date())
pd.concat({"D2": d2_raw.pub_date.dt.to_period("M").value_counts().sort_index(),
           "D3": d3_raw.pub_date.dt.to_period("M").value_counts().sort_index()}, axis=1).fillna(0).astype(int).tail(14)

# %% [markdown]
# ## D5. 태그 지수
# 대표 종목 일별 종가 → 일간 수익률 동일가중 평균 → 누적 지수. ±30%(가격제한폭)를 넘는 일간 변동은
# 액면분할 등 수정주가 미반영으로 보고 0으로 처리했다.

# %%
if RUN_COLLECT or not (RAW / "d5_prices.csv").exists():
    prices = crawler.collect_prices(tag_map, "2025-10-01")
    prices.to_csv(RAW / "d5_prices.csv", index=False, encoding="utf-8-sig")
prices = pd.read_csv(RAW / "d5_prices.csv", dtype={"code": str}, parse_dates=["date"])
tag_index = crawler.build_tag_index(prices, tag_map)
tag_index.to_csv(RAW / "d5_tag_index.csv", encoding="utf-8-sig")
tag_index.iloc[-1].sort_values(ascending=False).round(2).rename("2025-10 대비 누적")

# %% [markdown]
# ## 이 단계에서 알게 된 것
# - 네이버 금융·뉴스 웹페이지는 robots.txt로 크롤링이 금지돼 있어 공식 검색 API(NAVER API HUB)로 경로를 바꿨다.
#   그 대가로 본문 대신 요약(1~2문장, 평균 160자)만 쓴다. API 호출은 D2 4,198회 + D3 1,666회로 하루 한도(2.5만) 안이다.
# - D2 원본 41.9만 행(161종목), D3 원본 16.7만 행(검색어 84개). 종목 검색 결과 중 제목에 종목명이 있는 행은 56%다.
#   HD한국조선해양(6%), SBS(12%)처럼 이름이 다른 기사(그룹사 기사, 방송 프로그램)에 자주 섞이는 종목이 낮다.
# - 검색 API는 정렬별로 최대 약 1,000건만 주므로 기사 시점이 최근에 몰린다(2026년 9~10월이 대부분).
#   `sim` 정렬은 오래된 기사(최대 2000년대)도 섞어 주므로 03에서 최근 12개월로 자른다.
# - 태그 지수(D5): 지난 1년 반도체(×2.27)·건설(×1.61)·원전·전력기기(×1.42)가 오르고, 엔터(×0.52)·조선(×0.56)·
#   항공·여행(×0.66)이 내렸다. 구성 종목들이 같은 방향으로 움직여 수정주가 문제가 아님을 확인했다.
