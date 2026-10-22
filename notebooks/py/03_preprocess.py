# %% [markdown]
# # 03. 전처리 — 정제 · 종목명 마스킹 · 형태소 분석 · 중복 제거 · 분할 · 골드셋 샘플링
#
# 핵심은 **종목명 마스킹**이다. 학습 기사는 전부 제목에 종목명이 있으므로, 가리지 않으면
# 모델이 "하이닉스 → 반도체"만 외우고 종목명 없는 기사에서는 아무것도 못 맞힌다.

# %%
import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
from src.config import RAW, PROCESSED, GOLD, TAGS, MASK_TOKEN
from src.crawler import load_krx_listing
from src.dataset import build_weak_labels, dedupe, cap_per_tag, time_split, article_key, label_matrix
from src.masking import build_company_dict, CompanyMasker, unmasked_proper_nouns, default_masker
from src.config import EXTRA_MASK_NAMES
from src.text import clean, lead_sentences, tokenize_many
pd.set_option("display.max_colwidth", 120)

CAP_PER_TAG = 2500   # 태그별 상한: 흔한 태그가 학습을 지배하지 않도록
SEED = 42

d2_raw = pd.read_csv(RAW / "d2_stock_news_raw.csv", dtype={"code": str})
d3_raw = pd.read_csv(RAW / "d3_general_news_raw.csv")
listing = load_krx_listing()

# %% [markdown]
# ## 1. 정제
# HTML 태그·엔티티(API 단계에서 제거), 기자명·이메일, `[사진]`, `(서울=연합뉴스)` 머리말, 저작권 문구, URL 제거.
# 입력 텍스트는 **제목 + 요약 앞 4문장**(요약은 대부분 1~3문장이라 사실상 전체).

# %%
def prepare(df):
    df = df.copy()
    df["title_c"] = df.title.map(clean)
    df["desc_c"] = df.description.map(clean).map(lambda s: lead_sentences(s, 4))
    df["text_raw"] = df.title_c + " . " + df.desc_c
    df["pub_date"] = pd.to_datetime(df.pub_date)
    return df

ex = d2_raw.description.dropna()
ex = ex[ex.str.contains("기자|=|@|사진", regex=True)].head(3)
pd.DataFrame({"원문": ex.values, "정제 후": ex.map(clean).values})

# %% [markdown]
# ## 2. 약한 라벨 + 중복 제거
# - 제목에 종목명/약칭이 있는 행만 → 같은 URL은 병합, 태그는 합집합 (02와 동일 규칙)
# - 같은 URL, 그리고 정규화한 제목 앞 25자가 같은 기사(통신사 재전송·[속보] 머리말 차이) 제거

# %%
START = "2025-10-20"  # 수집 기간: 최근 12개월 (검색 API의 sim 정렬이 오래된 기사도 섞어 줌)
weak = prepare(build_weak_labels(d2_raw, title_filter=True))
print(f"기간 필터 전 {len(weak):,}건 (가장 오래된 기사 {weak.pub_date.min():%Y-%m-%d})")
weak = weak[weak.pub_date >= START]
n0 = len(weak)
weak = dedupe(weak)
print(f"약한 라벨 기사 {n0:,} → 중복 제거 후 {len(weak):,}")

# 비교 실험 E5용: 제목 필터 없이 만든 데이터
weak_nf = prepare(build_weak_labels(d2_raw, title_filter=False))
weak_nf = dedupe(weak_nf[weak_nf.pub_date >= START])
print(f"(E5) 제목 필터 없음: {len(weak_nf):,}")

# %% [markdown]
# ## 3. 종목명 마스킹
# KRX 전체 상장사명(우선주 표기 제거) + 약칭·영문명 + 그룹명·해외 빅테크명을 사전으로 만들고 `[기업]`으로 바꾼다.
# 일반 명사와 겹치는 회사명(대상, 한국, 동방, 태양 …)은 사전에서 뺀다.
#
# ### 3-1. 1차 사전으로 마스킹 → 남은 고유명사 확인

# %%
names = build_company_dict(listing)
masker = CompanyMasker(names)
print("사전 크기:", len(names))
sample = weak.sample(3000, random_state=SEED)
masked_sample = sample.text_raw.map(masker.mask)
left = unmasked_proper_nouns(masked_sample, top=60)
print(", ".join(f"{w}({c})" for w, c in left.values))

# %% [markdown]
# ### 3-2. 사전 보강 (사람이 고름)
# 위 표에서 **기업을 가리키는 고유명사**만 골라 추가한다. 지명·제품명·인명·기관명(정부 부처, 한국은행 등)은 남긴다.
# 제품·브랜드명(예: 갤럭시, 아이폰)은 산업 신호이므로 가리지 않는다.

# %%
# 고른 목록은 config.EXTRA_MASK_NAMES에 있다 (학습·적용·골드셋 평가에서 같은 마스커를 쓰도록)
print(EXTRA_MASK_NAMES)
masker = default_masker()
left2 = unmasked_proper_nouns(sample.text_raw.map(masker.mask), top=40)
print(", ".join(f"{w}({c})" for w, c in left2.values))

# %%
weak["text"] = weak.text_raw.map(masker.mask)
weak["title_m"] = weak.title_c.map(masker.mask)
weak_nf["text"] = weak_nf.text_raw.map(masker.mask)
print("기사당 마스크 토큰 수 평균:", weak.text.str.count(r"\[기업\]").mean().round(2))
weak[["text_raw", "text"]].head(5)

# %% [markdown]
# ## 4. 태그별 상한 적용 (불균형 완화)
# 태그별로 최대 `CAP_PER_TAG`건. 희귀 태그 기사부터 확보한 뒤 흔한 태그를 채운다(다중 라벨 기사 때문에
# 흔한 태그는 상한을 약간 넘을 수 있다).

# %%
weak_c = cap_per_tag(weak, CAP_PER_TAG, seed=SEED)
weak_nf_c = cap_per_tag(weak_nf, CAP_PER_TAG, seed=SEED)
pd.DataFrame({"상한 전": pd.Series(label_matrix(weak.tag_list).sum(0), TAGS),
              "상한 후": pd.Series(label_matrix(weak_c.tag_list).sum(0), TAGS),
              "(E5) 필터 없음": pd.Series(label_matrix(weak_nf_c.tag_list).sum(0), TAGS)})

# %% [markdown]
# ## 5. 형태소 분석 + 불용어 제거
# `kiwipiepy`로 명사(NNG/NNP), 외국어(SL), 한자(SH), 어근(XR), 동사·형용사 어간(+'다')을 남긴다.
# 불용어: 일반 불용어 + 경제 뉴스 상투어(기자, 오늘, 전년, 대비, 밝히다 …) + 언론사명.
# 실험용으로 (a) 마스킹 O 본문, (b) 마스킹 X 본문(E2), (c) 제목만(E4) 세 가지를 만든다.

# %%
for df in (weak_c, weak_nf_c):
    df["tokens"] = tokenize_many(df.text.tolist())
weak_c["tokens_nomask"] = tokenize_many(weak_c.text_raw.tolist())
weak_c["tokens_title"] = tokenize_many(weak_c.title_m.tolist())
weak_c["title_raw_tokens"] = tokenize_many(weak_c.title_c.tolist())
weak_c[["text", "tokens"]].head(5)

# %%
for df in (weak_c, weak_nf_c):
    df["log_len"] = np.log1p(df.text.str.len())

# %% [markdown]
# ## 6. 시간순 분할 (앞 80% 학습 / 뒤 20% 검증)

# %%
train, valid = time_split(weak_c, 0.8)
print(f"학습 {len(train):,}건 ({train.pub_date.min():%Y-%m-%d} ~ {train.pub_date.max():%Y-%m-%d})")
print(f"검증 {len(valid):,}건 ({valid.pub_date.min():%Y-%m-%d} ~ {valid.pub_date.max():%Y-%m-%d})")
pd.DataFrame({"train": label_matrix(train.tag_list).mean(0), "valid": label_matrix(valid.tag_list).mean(0)},
             index=TAGS).round(3).T

# %% [markdown]
# 검증셋에 태그 비율이 학습셋과 크게 다르면 (최근 이슈가 몰린 태그), 약한 라벨 검증 점수가 흔들릴 수 있다.
# 이 차이는 시간순 분할이 일부러 드러내는 '미래 분포 변화'다.

# %%
cols = ["key", "pub_date", "press", "title", "description", "title_c", "text_raw", "text", "title_m", "stocks",
        "tag_list", "n_tags", "tokens", "tokens_nomask", "tokens_title", "title_raw_tokens", "log_len"]
train[cols].to_pickle(PROCESSED / "weak_train.pkl")
valid[cols].to_pickle(PROCESSED / "weak_valid.pkl")
tr_nf, va_nf = time_split(weak_nf_c, 0.8)
tr_nf.to_pickle(PROCESSED / "weak_nofilter_train.pkl")
va_nf.to_pickle(PROCESSED / "weak_nofilter_valid.pkl")

# %% [markdown]
# ## 7. D3: 종목명 없는 경제 뉴스
# 1. 정제, URL·제목 중복 제거
# 2. **종목명 사전(마스킹 사전과 동일)에 하나도 안 걸리는 기사만** 남김
# 3. 사전에 걸리는 비율이 높은 이름을 확인해, 일반 명사와 겹치는 회사명이 필터를 망가뜨리지 않는지 점검

# %%
d3 = prepare(d3_raw)
d3 = d3[d3.pub_date >= START]
d3["key"] = article_key(d3)
d3 = dedupe(d3)
# (D2 기사는 제목에 종목명이 있으므로 아래 '종목명 없음' 필터에서 자동으로 빠진다)
hits = d3.text_raw.map(masker.find)
print(f"D3 중복 제거 후 {len(d3):,}건, 기업명 포함 {hits.str.len().gt(0).mean():.1%}")
top_hits = pd.Series([h for hs in hits for h in set(hs)]).value_counts().head(40)
print(", ".join(f"{w}({c})" for w, c in top_hits.items()))

# %% [markdown]
# 위 목록은 대부분 실제 기업 지칭(그룹명·빅테크 등)이다. 일반 명사로 오인된 회사명이 있으면
# `config.AMBIGUOUS_NAMES`에 추가하고 다시 실행한다 (1차 실행에서 확인 후 반영했다).

# %%
d3_clean = d3[hits.str.len() == 0].copy()
d3_clean["text"] = d3_clean.text_raw.map(masker.mask)  # 기업명은 없지만 경영자 이름 등은 학습과 같은 방식으로 가린다
d3_clean["tokens"] = tokenize_many(d3_clean.text.tolist())
d3_clean["log_len"] = np.log1p(d3_clean.text.str.len())
d3_clean = d3_clean[d3_clean.tokens.str.split().str.len() >= 5].reset_index(drop=True)
import hashlib
d3_clean["doc_id"] = "d3_" + d3_clean.key.map(lambda k: hashlib.md5(k.encode()).hexdigest()[:10])  # 사전을 고쳐도 ID 유지
print(f"종목명 없는 기사: {len(d3_clean):,}건")
print(d3_clean.groupby("query_group").size())
d3_clean[["doc_id", "pub_date", "query", "title_c", "desc_c", "text", "tokens", "press", "log_len", "query_group"]] \
    .to_pickle(PROCESSED / "d3_clean.pkl")
d3_clean.sample(5, random_state=SEED)[["query", "title_c", "desc_c"]]

# %% [markdown]
# ## 8. 골드 평가셋 샘플링 (D4)
# D3에서 무작위 300건 → 검증 100 + 테스트 200. 라벨링 기준: **"이 기사 내용 때문에 해당 산업 종목의 주가나
# 실적에 영향이 갈 수 있는가"** 를 기준으로 0~3개 태그, 해당 없으면 '없음'.
# 라벨링 파일: `data/gold/gold_labels.csv` = 표본 + `tags` 열(`|`로 구분, 빈칸 = 없음). 50건은 이중 라벨링용.
# 표본 파일에 기사 텍스트를 같이 저장해, 이후 사전을 보강해 D3 필터 결과가 바뀌어도 골드셋은 고정된다.

# %%
gold_path = GOLD / "gold_sample.csv"
if not gold_path.exists():  # 한 번 뽑은 표본은 고정 (라벨링 후 바뀌면 안 됨)
    g = d3_clean.sample(300, random_state=SEED).reset_index(drop=True)
    g["split"] = ["valid"] * 100 + ["test"] * 200
    g["double_label"] = [i < 50 for i in range(300)]
    g[["doc_id", "split", "double_label", "pub_date", "query", "press", "title_c", "desc_c", "text_raw"]].to_csv(
        gold_path, index=False, encoding="utf-8-sig")
gold_sample = pd.read_csv(gold_path)
gold_sample.split.value_counts()

# %% [markdown]
# ## 이 단계에서 알게 된 것
#
# - **기간·중복**: 약한 라벨 기사 16.2만 건 중 최근 12개월(2025-10-20~) 기사 12.6만 건 → 통신사 재전송·[속보] 중복을
#   제거하니 11.5만 건(약 9% 중복). 태그별 상한 2,500건을 적용해 학습 2.87만 / 약한 검증 7.2천 건으로 줄였다.
#   검증셋은 시간순 마지막 20%라 2026-10-07 ~ 10-20, 약 2주다.
# - **마스킹**: 기사당 평균 3.4개의 `[기업]` 토큰. 1차 사전만으로는 '하닉·삼전닉스' 같은 은어, 비상장 계열사
#   (신한은행, 롯데면세점 등), 해외 기업명이 남았다. 그룹 접두어 + 계열사 접미사 규칙(`한화투자증권`, `DB인재개발원`)과
#   추가 사전으로 보강한 뒤 남은 고유명사는 지명·기술 용어(AI, LFP, HVDC, FDA)·언론사명이 대부분이다.
# - **D3 필터**: 경제 기사 10.5만 건(중복 제거 후) 중 32.7%가 어떤 기업명이든 언급한다. 이를 빼고 7.0만 건이 남았다
#   (거시 검색어 3.8만, 산업 검색어 3.3만). 필터에 많이 걸린 이름은 삼성전자·SK하이닉스·현대차·그룹명·TSMC·엔비디아 순으로,
#   일반 경제 기사도 대형 기업을 통해 산업을 이야기하는 경우가 많다는 뜻이다.
# - **모호한 회사명**: '상보'(기사 표기), '배럴', '채비', '서산', 'DSR'(대출 규제), '이지스'(이지스함) 같은 이름이
#   D3 필터를 망가뜨리고 있어 모호 목록에 넣었다. 사전 매칭은 정밀도와 재현율 사이 수작업 조정이 계속 필요하다.
# - **골드셋**: D3에서 무작위 300건(검증 100 / 테스트 200)을 뽑아 라벨링했다. 표본 파일에 텍스트를 함께 저장해,
#   이후 사전을 고쳐 D3 필터 결과가 바뀌어도 골드셋은 그대로 유지된다.
