"""피처 엔지니어링: 단어/문자 TF-IDF, 도메인 키워드 사전, 메타 피처."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 태그별 시드 키워드 (형태소 분석 결과 기준 표기).
# 1차: 도메인 지식으로 작성 → 04_modeling에서 D2 태그별 TF-IDF 상위 단어를 뽑아 사람이 골라 보강(SEED_ADDED).
SEED_KEYWORDS = {
    "반도체": ["반도체", "HBM", "파운드리", "D램", "디램", "낸드", "웨이퍼", "메모리", "칩", "팹", "EUV",
             "패키징", "후공정", "식각", "증착", "노광", "AI칩", "GPU", "DDR", "SSD", "공정", "수율",
             "장비", "소부장", "TC본더", "프로브"],
    "2차전지": ["배터리", "2차전지", "이차전지", "양극재", "음극재", "전해질", "전해액", "분리막", "리튬",
              "니켈", "코발트", "전구체", "셀", "ESS", "LFP", "NCM", "전고체", "에너지저장장치", "캐즘",
              "IRA", "충전", "광물"],
    "자동차": ["자동차", "완성차", "전기차", "차량", "신차", "판매량", "내연기관", "하이브리드", "SUV",
             "부품", "타이어", "자율주행", "모빌리티", "EV", "공장", "딜러", "리콜", "車"],
    "조선": ["조선", "선박", "수주", "LNG선", "컨테이너선", "유조선", "선가", "건조", "해양플랜트", "MRO",
           "함정", "조선소", "발주", "선주", "친환경선", "암모니아", "VLCC", "도크", "해운", "인도"],
    "철강·비철": ["철강", "강판", "열연", "냉연", "후판", "철근", "고로", "전기로", "아연", "구리", "니켈",
              "비철", "제련", "철광석", "강재", "형강", "스테인리스", "알루미늄", "희토류", "고철", "합금"],
    "원전·전력기기": ["원전", "원자력", "SMR", "원자로", "변압기", "전력망", "송전", "배전", "전력기기",
                "전선", "케이블", "초고압", "체코", "전력", "발전소", "가스터빈", "데이터센터", "수소",
                "전기요금", "한수원", "차단기", "그리드"],
    "방산": ["방산", "방위산업", "무기", "전차", "K9", "자주포", "미사일", "전투기", "국방", "방위", "군",
           "레드백", "천궁", "탄약", "수출계약", "나토", "NATO", "폴란드", "우크라이나", "중동", "드론", "KF"],
    "바이오·제약": ["바이오", "제약", "신약", "임상", "FDA", "의약품", "바이오시밀러", "항체", "치료제",
               "허가", "기술이전", "CDMO", "위탁생산", "항암", "비만", "백신", "약가", "식약처", "제네릭",
               "ADC", "환자", "파이프라인"],
    "인터넷·게임": ["게임", "신작", "플랫폼", "포털", "검색", "광고", "AI", "인공지능", "모바일", "PC",
               "MMORPG", "출시", "이용자", "앱", "메신저", "클라우드", "콘솔", "스팀", "구독", "커머스",
               "웹툰", "LLM"],
    "금융(은행·증권·보험)": ["은행", "증권", "보험", "금융", "대출", "예금", "금리", "이자", "순이자마진",
                      "NIM", "충당금", "배당", "자사주", "밸류업", "주주환원", "브로커리지", "IB", "보험료",
                      "손해율", "연체율", "금융지주", "가계대출", "증시", "거래대금"],
    "건설·부동산": ["건설", "부동산", "분양", "아파트", "주택", "재건축", "재개발", "수주", "시공",
               "PF", "미분양", "착공", "건축", "인프라", "플랜트", "SOC", "정비사업", "청약", "공사비",
               "집값", "입주"],
    "정유·화학": ["정유", "석유화학", "화학", "유가", "정제마진", "나프타", "에틸렌", "NCC", "원유",
              "휘발유", "경유", "납사", "합성수지", "폴리머", "스프레드", "구조조정", "석유", "OPEC",
              "플라스틱", "PVC", "윤활유"],
    "항공·여행": ["항공", "항공사", "여객", "노선", "운항", "LCC", "저비용항공사", "여행", "관광",
              "관광객", "공항", "면세", "호텔", "카지노", "항공권", "유류할증료", "인바운드", "아웃바운드",
              "단체관광", "비자", "탑승객", "항공기"],
    "화장품·소비재": ["화장품", "뷰티", "K뷰티", "스킨케어", "색조", "ODM", "면세점", "올리브영",
                "브랜드", "소비자", "수출", "아마존", "인디", "선크림", "마스크팩", "생활용품", "중국",
                "소비", "유통", "럭셔리"],
    "엔터·미디어": ["엔터", "K팝", "아이돌", "앨범", "콘서트", "투어", "팬덤", "음원", "드라마", "영화",
               "OTT", "넷플릭스", "콘텐츠", "방송", "제작", "아티스트", "데뷔", "음반", "공연", "미디어",
               "IP", "한한령"],
}
# 04_modeling 1절: D2 태그별 TF-IDF 차이 상위 단어(데이터 후보) 중 사람이 산업 고유 어휘만 고른 것
SEED_ADDED: dict[str, list[str]] = {
    "반도체": ["본더"],
    "2차전지": ["양극", "전지", "분리", "고체", "인산", "이차", "저장장치"],
    "자동차": ["OE", "램프", "레이싱", "랠리", "아반떼"],
    "조선": ["운반선", "상선", "선사", "조선업", "해양", "액화"],
    "철강·비철": ["제철소", "특수강"],
    "원전·전력기기": ["고압", "HVDC", "해저", "설비", "풍력"],
    "방산": ["우주", "무인", "체계", "대공", "무인기", "위성", "유도탄", "발사체"],
    "바이오·제약": ["물질", "후보", "제약사", "질환", "약물", "주사", "제형", "MRNA", "핵산", "접합", "승인"],
    "인터넷·게임": ["업데이트", "서버", "RPG", "흥행", "캐주얼", "TGS"],
    "금융(은행·증권·보험)": ["연금", "어음", "환원", "보험사", "자본"],
    "건설·부동산": ["공사", "구역", "정비", "지구", "시공사", "단지", "주거", "공구", "가구", "건설사", "발주", "토목"],
    "정유·화학": ["태양광", "고무", "폴리", "합성"],
    "항공·여행": ["국제선", "마일리지", "취항", "탑승", "승무원", "객실", "패키지", "예약"],
    "화장품·소비재": ["피부", "케어", "헤어", "스킨", "노화", "입점"],
    "엔터·미디어": ["티빙", "음악", "프로듀서", "MD"],
}


def seed_keywords() -> dict[str, list[str]]:
    return {t: sorted(set(SEED_KEYWORDS[t]) | set(SEED_ADDED.get(t, []))) for t in SEED_KEYWORDS}


class KeywordCounter(BaseEstimator, TransformerMixin):
    """태그별 시드 키워드 등장 횟수 → 태그당 1개 피처 (log1p). 입력은 공백으로 나뉜 토큰 문자열."""

    def __init__(self, keywords: dict[str, list[str]] | None = None, log: bool = True):
        self.keywords = keywords
        self.log = log

    def fit(self, X, y=None):
        kw = self.keywords or seed_keywords()
        self.tags_ = list(kw)
        self.sets_ = [set(w.upper() if w.isascii() else w for w in kw[t]) for t in self.tags_]
        return self

    def transform(self, X):
        X = pd.Series(X).fillna("")
        M = np.zeros((len(X), len(self.tags_)), dtype=np.float32)
        for i, doc in enumerate(X):
            toks = doc.split()
            for j, s in enumerate(self.sets_):
                M[i, j] = sum(t in s for t in toks)
        if self.log:
            M = np.log1p(M)
        return sparse.csr_matrix(M)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"kw_{t}" for t in self.tags_])


def make_features(word=True, char=True, keyword=True, meta=True, word_max_features=30000,
                  word_ngram=(1, 2), word_min_df=3, char_ngram=(2, 4), char_max_features=50000, keywords=None):
    """DataFrame(tokens, text, press, length) → 희소 피처 행렬. E3 실험에서 그룹을 켜고 끈다."""
    tr = []
    if word:
        tr.append(("word", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=word_ngram,
                                           min_df=word_min_df, max_features=word_max_features,
                                           sublinear_tf=True), "tokens"))
    if char:
        tr.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, min_df=3,
                                           max_features=char_max_features, sublinear_tf=True), "text"))
    if keyword:
        # 사전을 생성 시점에 객체 안에 담는다 (병렬 워커는 모듈 전역 SEED_ADDED를 공유하지 않음)
        tr.append(("kw", KeywordCounter(keywords=keywords or seed_keywords()), "tokens"))
    if meta:
        tr.append(("press", OneHotEncoder(handle_unknown="ignore", min_frequency=30), ["press"]))
        tr.append(("len", StandardScaler(), ["log_len"]))
    return ColumnTransformer(tr, sparse_threshold=1.0)
