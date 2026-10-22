"""정제·형태소 분석·불용어 제거."""
from __future__ import annotations

import re

from .config import MASK_TOKEN, PERSON_TOKEN

_CLEAN_PATTERNS = [
    r"<[^>]+>",                                         # HTML 태그
    r"\[(?:사진|그래픽|포토|영상|표|자료|인포그래픽|이미지)[^\]]*\]",
    r"\((?:사진|그래픽|자료)\s*=?[^)]*\)",
    r"(?:사진|그래픽|자료|제공)\s*=\s*\S+(?:\s*제공)?",          # 사진=유한양행 제공
    r"\|[^|]{1,40}\|",                                     # |중앙이코노미뉴스 최경서 기자|
    r"[\w.+-]+@[\w-]+\.[\w.]+",                         # 이메일
    r"\([^()]{0,20}=\s*[^()]{0,20}\)\s*[가-힣]{2,4}\s*(?:기자|특파원)\s*=?",  # (서울=연합뉴스) 홍길동 기자 =
    r"[가-힣]{2,4}\s*(?:기자|특파원|선임기자|객원기자)(?:\s*=|\s*입력|\s|$)",
    r"[▶■□◆◇●○※☞▲△▼▽ⓒ©]",
    r"(?:무단\s*전재|재배포\s*금지|저작권자).{0,30}",
    r"https?://\S+",
]
_CLEAN_RE = re.compile("|".join(_CLEAN_PATTERNS))
_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", _CLEAN_RE.sub(" ", text)).strip()


def lead_sentences(text: str, n: int = 4) -> str:
    """본문 앞 n문장. (검색 API 요약은 이미 1~3문장이라 대부분 그대로 남는다)"""
    sents = re.split(r"(?<=[.!?다])\s+", text)
    return " ".join(sents[:n])


STOPWORDS = set("""
기자 뉴스 오늘 어제 내일 전년 대비 지난해 올해 이번 최근 관련 대한 위해 통해 따르 따라 경우 가운데 이후 이전
때문 정도 수준 이상 이하 가량 이날 당시 현재 오전 오후 시간 하루 주간 기간 년 월 일 억 만 천 원 달러 %
등 및 또 또한 그리고 하지만 그러나 한편 한다 했다 있다 없다 되다 하다 이다 아니다 같다 보다 밝히 말하 전하
설명 강조 예정 계획 가능 것 수 바 때 중 측 곳 데 점 개 명 건 번 차 회 위 말 뒤 앞 안 밖 간 내 외 전 후
밝히다 말하다 전하다 설명하다 강조하다 나타내다 보이다 들다 나오다 받다 주다 오르다 내리다 가다 오다 되다 하다 있다 없다 같다
따르다 위하다 대하다 지나다 통하다 나서다 열리다 늘다 이르다 갖다 알다 보다 놓다 앞두다
이투데이 아주경제 뉴데일리 뉴스투데이 아시아타임즈 머니투데이방송 MTN 뉴스핌 데일리안 서울신문 조선비즈 비즈니스포스트 더구루 디지털타임스 전자신문 ZDNET
연합뉴스 뉴시스 뉴스1 머니투데이 이데일리 서울경제 한국경제 매일경제 헤럴드경제 아시아경제 파이낸셜뉴스
""".split())

_KEEP_TAGS = {"NNG", "NNP", "SL", "SH", "XR", "VV", "VA", "SN"}
_kiwi = None


def get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi(num_workers=-1)
    return _kiwi


def tokenize_many(texts: list[str], stopwords: set[str] = STOPWORDS) -> list[str]:
    """형태소 분석 후 '토큰 공백 토큰' 문자열 리스트. 마스크 토큰은 한 단어 `[기업]`으로 보존."""
    kiwi = get_kiwi()
    placeholder = "XMASKX"  # 영문 한 단어(SL)로 두면 형태소 분석기가 쪼개지 않는다
    prepped = [(t or "").replace(MASK_TOKEN, f" {placeholder} ").replace(PERSON_TOKEN, " XPERSONX ") for t in texts]
    out = []
    for res in kiwi.tokenize(prepped):
        toks = []
        for tk in res:
            if tk.tag not in _KEEP_TAGS:
                continue
            form = tk.form
            if form.upper() == placeholder:
                toks.append(MASK_TOKEN)
                continue
            if form.upper() == "XPERSONX":
                toks.append(PERSON_TOKEN)
                continue
            if tk.tag in ("VV", "VA"):
                form = form + "다"
            elif tk.tag == "SL":
                form = form.upper()
            elif tk.tag == "SN":
                continue  # 숫자 단독은 버린다 (단위와 붙은 'HBM4' 등은 SL로 남는다)
            if form in stopwords or len(form) < 2 and tk.tag not in ("SL", "SH"):
                continue
            toks.append(form)
        out.append(" ".join(toks))
    return out
