"""종목명 사전 생성과 마스킹.

KRX 전체 상장사명 + 약칭·영문명 + 그룹명/해외 빅테크명을 사전으로 만들고, 기사에서 찾아 `[기업]`으로 바꾼다.
대표 종목만이 아니라 전체 상장사를 가려야 모델이 '종목명 → 태그'를 외우지 못한다.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .config import (ALIASES, AMBIGUOUS_NAMES, CORP_SUFFIXES, EXTRA_MASK_NAMES, GROUP_NAMES, GROUP_PREFIXES,
                     MASK_TOKEN, PERSON_TOKEN)

_PARTICLES = "은|는|이|가|을|를|의|에|와|과|도|로|만|측|과의|에서|으로|이다|처럼|보다"
_PARTICLE_LIST = sorted(_PARTICLES.split("|") + ["에게", "까지", "부터", "마저", "조차", "이나", "나", "랑", "이며", "였다",
                                                  "이었다", "에선", "에도", "와의", "으로는", "로는", "은요"],
                        key=len, reverse=True)
# 이름 확장으로 일반 단어가 되는 경우 (03에서 확장 매칭 상위 빈도를 보고 찾음: '하이브'+'리드')
_EXT_BLOCK = ("하이브리드",)
_TITLES = "회장|부회장|명예회장|사장|부사장|대표이사|대표|의장|행장|총재|창업자|창업주|총수|최고경영자|CEO"


def _clean_listing_name(n: str) -> str:
    # '삼성전자우', '현대차2우B' 같은 우선주 표기는 본명으로 줄여 사전에 넣는다
    return re.sub(r"(\d?우[B]?|\(전환\))$", "", n).strip()


def build_company_dict(listing: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    names = set()
    for n in listing["Name"].dropna():
        names.add(n)
        names.add(_clean_listing_name(n))
    for k, v in ALIASES.items():
        names.add(k)
        names.update(v)
    names.update(GROUP_NAMES)
    names.update(extra or [])
    amb = set(AMBIGUOUS_NAMES)
    # 대표 종목 본명과 약칭은 모호해도 살린다 (ALIASES에 명시한 것은 의도한 것)
    keep = set(ALIASES) | {a for v in ALIASES.values() for a in v}
    names = {n for n in names if n and len(n) >= 2 and (n not in amb or n in keep)}
    # 그룹명 중 2글자 영문(SK, LG 등)은 모호 목록에 있어도 그룹 지칭이 분명하므로 영문 단어 경계로 매칭
    names.update(g for g in GROUP_NAMES if re.fullmatch(r"[A-Z]{2}", g))
    return sorted(names, key=len, reverse=True)


class CompanyMasker:
    def __init__(self, names: list[str], token: str = MASK_TOKEN):
        self.token = token
        self.names = names
        ascii_names = [n for n in names if re.fullmatch(r"[A-Za-z0-9 .&\-]+", n)]
        kor_long = [n for n in names if n not in ascii_names and len(n) >= 3]
        kor_short = [n for n in names if n not in ascii_names and len(n) < 3]

        def alt(xs):
            return "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True))

        # (?P<ext>…): 이름 뒤에 붙은 글자까지 한 단어로 보고 함께 가린다 ('삼성바이오'+'로직스', 'NC'+'다이노스').
        # 붙은 부분 끝의 조사는 _sub에서 되돌려 놓는다. 1차 실행에서 이름 조각('로직스', '팜')이 남아 종목 식별자로 쓰였다.
        ext = r"(?P<ext>[가-힣A-Za-z0-9&]*)"
        parts = [  # 비상장사 표기: ㈜에이치이티, (주)○○, ○○㈜
            r"(?:㈜|\(주\))\s?[가-힣A-Za-z0-9&]+|[가-힣A-Za-z0-9&]+(?:㈜|\(주\))",
            # 그룹 접두어 + (짧은 이름) + 계열사 접미사: 한화투자증권, 신한은행, 롯데면세점 …
            rf"(?<![가-힣A-Za-z0-9])(?:{alt(GROUP_PREFIXES)})[가-힣A-Za-z&]{{0,6}}?(?:{alt(CORP_SUFFIXES)}){ext}"]
        if kor_long:  # 3글자 이상 국문(혼합 포함)명: 앞 글자만 확인
            parts.append(rf"(?<![가-힣A-Za-z0-9])(?:{alt(kor_long)}){ext}".replace("(?P<ext>", "(?P<ext2>"))
        if kor_short:  # 2글자 국문명: 뒤에 한글이 오면 조사일 때만 매칭 ('동성애' 같은 오탐 방지), 확장 없음
            parts.append(rf"(?<![가-힣A-Za-z0-9])(?:{alt(kor_short)})(?=[^가-힣]|$|(?:{_PARTICLES}))")
        if ascii_names:  # 영문명: 뒤에 영문·숫자가 오면 다른 단어. 한글이 붙으면 한 단어로 확장
            parts.append(rf"(?<![A-Za-z0-9])(?:{alt(ascii_names)})(?![A-Za-z0-9])(?P<ext3>[가-힣&]*)")
        self.pattern = re.compile("|".join(parts))
        # 경영자 이름: '곽동신 회장', '진옥동 [기업] 회장' 형태의 3글자 이름 (회사명을 가려도 인명으로 종목이 드러남)
        self.person_pattern = re.compile(
            rf"(?<![가-힣])[가-힣]{{3}}(?<![의은는을를에와과도로])"  # 조사로 끝나는 단어('회사의')는 제외
            rf"(?=\s*(?:{re.escape(token)}\s*)?(?:(?:신임|전|현|공동|각자|초대)\s*)?(?:{_TITLES})(?=[^가-힣]|$|(?:{_PARTICLES}|님|직|으로서|로서)))")
        self._token_re = re.compile(rf"(?:{re.escape(token)}\s*){{2,}}")

    def _sub(self, m: re.Match) -> str:
        ext = m.groupdict().get("ext") or m.groupdict().get("ext2") or m.groupdict().get("ext3") or ""
        if ext and m.group(0).startswith(_EXT_BLOCK):
            return m.group(0)
        for p in _PARTICLE_LIST:  # 붙은 부분 끝의 조사는 남긴다
            if ext.endswith(p):
                return self.token + p
        return self.token

    def mask(self, text: str, people: bool = True) -> str:
        if not isinstance(text, str):
            return ""
        out = self.pattern.sub(self._sub, text)
        if people:
            out = self.person_pattern.sub(PERSON_TOKEN, out)
        # '[기업][기업]'처럼 연달아 붙은 토큰은 하나로
        return self._token_re.sub(self.token + " ", out)

    def find(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        return [m.group(0) for m in self.pattern.finditer(text) if self._sub(m) != m.group(0)]

    def has_company(self, text: str) -> bool:
        return bool(self.find(text))


_DEFAULT = None


def default_masker() -> "CompanyMasker":
    """KRX 목록 + 약칭·그룹명 + EXTRA_MASK_NAMES로 만든 공용 마스커 (학습·적용·골드셋 평가 공통)."""
    global _DEFAULT
    if _DEFAULT is None:
        from .crawler import load_krx_listing
        _DEFAULT = CompanyMasker(build_company_dict(load_krx_listing(), extra=EXTRA_MASK_NAMES))
    return _DEFAULT


def unmasked_proper_nouns(texts, kiwi=None, top: int = 100) -> pd.DataFrame:
    """마스킹 후에도 남은 고유명사(NNP) 상위 빈도. 사전 보강 후보를 사람이 고르기 위한 표."""
    from kiwipiepy import Kiwi
    kiwi = kiwi or Kiwi()
    cnt = Counter()
    for res in kiwi.tokenize(list(texts)):
        cnt.update(t.form for t in res if t.tag in ("NNP", "SL") and len(t.form) >= 2)
    return pd.DataFrame(cnt.most_common(top), columns=["word", "count"])
