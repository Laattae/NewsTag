"""수집 실행 스크립트 (노트북 01_collect와 같은 함수를 터미널에서 오래 돌릴 때 사용).

    python scripts/collect.py d2   # 대표 종목 뉴스 (약한 라벨)
    python scripts/collect.py d3   # 종목명 없는 경제 뉴스 후보
중단돼도 다시 실행하면 이어서 받는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src.config import RAW
from src.crawler import NaverNewsAPI, collect_general_news, collect_stock_news

api = NaverNewsAPI()
if sys.argv[1] == "d2":
    collect_stock_news(api, pd.read_csv(RAW / "tag_map.csv", dtype={"code": str}))
else:
    collect_general_news(api)
print("DONE, API 호출", api.calls)
