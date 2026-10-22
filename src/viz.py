"""시각화 공통 스타일. 태그가 15개라 태그마다 색을 주지 않는다:
단일 계열은 한 색, 크기는 단일 색상 순차 팔레트, 여러 태그 비교는 소규모 다중 차트(small multiples)."""
import matplotlib.pyplot as plt
import seaborn as sns

from .config import OUTPUTS

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # 검증된 범주형 팔레트 앞 4칸
GRAY = "#8a8985"
SEQ = "Blues"
DIV = "RdBu_r"


def setup():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW]),
    })
    sns.set_palette([BLUE, ORANGE, AQUA, YELLOW])


def save(fig, name):
    fig.savefig(OUTPUTS / f"{name}.png", bbox_inches="tight", dpi=130)
