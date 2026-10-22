"""notebooks/py/*.py (# %% 셀 구분) → notebooks/*.ipynb 변환 후 실행.

    python scripts/build_notebooks.py            # 전부 변환·실행
    python scripts/build_notebooks.py 02 03      # 일부만
    python scripts/build_notebooks.py --md-only 01  # 마크다운만 갱신 (코드가 같으면 기존 출력 재사용, 실행 안 함)
"""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks" / "py", ROOT / "notebooks"


def py_to_nb(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip().startswith("[markdown]"):
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(md))
        elif body:
            nb.cells.append(nbformat.v4.new_code_cell(body))
    return nb


def reuse_outputs(nb, old) -> bool:
    new_code = [c for c in nb.cells if c.cell_type == "code"]
    old_code = [c for c in old.cells if c.cell_type == "code"]
    if [c.source for c in new_code] != [c.source for c in old_code]:
        return False
    for n, o in zip(new_code, old_code):
        n.outputs, n.execution_count = o.outputs, o.execution_count
    return True


def main(keys):
    md_only = "--md-only" in keys
    keys = [k for k in keys if k != "--md-only"]
    for src in sorted(SRC.glob("*.py")):
        if keys and not any(src.name.startswith(k) for k in keys):
            continue
        nb = py_to_nb(src.read_text(encoding="utf-8"))
        out = OUT / (src.stem + ".ipynb")
        if md_only:
            if out.exists() and reuse_outputs(nb, nbformat.read(out, as_version=4)):
                nbformat.write(nb, out)
                print("마크다운 갱신:", out.name)
            else:
                print("코드가 바뀌어 다시 실행해야 함:", out.name)
            continue
        print("실행:", out.name, flush=True)
        try:
            NotebookClient(nb, timeout=None, kernel_name="python3",
                           resources={"metadata": {"path": str(OUT)}}).execute()
        finally:  # 중간에 에러가 나도 그때까지의 출력은 남긴다
            nbformat.write(nb, out)


if __name__ == "__main__":
    main(sys.argv[1:])
