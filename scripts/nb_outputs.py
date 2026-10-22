"""실행된 ipynb의 텍스트 출력만 훑어보기 (README 작성용)."""
import sys
import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        txt = o.get("text") or o.get("data", {}).get("text/plain", "")
        if o.get("output_type") == "error":
            txt = "ERROR: " + o.get("ename", "") + " " + o.get("evalue", "")
        if txt:
            print(f"--- [{i}]\n{txt[:3000]}")
