"""노트북 형식과 코드 셀, 내장 보조 코드의 Python 문법을 검사한다."""
import ast
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
path=root/"notebooks/SSPNet_무료_Colab_추론과_수영자평가.ipynb"
nb=json.loads(path.read_text(encoding="utf-8"))
assert nb["nbformat"]==4 and nb["nbformat_minor"]==5
ids=set()
code_count=0
embedded=None
for index,cell in enumerate(nb["cells"]):
    assert cell["cell_type"] in ("markdown","code")
    assert isinstance(cell["source"],list) and all(isinstance(s,str) for s in cell["source"])
    assert cell["id"] not in ids
    ids.add(cell["id"])
    source="".join(cell["source"])
    if cell["cell_type"]=="code":
        assert cell["outputs"]==[] and cell["execution_count"] is None
        compile(source,"셀 %d" % index,"exec")
        tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=="HELPER_SOURCES" for n in node.targets):
                embedded=ast.literal_eval(node.value)
        code_count+=1
assert embedded is not None
for name,source in embedded.items():
    assert source==(root/"sspnet_colab"/name).read_text(encoding="utf-8")
    ast.parse(source,feature_version=(3,8))
    compile(source,name,"exec")
assert nb["metadata"]["accelerator"]=="GPU"
update=(root/"notebooks/기존_Colab_전체635장_평가_업데이트셀.txt").read_text(encoding="utf-8")
compile(update,"기존 런타임 업데이트 셀","exec")
tree=ast.parse(update)
values={}
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=="HELPER_SOURCES" for n in node.targets):
        values=ast.literal_eval(node.value)
assert values==embedded
print("노트북 구조·코드 셀 %d개·내장 Python 3.8 코드 %d개 검사 통과" % (code_count,len(embedded)))
print("기존 런타임 업데이트 셀 문법·내장 코드 일치 확인 통과")
