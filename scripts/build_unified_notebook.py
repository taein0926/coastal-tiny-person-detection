"""중복 설정과 평가 셀을 합친 5단계 독립 실행 노트북을 만든다."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cells=[]


def add(kind,source,form=False):
    source=source.strip()+"\n"
    cell=dict(cell_type=kind,metadata={"cellView":"form"} if form else {},
              id=hashlib.sha256(source.encode()).hexdigest()[:12],source=source.splitlines(keepends=True))
    if kind=="code":cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add("markdown",'''# SSPNet 통합 평가 · 무료 Colab

**처음에는 1→2→3→4→5 순서로 실행하세요.** 무료 T4 GPU를 선택합니다.

1. **설정:** 전체 검사(`full`)·분할 검사(`tiles`)와 데이터 범위를 한 곳에서 선택합니다.
2. **환경·가중치 준비:** 설치와 가중치 준비를 한 번에 처리합니다. 같은 실행 환경에서는 기존 파일을 재사용합니다.
3. **입력 준비:** 데이터 COCO ZIP을 업로드합니다. 전체 검사와 비교하려면 기존 전체 검사 결과 ZIP도 준비합니다.
4. **추론·평가:** 모델은 사진 전체에서 사람을 찾습니다. 동일한 예측을 전체 사람·물속 사람 기준으로 각각 채점합니다.
5. **결과·다운로드:** 두 기준의 결과와 전체/분할 비교표를 확인하고 ZIP을 받습니다.

기본값은 **AnnotationPiriapolis v4 전체 635장 / 분할 320픽셀 / 기본 겹침 20% / 점수 0.25 / IoU 0.3**입니다. IoU 0.5 참고 결과도 저장합니다. 가장자리에서는 설정한 비율보다 더 겹칠 수 있습니다.

**설정 변경 시:** 1→2→4→5를 실행합니다. 데이터·분할도 변경했다면 3단계도 다시 실행합니다. **Colab 임시 컴퓨터가 새로 배정됨:** 1→2→3→4→5를 실행합니다. 같은 설정의 중단된 추론은 저장된 완료 기록을 재사용합니다.

학습 전후 비교는 `EVALUATION_MODE=full`, `SPLIT=all`, `MAX_IMAGES=0`으로 설정합니다. 공개 가중치는 `public`, 학습 가중치는 Drive 경로를 입력하는 `trained_path` 또는 직접 올리는 `trained_upload`를 고르고 2단계를 실행합니다. Drive 경로를 사용하려면 `USE_GOOGLE_DRIVE`를 켭니다. 학습 가중치 결과는 가중치 해시별 별도 폴더에 저장됩니다.

크기 제한 없는 전체 사람 결과가 주 결과입니다. 물속 사람 결과와 `s=√(가로×세로)` 기준의 2~8·8~20픽셀 재현율은 보조 결과입니다. 물속 사람 기준에서는 모래사장 사람 탐지도 오탐에 포함됩니다. 기존 주석의 누락·위치 오류를 자동 보정하지 않습니다.''')

add("markdown","## 1. 설정\n\n전체 635장 분할 평가에는 기본값을 유지하세요. 전체 검사 결과 ZIP과 비교하려면 `COMPARE_WITH_FULL`을 켭니다. `MAX_IMAGES=0`은 선택한 분할의 모든 이미지입니다.")
add("code",'''#@title 1. 설정
import os,sys,json,shutil,subprocess,zipfile
from pathlib import Path
from IPython.display import display,Image as DisplayImage

TASK = "dataset" #@param ["dataset", "single_image"]
EVALUATION_MODE = "tiles" #@param ["full", "tiles"]
WEIGHT_SOURCE = "public" #@param ["public", "trained_path", "trained_upload"]
TRAINED_WEIGHT_PATH = "" #@param {type:"string"}
DATASET_NAME = "AnnotationPiriapolis_v4" #@param ["AnnotationPiriapolis_v4", "YoloBeachTrain_v2", "YoloBeachTrain_v3"]
SPLIT = "all" #@param ["all", "test", "valid", "train"]
MAX_IMAGES = 0 #@param {type:"integer"}
TILE_SIZE = 320 #@param {type:"integer"}
TILE_OVERLAP = 0.20 #@param {type:"number"}
REPORT_SCORE = 0.25 #@param {type:"number"}
MATCH_IOU = 0.3 #@param {type:"number"}
COMPARE_WITH_FULL = True #@param {type:"boolean"}
USE_GOOGLE_DRIVE = False #@param {type:"boolean"}

# 이미 Colab/Drive에 있는 파일은 아래 경로를 입력할 수 있습니다. 비어 있으면 기존 파일을 찾거나 업로드합니다.
DATASET_ZIP_PATH = ""
BASELINE_ZIP_PATH = ""
IMAGE_SOURCE = "upload" # upload / demo / path
IMAGE_PATH_SETTING = ""
WORK=Path("/content/sspnet_free")
WORK.mkdir(exist_ok=True)
CONFIG=dict(task=TASK,mode=EVALUATION_MODE,dataset_name=DATASET_NAME,split=SPLIT,max_images=MAX_IMAGES,
            tile_size=TILE_SIZE,tile_overlap=TILE_OVERLAP,report_score=REPORT_SCORE,match_iou=MATCH_IOU,
            long_edge=1280,short_edge=1024,compare_with_full=COMPARE_WITH_FULL)
CONFIG["weight_source"]=WEIGHT_SOURCE
if MAX_IMAGES<0 or TILE_SIZE<=0 or not 0<=TILE_OVERLAP<1 or not .001<=REPORT_SCORE<=1 or not 0<MATCH_IOU<=1:
    raise ValueError("이미지 수·분할 크기·점수·IoU 설정을 확인하세요.")
CONFIG_PATH=WORK/"unified_settings.json"
CONFIG_PATH.write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding="utf-8")
print("설정 저장:",TASK,EVALUATION_MODE,DATASET_NAME,SPLIT,"/ 전체 이미지" if MAX_IMAGES==0 else f"/ 최대 {MAX_IMAGES}장")
''',True)

add("markdown","## 2. 환경·가중치 준비\n\n처음 또는 새 실행 환경에서 실행하세요. 설치가 끝날 때까지 기다립니다. GPU 연산 검사를 통과해야 합니다. Drive 저장을 켰다면 인증 화면을 직접 완료하세요.")
helpers={name:(ROOT/"sspnet_colab"/name).read_text(encoding="utf-8") for name in
         ("setup_environment.py","metrics.py","engine.py","summarize_scopes.py","workflow.py")}
add("code",'#@title 2. 환경·가중치 준비\nHELPER_SOURCES = '+repr(helpers)+'''
if not shutil.which("nvidia-smi"):
    raise RuntimeError("런타임 유형을 무료 GPU로 변경하세요.")
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    STORE=Path("/content/drive/MyDrive/SSPNet_free_colab")
else:
    STORE=WORK/"saved"
STORE.mkdir(parents=True,exist_ok=True)
HELPERS=WORK/"helpers"
HELPERS.mkdir(exist_ok=True)
for name,source in HELPER_SOURCES.items():
    (HELPERS/name).write_text(source,encoding="utf-8")
subprocess.run([sys.executable,str(HELPERS/"setup_environment.py")],check=True)
PYTHON=WORK/"venv/bin/python"
REPO=WORK/"SSPNet"
ENV=os.environ.copy()
ENV.pop("PYTHONHOME",None)
ENV.update(PYTHONNOUSERSITE="1",PYTHONPATH=str(REPO)+os.pathsep+str(HELPERS),MPLBACKEND="Agg")
def checked(command):
    subprocess.run([str(x) for x in command],check=True,env=ENV)
CHECKPOINT=WORK/"faster_rcnn_sspnet.pth"
WEIGHT_STORE=STORE/"weights/faster_rcnn_sspnet.pth"
WEIGHT_STORE.parent.mkdir(parents=True,exist_ok=True)
def valid_weight(path):
    return path.is_file() and path.stat().st_size>10_000_000
if WEIGHT_SOURCE == "public" and not valid_weight(CHECKPOINT):
    if valid_weight(WEIGHT_STORE):shutil.copy2(WEIGHT_STORE,CHECKPOINT)
    else:
        try:
            checked([PYTHON,"-c","import gdown; p=gdown.download(id='1IfPCt5xZqqBJ3sYVIuD5F9l29Jcy2Hn1',output=%r,quiet=False); assert p"%str(CHECKPOINT)])
        except subprocess.CalledProcessError:
            from google.colab import files
            print("가중치 자동 다운로드가 제한됐습니다. 저자의 Faster R-CNN SSPNet .pth 파일을 선택하세요.")
            uploaded=files.upload()
            if len(uploaded)!=1:raise ValueError("가중치 한 개를 선택하세요.")
            CHECKPOINT.write_bytes(next(iter(uploaded.values())))
if WEIGHT_SOURCE == "public":
    if not valid_weight(CHECKPOINT):raise ValueError("가중치 파일이 유효하지 않습니다.")
    import hashlib
    def weight_hash(path):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
        return h.hexdigest()
    if weight_hash(CHECKPOINT)!="912cc5b6f8a45b0f4521c9bf325868ed10dd4757497fb6ba8843fdbd26b09e4a":
        raise ValueError("공개 가중치 해시가 기준과 다릅니다. 기존 파일을 확인하세요.")
    if CHECKPOINT.resolve()!=WEIGHT_STORE.resolve():shutil.copy2(CHECKPOINT,WEIGHT_STORE)
elif WEIGHT_SOURCE == "trained_path":
    CHECKPOINT=Path(TRAINED_WEIGHT_PATH)
elif WEIGHT_SOURCE == "trained_upload":
    from google.colab import files
    print("추가 학습한 최종 .pth 가중치를 선택하세요.")
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError("가중치 한 개를 선택하세요.")
    CHECKPOINT=WORK/"trained_uploaded.pth"
    CHECKPOINT.write_bytes(next(iter(uploaded.values())))
else: raise ValueError("가중치 선택을 확인하세요.")
if not valid_weight(CHECKPOINT):raise ValueError("학습 가중치 경로 또는 파일을 확인하세요.")
CONFIG["checkpoint"]=str(CHECKPOINT)
CONFIG_PATH.write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding="utf-8")
shutil.copy2(WORK/"installed_packages.txt",STORE/"installed_packages.txt")
print("환경·가중치 준비 완료. 다음은 3단계입니다.")
''',True)

add("markdown","## 3. 입력 준비\n\n`dataset`이면 데이터 COCO ZIP을 선택합니다. 비교를 켰다면 기존 `SSPNet_all635_full_results.zip`도 선택합니다. 이미 준비된 파일은 재사용합니다. `single_image`이면 이미지 한 장을 선택합니다.")
add("code",'''#@title 3. 입력 준비
if not (WORK/"venv/bin/python").is_file() or "ENV" not in globals():
    raise RuntimeError("2단계를 먼저 실행하세요.")
def upload_one(prompt):
    from google.colab import files
    print(prompt)
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError("파일 한 개를 선택하세요.")
    return next(iter(uploaded.items()))
if CONFIG["task"]=="dataset":
    DATA_ZIP=Path(DATASET_ZIP_PATH) if DATASET_ZIP_PATH else WORK/(CONFIG["dataset_name"]+".zip")
    if not DATA_ZIP.is_file():
        _,content=upload_one("데이터셋 COCO JSON ZIP을 선택하세요.")
        DATA_ZIP=WORK/(CONFIG["dataset_name"]+".zip")
        DATA_ZIP.write_bytes(content)
    checked([PYTHON,HELPERS/"workflow.py","prepare","--config",CONFIG_PATH,"--zip",DATA_ZIP,"--store",STORE])
    baseline=STORE/"baselines"/CONFIG["dataset_name"]
    local_full=STORE/"evaluation"/CONFIG["dataset_name"]/CONFIG["split"]/"full"
    if CONFIG["compare_with_full"] and CONFIG["mode"]!="full":
        if BASELINE_ZIP_PATH or not any((p/"predictions_coco.json").is_file() for p in (baseline,local_full)):
            source=Path(BASELINE_ZIP_PATH) if BASELINE_ZIP_PATH else WORK/"baseline_full_results.zip"
            if not source.is_file():
                _,content=upload_one("이전에 다운로드한 전체 검사 결과 ZIP을 선택하세요.")
                source.write_bytes(content)
            checked([PYTHON,HELPERS/"workflow.py","baseline","--config",CONFIG_PATH,"--zip",source,"--store",STORE])
else:
    if IMAGE_SOURCE=="demo":IMAGE_PATH=REPO/"demo/demo.jpg"
    elif IMAGE_SOURCE=="path":IMAGE_PATH=Path(IMAGE_PATH_SETTING)
    else:
        name,content=upload_one("추론할 이미지 한 장을 선택하세요.")
        IMAGE_PATH=WORK/("input"+Path(name).suffix.lower())
        IMAGE_PATH.write_bytes(content)
    if not IMAGE_PATH.is_file():raise FileNotFoundError("이미지 파일이 없습니다.")
    display(DisplayImage(filename=str(IMAGE_PATH),width=800))
print("입력 준비 완료. 다음은 4단계입니다.")
''',True)

add("markdown","## 4. 추론·평가\n\n이 셀만 실제 모델 추론을 실행합니다. 사진마다 완료 기록을 저장합니다. 전체 사람·물속 사람 결과와 IoU 0.5 참고 결과를 같은 예측으로 집계합니다. 비교를 켰다면 기존 전체 검사 예측도 동일한 기준으로 다시 채점합니다.")
add("code",'''#@title 4. 추론·평가
if "ENV" not in globals() or not CHECKPOINT.is_file():raise RuntimeError("2~3단계를 먼저 실행하세요.")
if CONFIG["task"]=="dataset":
    checked([PYTHON,HELPERS/"workflow.py","run","--config",CONFIG_PATH,"--store",STORE])
else:
    if "IMAGE_PATH" not in globals() or not IMAGE_PATH.is_file():raise RuntimeError("3단계에서 이미지를 준비하세요.")
    settings=dict(repo=str(REPO),checkpoint=str(CHECKPOINT),mode=CONFIG["mode"],long_edge=CONFIG["long_edge"],short_edge=CONFIG["short_edge"],
                  tile_size=CONFIG["tile_size"],tile_overlap=CONFIG["tile_overlap"],merge_iou=.5,report_score=CONFIG["report_score"],
                  save_min_score=.001,max_detections=500,target_names=["person water"],match_iou=CONFIG["match_iou"])
    path=WORK/"single_settings.json"
    path.write_text(json.dumps(settings),encoding="utf-8")
    out=STORE/("single_image_"+CONFIG["mode"])
    checked([PYTHON,HELPERS/"engine.py","--settings",path,"--image",IMAGE_PATH,"--out",out])
    (WORK/"last_output.json").write_text(json.dumps(dict(task="single_image",paths=[str(out)],label=CONFIG["mode"],config=CONFIG)),encoding="utf-8")
print("실행 완료. 5단계에서 결과를 확인하고 다운로드하세요.")
''',True)

add("markdown","## 5. 결과 확인·다운로드\n\n전체 사람 결과가 주 결과입니다. 비교표의 재현율·정밀도는 0~1 비율입니다. 같은 모델·이미지·점수·IoU로 전체/분할을 비교합니다. 주석 누락이 있으면 실제 사람 탐지도 주석 기준 오탐에 포함될 수 있습니다. 현재 실행 결과와 비교 자료만 ZIP에 담습니다.")
add("code",'''#@title 5. 결과 확인·다운로드
if not (WORK/"last_output.json").is_file():raise RuntimeError("4단계를 먼저 실행하세요.")
last=json.loads((WORK/"last_output.json").read_text(encoding="utf-8"))
if last.get("config")!=CONFIG:raise RuntimeError("설정이 바뀌었습니다. 4단계를 실행한 뒤 결과를 확인하세요.")
paths=[Path(p) for p in last["paths"]]
if last["task"]=="dataset":
    out=paths[0]
    print((out/"전체사람_물속사람_결과요약.txt").read_text(encoding="utf-8"))
    import pandas as pd
    comparison=out/"전체검사_분할검사_비교.json"
    if CONFIG["compare_with_full"] and CONFIG["mode"]!="full" and comparison.is_file():
        print("전체 검사·분할 검사 비교표")
        display(pd.DataFrame(json.loads(comparison.read_text(encoding="utf-8"))))
else:
    display(DisplayImage(filename=str(paths[0]/"prediction.jpg"),width=1100))
result_zip=WORK/("SSPNet_"+last["task"]+"_"+last["label"]+"_results.zip")
with zipfile.ZipFile(result_zip,"w",compression=zipfile.ZIP_DEFLATED) as archive:
    for folder in paths:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() not in (".pth",".pt",".tmp"):
                archive.write(path,path.relative_to(STORE))
    archive.write(CONFIG_PATH,"unified_settings.json")
    if (STORE/"installed_packages.txt").is_file():archive.write(STORE/"installed_packages.txt","installed_packages.txt")
print("결과 ZIP",result_zip)
from google.colab import files
files.download(str(result_zip))
''',True)

nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(colab=dict(name="SSPNet_통합평가_무료Colab.ipynb",provenance=[]),accelerator="GPU",
       kernelspec=dict(display_name="Python 3",language="python",name="python3"),language_info=dict(name="python",version="3")))
path=ROOT/"notebooks/SSPNet_통합평가_무료Colab.ipynb"
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print(path)
