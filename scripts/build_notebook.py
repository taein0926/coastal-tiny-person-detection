"""보조 코드를 포함한 독립 실행 Colab 노트북을 생성한다."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def cell(kind, source):
    source = source.strip() + "\n"
    item = dict(cell_type=kind, metadata={}, id=hashlib.sha256(source.encode()).hexdigest()[:12],
                source=source.splitlines(keepends=True))
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    CELLS.append(item)


def md(source):
    cell("markdown", source)


def code(source):
    cell("code", source)


md("""
# 무료 Colab에서 SSPNet 실행과 작은 수영자 평가

태인님을 위한 실행 노트북입니다. **무료 GPU 런타임만 사용**하며 결제·유료 서비스·상시 연결 기능을 사용하지 않습니다.

1. Colab에서 **파일 → 노트북 업로드**로 이 파일을 여세요.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택하세요. 무료 GPU가 배정되지 않으면 여기서 멈추고 나중에 다시 시도하세요.
3. 1~7단계를 위에서부터 실행하면 환경 확인 → 공개 가중치 로드 → 예제 이미지 한 장 추론 → 결과 다운로드까지 진행합니다.
4. 실제 수영자 이미지는 6단계에서 업로드할 수 있습니다. 8단계 이후는 데이터셋 평가이며, 기본 범위는 **train·valid·test 전체**입니다.

이번 개정본은 AnnotationPiriapolis v4의 전체 635장과 `s=√(박스 가로×박스 세로)` 기준의 `2≤s<20`, `2≤s<8` 구간을 집계합니다. 다른 모델과의 비교표를 만들지 않고 이번 실행 결과의 문서용 요약을 저장합니다. 정답 개수 대조로 확인한 범위와 크기 구분을 사용하며, 외부 테스트의 미공개 임계값·IoU·입력 해상도까지 재현했다는 의미는 아닙니다.

**검증 범위:** 배포 전 Python 문법과 정답 변환·매칭·크기별 평가·분할 병합을 로컬에서 검사했습니다. 이 파일에 Colab GPU 실행 결과를 미리 넣지 않았습니다. 실제 CUDA·가중치·추론 성공은 아래 셀의 출력으로 확인하세요. 무료 GPU와 외부 다운로드의 가용성은 보장되지 않습니다.

**저장 위치:** 설치·원본 데이터는 Colab의 `/content`에 저장합니다. 이 노트북만 업로드하면 되며, 별도 `.py` 파일은 필요 없습니다. 런타임이 삭제되면 `/content`도 없어집니다. 세션을 넘어 이어서 평가하려면 2단계에서 Google Drive 저장을 켜고, 새 세션에서 데이터 ZIP을 다시 준비하세요.

**실험 범위:** 추가 학습 없이 TinyPerson 사전학습 SSPNet을 다른 해변 이미지에 적용합니다. 원 논문의 멀티스케일 평가 재현과 구분하세요. 이 노트북은 기본적으로 한 장씩·단일 해상도로 처리합니다.
""")

md("""
## 1. GPU와 저장공간 확인

이 셀과 설치 셀은 무료 Colab의 Linux GPU 런타임에서 실행합니다. TPU와 CPU 런타임은 사용하지 않습니다. 여기서 표시하는 디스크·GPU는 태인님 노트북 사양이 아니라 Colab에 배정된 자원입니다.
""")
code('''
import os, sys, json, shutil, subprocess, platform, zipfile
from pathlib import Path
from IPython.display import display, Image as DisplayImage

WORK = Path("/content/sspnet_free")
if platform.system() != "Linux" or not Path("/content").exists():
    raise RuntimeError("이 노트북을 Google Colab에서 열어 실행하세요.")
if not shutil.which("nvidia-smi"):
    raise RuntimeError("무료 GPU 런타임이 필요합니다. 런타임 유형을 GPU로 변경하세요.")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True)
disk = shutil.disk_usage("/content")
print("Colab 임시 디스크 여유: %.1f GiB" % (disk.free / 1024**3))
print("Colab 기본 Python:", sys.version.split()[0])
WORK.mkdir(exist_ok=True)
''')

md("""
## 2. 결과 보관 위치 선택

처음 한 장만 확인할 때는 기본값 `False`로 실행해도 됩니다. 두 데이터셋 평가를 이어갈 때는 **`USE_GOOGLE_DRIVE = True`를 권합니다.** 인증 화면은 태인님이 직접 완료하세요.

Drive를 켜면 가중치와 이미지별 완료 기록·평가 결과를 `MyDrive/SSPNet_free_colab`에 보관합니다. 실제 이미지 처리와 설치는 `/content`에서 수행합니다. Drive의 남은 용량은 별도로 확인하세요. 무료 Colab 사용과 Google Drive의 저장공간은 별개입니다.
""")
code('''
USE_GOOGLE_DRIVE = False #@param {type:"boolean"}
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    STORE = Path("/content/drive/MyDrive/SSPNet_free_colab")
else:
    STORE = WORK / "saved"
STORE.mkdir(parents=True, exist_ok=True)
print("가중치·결과 저장:", STORE)
if not USE_GOOGLE_DRIVE:
    print("현재는 임시 저장입니다. 세션 종료 전에 마지막 결과 ZIP을 다운로드하세요.")
''')

md("""
## 3. 실행 코드 준비

이 셀은 노트북에 포함된 보조 코드를 Colab 임시 폴더에 씁니다. 외부의 별도 보조 파일은 필요 없습니다.
""")
embedded = {name: (ROOT/"sspnet_colab"/name).read_text(encoding="utf-8")
            for name in ("setup_environment.py", "metrics.py", "engine.py")}
code("#@title 내장 실행 코드 준비\nHELPER_SOURCES = " + repr(embedded) + '''
HELPERS = WORK / "helpers"
HELPERS.mkdir(exist_ok=True)
for name, source in HELPER_SOURCES.items():
    (HELPERS / name).write_text(source, encoding="utf-8")
print("보조 코드 준비 완료:", list(HELPER_SOURCES))
''')
CELLS[-1]["metadata"]["cellView"] = "form"

md("""
## 4. Python 3.8 환경 설치와 GPU 연산 검사

Colab 기본 Python·PyTorch는 바꾸지 않습니다. `uv`로 별도 Python 3.8.20 환경을 만들고 모든 SSPNet 명령을 그 환경의 Python으로 실행합니다.

| 구성 | 고정 버전 |
|---|---|
| Python | 3.8.20 |
| PyTorch / torchvision | 1.10.0+cu111 / 0.11.1+cu111 |
| MMCV | mmcv-full 1.3.18, 사전 빌드 wheel |
| NumPy | 1.21.6 |
| SSPNet 코드 | `f2d0f42273945577a5c6497fd2b384cb545f8f66` |

README의 Python 3.7·CUDA 10.2 대신, 배포된 Linux wheel이 있는 Python 3.8·CUDA 11.1 조합을 사용합니다. CUDA 툴킷 전체를 설치하거나 MMCV를 소스에서 컴파일하지 않습니다. 첫 설치는 네트워크에 따라 수 분 이상 걸리고 PyTorch 다운로드가 큽니다. 완료 전에 다음 셀을 실행하지 마세요.

설치 마지막에 **CUDA 행렬 연산·NMS·RoIAlign**이 성공해야 합니다. 실패하면 출력의 첫 오류를 확인하세요. 새 GPU에서 구형 CUDA 연산이 지원되지 않는 경우까지 자동으로 해결한다고 가정하지 않습니다.
""")
code('''
def checked(command, env=None):
    subprocess.run([str(x) for x in command], check=True, env=env)

checked([sys.executable, HELPERS / "setup_environment.py"])
PYTHON = WORK / "venv/bin/python"
REPO = WORK / "SSPNet"
ENV = os.environ.copy()
ENV.pop("PYTHONHOME", None)
ENV["PYTHONNOUSERSITE"] = "1"
ENV["PYTHONPATH"] = str(REPO) + os.pathsep + str(HELPERS)
ENV["MPLBACKEND"] = "Agg"
shutil.copy2(WORK / "installed_packages.txt", STORE / "installed_packages.txt")
print("SSPNet 전용 Python:", PYTHON)
''')

md("""
## 5. Faster R-CNN SSPNet 가중치 준비

모델 저자가 README에 연결한 **Faster R-CNN SSPNet** 가중치를 받습니다. Fovea·일반 Faster R-CNN·YOLO 가중치를 섞어 쓰지 마세요. 7단계에서 모든 모델 파라미터를 엄격하게 대조하며, 누락·불일치를 무시하지 않습니다.

공유 다운로드가 제한되면 [저자의 다운로드 링크](https://drive.google.com/file/d/1IfPCt5xZqqBJ3sYVIuD5F9l29Jcy2Hn1/view)에서 직접 받은 `.pth`를 사용하세요. `WEIGHT_SOURCE`를 `upload`로 바꾸면 업로드할 수 있고, `path`는 기존 Colab/Drive 파일을 사용합니다. 캐시된 가중치가 있으면 다시 다운로드하지 않습니다.
""")
code('''
WEIGHT_SOURCE = "download" #@param ["download", "upload", "path"]
EXISTING_WEIGHT_PATH = "" #@param {type:"string"}
WEIGHT_STORE = STORE / "weights/faster_rcnn_sspnet.pth"
WEIGHT_STORE.parent.mkdir(exist_ok=True)
CHECKPOINT = WORK / "faster_rcnn_sspnet.pth"
if WEIGHT_SOURCE == "upload":
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("가중치 파일 한 개만 선택하세요.")
    CHECKPOINT.write_bytes(next(iter(uploaded.values())))
elif WEIGHT_SOURCE == "path":
    source = Path(EXISTING_WEIGHT_PATH)
    if not source.is_file():
        raise FileNotFoundError("기존 가중치 경로를 확인하세요.")
    if source.resolve() != CHECKPOINT.resolve():
        shutil.copy2(source, CHECKPOINT)
elif WEIGHT_SOURCE == "download":
    if WEIGHT_STORE.is_file() and WEIGHT_STORE.stat().st_size > 10_000_000:
        if WEIGHT_STORE.resolve() != CHECKPOINT.resolve():
            shutil.copy2(WEIGHT_STORE, CHECKPOINT)
    else:
        download_code = (
            "import gdown; p=gdown.download(id='1IfPCt5xZqqBJ3sYVIuD5F9l29Jcy2Hn1', "
            "output=" + repr(str(CHECKPOINT)) + ", quiet=False); "
            "assert p, '가중치 다운로드에 실패했습니다.'"
        )
        checked([PYTHON, "-c", download_code], ENV)
else:
    raise ValueError("가중치 입력 방식을 확인하세요.")
if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 10_000_000:
    raise ValueError("가중치 파일이 너무 작습니다. 다운로드 오류 또는 HTML 파일인지 확인하세요.")
print("가중치 준비:", CHECKPOINT, "%.1f MiB" % (CHECKPOINT.stat().st_size / 1024**2))
''')

md("""
## 6. 추론할 이미지 한 장 선택

기본 `demo`는 SSPNet 저장소의 예제 이미지로 **실행 경로 확인용**이며 수영자 성능 평가용이 아닙니다. 태인님의 수영자 이미지가 있으면 `upload`, 이미 Colab에 있다면 `path`를 선택하세요. 업로드 이미지가 작게 보이더라도 정답 박스의 크기는 확대 전 픽셀 기준으로 판단합니다.
""")
code('''
IMAGE_SOURCE = "demo" #@param ["demo", "upload", "path"]
EXISTING_IMAGE_PATH = "" #@param {type:"string"}
if IMAGE_SOURCE == "demo":
    IMAGE_PATH = REPO / "demo/demo.jpg"
elif IMAGE_SOURCE == "upload":
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("이미지 한 장만 선택하세요.")
    name, content = next(iter(uploaded.items()))
    IMAGE_PATH = WORK / ("input" + Path(name).suffix.lower())
    IMAGE_PATH.write_bytes(content)
else:
    IMAGE_PATH = Path(EXISTING_IMAGE_PATH)
if not IMAGE_PATH.is_file():
    raise FileNotFoundError("이미지 경로를 확인하세요.")
display(DisplayImage(filename=str(IMAGE_PATH), width=900))
''')

md("""
## 7. 이미지 한 장 추론과 결과 확인

`REPORT_SCORE`는 화면 표시와 재현율 보고용 임계값입니다. 낮은 점수의 박스도 `SAVE_MIN_SCORE` 이상이면 저장하여 나중에 정밀도·재현율을 비교할 수 있습니다. 출력 라벨 `person`은 모델의 예측 클래스이며, 수영 또는 익수 상태를 판단했다는 의미가 아닙니다.

`full`은 이미지 전체, `tiles`는 겹치는 패치로 분할하는 방식입니다. 먼저 기본 `full`로 동작을 확인하세요. 메모리 부족이면 해상도를 낮추거나 한 장씩 분할하세요. 해상도를 바꿨다면 별도 실험으로 기록해야 합니다.
""")
code('''
MODE = "full" #@param ["full", "tiles"]
LONG_EDGE = 1280 #@param {type:"integer"}
SHORT_EDGE = 1024 #@param {type:"integer"}
TILE_SIZE = 640 #@param {type:"integer"}
TILE_OVERLAP = 0.25 #@param {type:"number"}
REPORT_SCORE = 0.25 #@param {type:"number"}
SAVE_MIN_SCORE = 0.001 #@param {type:"number"}
MAX_DETECTIONS = 500 #@param {type:"integer"}
if not 0 < SHORT_EDGE <= LONG_EDGE or TILE_SIZE <= 0 or MAX_DETECTIONS <= 0:
    raise ValueError("해상도와 최대 탐지 개수는 양수로 설정하세요.")
if not 0 <= TILE_OVERLAP < 1:
    raise ValueError("겹침은 0 이상 1 미만이어야 합니다.")
SETTINGS = dict(repo=str(REPO), checkpoint=str(CHECKPOINT), mode=MODE,
    long_edge=LONG_EDGE, short_edge=SHORT_EDGE, tile_size=TILE_SIZE,
    tile_overlap=TILE_OVERLAP, merge_iou=0.5, report_score=REPORT_SCORE,
    save_min_score=SAVE_MIN_SCORE, max_detections=MAX_DETECTIONS,
    target_names=["person water"])
SETTINGS_PATH = WORK / "settings.json"
SETTINGS_PATH.write_text(json.dumps(SETTINGS), encoding="utf-8")
SINGLE_OUT = STORE / ("single_image_" + MODE)
checked([PYTHON, HELPERS / "engine.py", "--settings", SETTINGS_PATH,
         "--image", IMAGE_PATH, "--out", SINGLE_OUT], ENV)
# 엄격한 가중치 로드와 실제 추론을 통과한 파일만 보관한다.
if CHECKPOINT.resolve() != WEIGHT_STORE.resolve():
    shutil.copy2(CHECKPOINT, WEIGHT_STORE)
display(DisplayImage(filename=str(SINGLE_OUT / "prediction.jpg"), width=1200))
record = json.loads((SINGLE_OUT / "predictions.json").read_text())
visible = sum(b[4] >= REPORT_SCORE for b in record["boxes"])
print("실제 추론 완료. 표시 임계값 이상 박스:", visible)
print("탐지 0개도 실행 성공일 수 있습니다. 성능은 정답과 비교해야 판단할 수 있습니다.")
print("추론·분할 병합 시간: %.3f초 / 최대 할당 VRAM: %.1f MiB" %
      (record["seconds"], record["peak_vram_mib"]))
''')

md("""
## 8. 선택 사항: 두 데이터셋의 COCO JSON ZIP 준비

**여기부터는 데이터셋을 확보한 뒤 실행합니다.** 지금은 `PREPARE_DATASET = False`여도 1~7단계 결과를 마지막 셀에서 다운로드할 수 있습니다.

- [Roboflow 프로젝트 모음](https://universe.roboflow.com/count-people-at-beaches)에서 **AnnotationPiriapolis v4** 또는 **YoloBeachTrain v2**를 선택하고 **COCO JSON 형식**으로 내보내세요. Roboflow 모델 API로 원격 추론하는 방식은 사용하지 않습니다. 데이터 다운로드에 유료 결제를 요구하면 진행하지 마세요.
- 두 데이터셋은 한 번에 하나씩 평가합니다. `DATASET_NAME`을 바꾸고 8~11단계를 반복하세요.
- ZIP에 `train/`, `valid/`, `test/`와 각 분할의 `_annotations.coco.json`·이미지가 함께 있어야 합니다. **`SPLIT=all`은 세 분할을 모두 합칩니다.** 분할 간 중복되는 ID와 파일명은 구분하고 이미지를 복사하지 않습니다. 누락된 분할이 있으면 중단합니다.
- `upload`는 로컬 ZIP, `path`는 Drive/Colab의 기존 ZIP입니다. 큰 ZIP은 Drive에 보관하고 `path`로 사용하는 편이 편리합니다. API 키는 필요하지 않습니다.
- 전체 평가에는 배포된 증강 이미지도 포함합니다. 향후 이 데이터로 추가 학습한 모델의 독립 테스트 성능을 보고할 때는 학습 분할을 분리하세요.
""")
code('''
PREPARE_DATASET = False #@param {type:"boolean"}
DATASET_NAME = "AnnotationPiriapolis_v4" #@param ["AnnotationPiriapolis_v4", "YoloBeachTrain_v2"]
ZIP_SOURCE = "upload" #@param ["upload", "path"]
EXISTING_ZIP_PATH = "" #@param {type:"string"}
SPLIT = "all" #@param ["all", "test", "valid", "train"]
ANNOTATION_OVERRIDE = "" #@param {type:"string"}
ANNOTATIONS = None
if PREPARE_DATASET:
    if ZIP_SOURCE == "upload":
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("데이터셋 ZIP 한 개만 선택하세요.")
        DATA_ZIP = WORK / (DATASET_NAME + ".zip")
        DATA_ZIP.write_bytes(next(iter(uploaded.values())))
    else:
        DATA_ZIP = Path(EXISTING_ZIP_PATH)
    if not DATA_ZIP.is_file():
        raise FileNotFoundError("데이터셋 ZIP 경로를 확인하세요.")
    DATA_ROOT = WORK / "datasets" / DATASET_NAME
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP) as archive:
        members = archive.infolist()
        total = sum(m.file_size for m in members)
        print("ZIP 압축 해제 크기: %.2f GiB" % (total / 1024**3))
        if total + 2 * 1024**3 > shutil.disk_usage("/content").free:
            raise RuntimeError("압축 해제와 결과 저장을 위한 임시 디스크 여유가 부족합니다.")
        base = DATA_ROOT.resolve()
        for member in members:
            target = (base / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError("ZIP에 폴더 밖을 가리키는 경로가 있습니다.")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP의 심볼릭 링크는 지원하지 않습니다.")
        archive.extractall(DATA_ROOT)
    if ANNOTATION_OVERRIDE:
        if SPLIT == "all":
            raise ValueError("전체 분할 모드에서는 ANNOTATION_OVERRIDE를 비워두세요.")
        ANNOTATIONS = Path(ANNOTATION_OVERRIDE)
    elif SPLIT == "all":
        sources = []
        for part in ("train", "valid", "test"):
            candidates = [p for p in DATA_ROOT.rglob("_annotations.coco.json") if p.parent.name == part]
            if len(candidates) != 1:
                raise ValueError(part + " 분할의 정답 파일이 정확히 한 개 필요합니다: " + str(candidates))
            sources.append((part, str(candidates[0])))
        ANNOTATIONS = DATA_ROOT / "all_annotations.coco.json"
        merge_code = "from metrics import merge_splits; merge_splits(%r, %r)" % (sources, str(ANNOTATIONS))
        checked([PYTHON, "-c", merge_code], ENV)
    else:
        candidates = [p for p in DATA_ROOT.rglob("_annotations.coco.json") if p.parent.name == SPLIT]
        if len(candidates) != 1:
            raise ValueError("분할의 정답 파일이 한 개로 정해지지 않습니다. ANNOTATION_OVERRIDE를 지정하세요. 후보: " +
                             str([str(p) for p in DATA_ROOT.rglob("_annotations.coco.json")]))
        ANNOTATIONS = candidates[0]
    if not ANNOTATIONS.is_file():
        raise FileNotFoundError("정답 JSON 파일이 없습니다.")
    print("선택한 데이터셋:", DATASET_NAME, "/ 분할:", SPLIT, "/ 정답:", ANNOTATIONS)
    if SPLIT == "train":
        print("학습 분할을 선택했습니다. 이 결과를 최종 테스트 성능으로 보고하지 마세요.")
    if SPLIT == "all":
        print("train·valid·test 통합 완료. 9단계에서 전체 이미지·대상 수를 확인하세요.")
else:
    print("데이터셋 준비 생략. 단일 이미지 결과만 다운로드할 수 있습니다.")
''')

md("""
## 9. 정답 라벨·이미지 크기·작은 객체 수 확인

기본 양성은 `person water` 한 종류입니다. `person_water` 표기도 같은 이름으로 정규화합니다. 실제 이름이 다르면 출력된 종류를 보고 수정하세요. `person sand`와 `umbrella`를 물속 사람으로 합치지 않습니다.

주 집계는 **`s=√(박스 가로×박스 세로)`의 `2≤s<20`, `2≤s<8` 구간**입니다. AnnotationPiriapolis v4의 세 분할 전체라면 이미지 **635장**, 물속 사람 **2,904개**, `2≤s<20` 정답 **2,823개**, `2≤s<8` 정답 **263개**가 나와야 합니다. 이전의 긴 변 기준 `d≤20`도 보조 지표로 유지하며, 두 정의는 서로 다릅니다.

두 크기는 모두 **현재 배포된 이미지 픽셀** 기준입니다. 원본을 축소한 데이터라면 원래 카메라 영상의 픽셀 크기와 같지 않을 수 있습니다. 배포 이미지를 임의로 640×640으로 다시 저장하지 마세요. 이미지 크기와 정답 좌표가 다르면 중단합니다.

해수면 ROI는 기본 적용하지 않습니다. SSPNet은 사람 1종 모델이므로 모래사장 사람도 검출할 수 있고, 물속 사람 평가에서 그 예측은 오탐입니다. 해수면 ROI를 도입한다면 정답 위치로 자동 생성하지 말고 사전에 정한 같은 ROI를 모든 모델에 적용하세요.
""")
code('''
TARGET_NAMES = ["person water"]
if PREPARE_DATASET:
    probe = """import json; from metrics import load_dataset
_, summary = load_dataset(%r, %r)
print(json.dumps(summary, ensure_ascii=False, indent=2))
""" % (str(ANNOTATIONS), TARGET_NAMES)
    checked([PYTHON, "-c", probe], ENV)
else:
    print("8단계에서 데이터셋을 먼저 준비하세요.")
''')

md("""
## 10. 작은 표본 평가 → 전체 평가

**전체 635장 평가는 `RUN_EVALUATION=True`, `MAX_IMAGES=0`으로 실행합니다.** 표본만 확인하려면 50으로 바꿀 수 있지만 이 경우 전체 결과가 아닙니다. 한 장씩 처리하고, 이미지가 끝날 때마다 완료 기록을 저장합니다. Drive를 켰다면 새 세션에서도 코드·동일 가중치·동일 데이터를 준비한 뒤 완료된 이미지를 재사용합니다.

캐시는 코드·모델 가중치·원본 정답·추론 설정·각 이미지의 SHA256을 검사합니다. 해상도·분할·가중치가 바뀌면 다른 캐시를 사용합니다. 보고 임계값만 바꾸면 저장된 저점수 예측을 재사용합니다. 같은 저장 폴더에서 동시에 두 평가를 실행하지 마세요.

평가 방식은 이 셀에서 직접 고릅니다. 기본 `full`은 사진 전체를 검사합니다. `tiles`는 기본 320픽셀 패치이며 겹침을 지정할 수 있습니다. 외부 문서에 겹침·입력 해상도가 공개되지 않았으므로 해당 설정을 정확하게 재현했다고 표시하지 않습니다. 각 방식은 별도 폴더에 저장합니다.
""")
code('''
RUN_EVALUATION = False #@param {type:"boolean"}
MAX_IMAGES = 0 #@param {type:"integer"}
EVALUATION_MODE = "full" #@param ["full", "tiles"]
EVALUATION_TILE_SIZE = 320 #@param {type:"integer"}
EVALUATION_TILE_OVERLAP = 0.25 #@param {type:"number"}
MATCH_IOU = 0.5 #@param {type:"number"}
if RUN_EVALUATION:
    if not PREPARE_DATASET or ANNOTATIONS is None:
        raise RuntimeError("8~9단계에서 데이터셋과 정답을 먼저 확인하세요.")
    EVAL_SETTINGS = dict(SETTINGS, target_names=TARGET_NAMES,
                         dataset_name=DATASET_NAME, split=SPLIT, mode=EVALUATION_MODE,
                         tile_size=EVALUATION_TILE_SIZE, tile_overlap=EVALUATION_TILE_OVERLAP,
                         match_iou=MATCH_IOU)
    EVAL_SETTINGS_PATH = WORK / "evaluation_settings.json"
    EVAL_SETTINGS_PATH.write_text(json.dumps(EVAL_SETTINGS), encoding="utf-8")
    EVAL_LABEL = EVALUATION_MODE if EVALUATION_MODE == "full" else "tiles_%d_overlap%g" % (EVALUATION_TILE_SIZE,EVALUATION_TILE_OVERLAP)
    EVAL_OUT = STORE / "evaluation" / DATASET_NAME / SPLIT / EVAL_LABEL
    checked([PYTHON, HELPERS / "engine.py", "--settings", EVAL_SETTINGS_PATH,
             "--annotations", ANNOTATIONS, "--limit", MAX_IMAGES, "--out", EVAL_OUT], ENV)
else:
    print("평가 생략. 준비되면 RUN_EVALUATION을 True로 바꾸세요.")
''')

md("""
## 11. 결과 읽는 방법

- **`문서용_결과요약.txt`**: 평가 범위·처리 방식·`2≤s<20` 및 `2≤s<8` 탐지 결과·정밀도·오탐·실제 임계값을 문서에 옮기기 쉬운 문장으로 저장합니다. 다른 모델의 결과나 비교표는 넣지 않습니다.
- **`geometric_2to20`, `geometric_2to8`**: `s=√(가로×세로)` 기준의 주 결과입니다. 각 항목의 `positives`는 정답 수, `hits`는 찾은 수, `recall`은 찾은 비율입니다.
- `metrics.json`·`metrics.csv`: 설정한 IoU와 점수 기준으로 전체 정밀도·재현율·오탐과 크기별 결과를 저장합니다. `by_split`에는 train·valid·test 각각의 결과도 남깁니다. 값은 0~1 비율이며 `null`은 분모가 없어 계산할 수 없다는 뜻입니다.
- `AP50_saved_predictions_101point`: 저장 점수 하한 이상·최대 탐지 개수 이내 예측에서 계산한 101점 보간 AP50입니다. 원 논문 TinyPerson의 별도 지표나 COCO mAP50:95와 혼동하지 마세요. AP50은 화면 표시 임계값과 별도로 저장된 전체 점수 범위를 사용합니다.
- 큰 물속 사람에 맞은 예측은 전체에서 정상 매칭한 뒤 작은 사람 재현율의 분모·분자에서 제외합니다. 중복 탐지와 사람이 없는 이미지의 예측은 오탐에 포함합니다.
- `predictions_coco.json`: 원본 이미지 좌표의 COCO 예측. `ground_truth_water.json`은 물속 사람만 남긴 정답이며 종류 ID는 1입니다. 다른 모델과 공통 평가기를 쓰는 데 활용할 수 있습니다.
- `run_manifest.json`·`resolved_config.py`·`installed_packages.txt`: 코드 버전, 가중치 해시, 실제 GPU, 이미지 ID, 해상도·분할·점수·탐지 개수 설정을 기록합니다.
- 처리 시간은 준비 호출 후의 추론·패치 병합 시간을 측정하고 이미지 읽기·파일 저장 시간은 제외합니다. GPU가 다르면 모델 속도를 직접 비교할 수 없습니다. 캐시를 재사용한 결과의 시간은 원래 추론 당시 측정값입니다.

문서에는 전체 분할 635장 평가인지와 `s=√(가로×세로)` 정의를 명시하세요. `s<20`은 가로·세로 모두 20 이하와 다릅니다. `tiny_le20`은 기존 긴 변 기준의 보조 지표입니다. 전체 분할 평가와 독립 test 분할 평가는 구분해 기록하세요.
""")
code('''
if RUN_EVALUATION:
    result = json.loads((EVAL_OUT / "metrics.json").read_text(encoding="utf-8"))
    display(result)
    print("주 결과 2≤√(가로×세로)<20:", result["geometric_2to20"])
    print("가장 작은 구간 2≤√(가로×세로)<8:", result["geometric_2to8"])
    print((EVAL_OUT / "문서용_결과요약.txt").read_text(encoding="utf-8"))
    if result["partial"]:
        print("현재는 표본 평가입니다. 전체 평가에는 MAX_IMAGES=0을 사용하세요.")
''')

md("""
## 12. 결과 ZIP 다운로드

기본적으로 모델 가중치·원본 이미지·설치 환경은 ZIP에 넣지 않습니다. 결과 이미지, 예측, 평가표, 재시작 캐시, 설정을 포함합니다. Drive를 사용하지 않았다면 무료 런타임이 종료되기 전에 이 셀로 내려받으세요. 파일 다운로드는 브라우저에서 허용해야 할 수 있습니다.

중단된 평가를 Drive 없이 복구할 때는 내려받은 ZIP의 `evaluation/` 폴더를 새 세션의 `STORE` 아래에 복원하고, 같은 데이터·가중치·설정을 다시 준비하세요.
""")
code('''
DOWNLOAD_RESULTS = True #@param {type:"boolean"}
RESULT_ZIP = WORK / "SSPNet_results.zip"
with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(STORE.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pth", ".pt", ".tmp"):
            archive.write(path, path.relative_to(STORE))
print("결과 ZIP:", RESULT_ZIP, "%.2f MiB" % (RESULT_ZIP.stat().st_size / 1024**2))
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(RESULT_ZIP))
''')

md("""
## 오류가 나면

| 증상 | 확인·대응 |
|---|---|
| GPU가 없음 | 무료 GPU가 배정될 때 다시 연결. 유료 업그레이드가 필수인 노트북이 아님 |
| Google Drive 다운로드 제한 | 5단계의 직접 업로드 또는 기존 가중치 경로 사용 |
| `No module named mmcv` | 기본 Python에서 실행했는지 확인. 모든 모델 실행은 4단계에서 만든 `PYTHON` 사용 |
| CUDA 연산·아키텍처 오류 | 4단계 GPU 검사 출력 확인. 다른 무료 런타임에서 재시도할 수 있으나 성공 보장은 없음 |
| 가중치 파라미터 불일치 | Faster R-CNN SSPNet용 가중치인지 확인. 오류를 무시해 임의 가중치로 평가하지 않음 |
| CUDA 메모리 부족 | 단일 이미지·낮은 해상도부터 확인. 변경된 해상도는 별도 실험으로 기록 |
| 물속 사람 라벨 없음 | 9단계의 실제 종류 이름을 보고 TARGET_NAMES 수정. 육상 사람을 임의로 합치지 않음 |
| test 분할 없음 | 검증 분할 사용 여부를 팀과 정하고 명시. train으로 자동 대체하지 않음 |
| 세션 중단 | 재설치 후 같은 설정으로 재실행. Drive에 저장된 완료 이미지부터 이어감 |

설치 참고: [SSPNet](https://github.com/MingboHong/SSPNet), [PyTorch 구버전](https://pytorch.org/get-started/previous-versions/), [CUDA 11.1·PyTorch 1.10용 MMCV wheel](https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html), [uv Python 설치](https://docs.astral.sh/uv/guides/install-python/), [Colab 사용 제한](https://research.google.com/colaboratory/faq.html).
""")

NOTEBOOK = dict(nbformat=4, nbformat_minor=5, cells=CELLS,
                metadata=dict(colab=dict(name="SSPNet_무료_Colab_추론과_수영자평가.ipynb", provenance=[]),
                              accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python", version="3")))
OUT = ROOT / "notebooks/SSPNet_무료_Colab_추론과_수영자평가.ipynb"
OUT.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(str(OUT))
print("셀 수:", len(CELLS))

# 이미 환경을 구성한 Colab 런타임에서 추가 셀 한 개로 갱신할 수 있게 한다.
update_cell = '''# 기존 무료 Colab의 새 코드 셀에 이 파일 전체를 붙여넣고 실행하세요.
# 현재 세션의 SSPNet 환경·가중치·데이터 ZIP을 재사용합니다.
import os, sys, json, shutil, subprocess, zipfile
from pathlib import Path

EVALUATION_MODE = "full" #@param ["full", "tiles"]
TILE_SIZE = 320 #@param {type:"integer"}
TILE_OVERLAP = 0.25 #@param {type:"number"}
REPORT_SCORE = 0.25 #@param {type:"number"}
MATCH_IOU = 0.5 #@param {type:"number"}

WORK = Path("/content/sspnet_free")
PYTHON = WORK / "venv/bin/python"
REPO = WORK / "SSPNet"
CHECKPOINT = WORK / "faster_rcnn_sspnet.pth"
if not PYTHON.is_file() or not CHECKPOINT.is_file() or not REPO.is_dir():
    raise RuntimeError("기존 노트북의 1~7단계를 완료한 같은 런타임에서 실행하세요.")
STORE = Path(globals().get("STORE", WORK / "saved"))
STORE.mkdir(parents=True, exist_ok=True)
HELPERS = WORK / "helpers"
HELPERS.mkdir(exist_ok=True)
'''
update_cell += "HELPER_SOURCES = " + repr(embedded) + '''
for name, source in HELPER_SOURCES.items():
    (HELPERS / name).write_text(source, encoding="utf-8")
ENV = os.environ.copy()
ENV.pop("PYTHONHOME", None)
ENV["PYTHONNOUSERSITE"] = "1"
ENV["PYTHONPATH"] = str(REPO) + os.pathsep + str(HELPERS)
ENV["MPLBACKEND"] = "Agg"
def checked(command):
    subprocess.run([str(x) for x in command], check=True, env=ENV)

DATA_ZIP = WORK / "AnnotationPiriapolis_v4.zip"
if not DATA_ZIP.is_file():
    from google.colab import files
    print("기존 데이터 ZIP이 없습니다. AnnotationPiriapolis v4의 COCO ZIP을 선택하세요.")
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("데이터 ZIP 한 개만 선택하세요.")
    DATA_ZIP.write_bytes(next(iter(uploaded.values())))
DATA_ROOT = WORK / "datasets/AnnotationPiriapolis_v4"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(DATA_ZIP) as archive:
    members = archive.infolist()
    total = sum(m.file_size for m in members)
    if total + 2*1024**3 > shutil.disk_usage("/content").free:
        raise RuntimeError("압축 해제에 필요한 디스크 여유가 부족합니다.")
    base = DATA_ROOT.resolve()
    for member in members:
        target = (base / member.filename).resolve()
        if target != base and base not in target.parents:
            raise ValueError("ZIP 경로가 데이터 폴더 밖을 가리킵니다.")
        if (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("ZIP의 심볼릭 링크는 지원하지 않습니다.")
    archive.extractall(DATA_ROOT)
sources = []
for part in ("train", "valid", "test"):
    candidates = [p for p in DATA_ROOT.rglob("_annotations.coco.json") if p.parent.name == part]
    if len(candidates) != 1:
        raise ValueError(part + " 분할의 정답 JSON을 한 개 찾을 수 없습니다: " + str(candidates))
    sources.append((part, str(candidates[0])))
ANNOTATIONS = DATA_ROOT / "all_annotations.coco.json"
prepare = "from metrics import merge_splits,load_dataset; import json; " + \\
    "p=merge_splits(%r,%r); _,s=load_dataset(p); print(json.dumps(s,ensure_ascii=False,indent=2)); " % (sources,str(ANNOTATIONS)) + \\
    "assert (s['image_count'],s['target_count'],s['geometric_2to20_count'],s['geometric_2to8_count'])==(635,2904,2823,263), '데이터 버전·분할·종류를 확인하세요.'"
checked([PYTHON, "-c", prepare])
settings = dict(globals().get("SETTINGS", {}))
settings.update(repo=str(REPO), checkpoint=str(CHECKPOINT), mode=EVALUATION_MODE,
    long_edge=settings.get("long_edge",1280), short_edge=settings.get("short_edge",1024),
    tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP, merge_iou=0.5,
    report_score=REPORT_SCORE, save_min_score=0.001, max_detections=500,
    target_names=["person water"], dataset_name="AnnotationPiriapolis_v4", split="all",match_iou=MATCH_IOU)
settings_path = WORK / "evaluation_all635_settings.json"
settings_path.write_text(json.dumps(settings),encoding="utf-8")
label = EVALUATION_MODE if EVALUATION_MODE == "full" else "tiles_%d_overlap%g" % (TILE_SIZE,TILE_OVERLAP)
out = STORE / "evaluation/AnnotationPiriapolis_v4/all" / label
print("전체 635장 평가를 시작합니다. 이번 모드:", label)
checked([PYTHON, HELPERS / "engine.py", "--settings", settings_path,
         "--annotations", ANNOTATIONS, "--limit", "0", "--out", out])
print((out / "문서용_결과요약.txt").read_text(encoding="utf-8"))
result_zip = WORK / ("SSPNet_all635_" + label + "_results.zip")
with zipfile.ZipFile(result_zip,"w",compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix != ".tmp":
            archive.write(path,path.relative_to(STORE))
    if (STORE / "installed_packages.txt").is_file():
        archive.write(STORE / "installed_packages.txt","installed_packages.txt")
print("결과 ZIP:", result_zip)
from google.colab import files
files.download(str(result_zip))
'''
(ROOT/"notebooks/기존_Colab_전체635장_평가_업데이트셀.txt").write_text(update_cell,encoding="utf-8")
