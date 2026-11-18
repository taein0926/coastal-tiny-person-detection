# 실험별 실행 환경과 경로

루트 README는 「YOLO26m과 SSPNet 공개 모델 비교」와 「SSPNet 모델의 해변 미세 객체 탐지 테스트」의 완료된 실험을 정리한다. 이 문서는 해당 결과와 코드의 대응 관계를 설명한다.

## 데이터와 평가 범위

| 용도 | 데이터 | 범위 |
|---|---|---|
| 5 epoch 추가 학습 | YoloBeachTrain v2 | train·valid·test 전체 2,961장, 증강본 포함 |
| 해변 외부 평가 | AnnotationPiriapolis v4 | 전체 635장, 전체 사람 8,099개 / 물속 사람 2,904개 |
| 해상 외부 평가·공개 모델 비교 | SeaDronesSee OD v2 validation | 전체 1,547장, swimmer 6,206개 |

AnnotationPiriapolis의 배포본 train·valid·test는 평가 범위를 나타내며 이 데이터로 SSPNet을 학습했다는 뜻이 아니다. YoloBeachTrain과 AnnotationPiriapolis의 사전 중복 점검은 `analysis_results/dataset_independence_20261111/`에 보존돼 있다.

## SSPNet

원본 SSPNet 소스는 `.sspnet_source/`에 위치한다. SeaDronesSee 실행 기록의 원본 커밋은 `f2d0f42273945577a5c6497fd2b384cb545f8f66`이다. [Dockerfile](../local_gpu/Dockerfile)의 기본 환경은 PyTorch 1.10.0, CUDA 11.3, MMCV-full 1.3.18이다. Colab용 별도 환경은 [setup_environment.py](../sspnet_colab/setup_environment.py)를 참고한다.

새 체크아웃에서는 아래 명령으로 원본 코드를 별도로 준비한다. `.sspnet_source/`는 이 저장소의 커밋 대상에서 제외한다.

```text
git clone https://github.com/MingboHong/SSPNet.git .sspnet_source
git -C .sspnet_source checkout f2d0f42273945577a5c6497fd2b384cb545f8f66
```

| 가중치 | SHA-256 |
|---|---|
| 공개 SSPNet | `912cc5b6f8a45b0f4521c9bf325868ed10dd4757497fb6ba8843fdbd26b09e4a` |
| 추가 학습 epoch 5 | `690f943c0a01fe40954b823942a65f64df94bb2157766156fa731d5a89f249a1` |

공통 추론 설정은 리사이즈 `(1280, 1024)`, 종횡비 유지, 최대 탐지 수 500이다. 분할 추론은 원본 이미지에서 320×320 타일·20% 겹침으로 검사하고 NMS IoU 0.5로 병합한다. 보고 신뢰도 0.25와 정답 매칭 IoU 0.3은 별도로 적용한다.

### 추가 학습

README의 학습 결과에 대응하는 보존 코드는 [train_legacy.py](../sspnet_colab/train_legacy.py)와 [build_training_notebook_legacy.py](../scripts/build_training_notebook_legacy.py)다.

별도 [5 epoch 보존 학습 노트북](../notebooks/SSPNet_해변_5epoch_학습_보존본.ipynb)은 `train_legacy.py`를 내장한다. 현재의 후속 학습 노트북과 파일명을 분리했다.

- `person water`와 `person sand`를 `person`으로 합친다.
- seed 0, batch size 1, 5 epoch, SGD, 학습률 0.0001, momentum 0.9, weight decay 0.0001을 사용한다.
- 별도 검증 기반 체크포인트 선택 없이 최종 epoch 5를 평가했다.
- 빈 정답에서 발생하는 sampler 오류와 heatmap 인덱스 처리를 학습 소스에서 보완한다.

현재 `train.py`와 `SSPNet_추가학습_무료Colab.ipynb`는 후속 재현율 우선 학습용으로 변경됐다. 실행 시 현재 노트북의 결과를 기존 5 epoch 결과로 간주하지 않는다.

### 평가 진입점

| 평가 | 코드·설정 | 기존 결과 경로 |
|---|---|---|
| AnnotationPiriapolis 공개 전체·분할 | [Colab 평가 노트북](../notebooks/SSPNet_무료_Colab_추론과_수영자평가.ipynb), [평가 함수](../sspnet_colab/metrics.py) | `analysis_results/tiles_comparison_20261111/` |
| AnnotationPiriapolis 추가 학습 전체·분할 | [공개 평가 실행기](../scripts/evaluate_piriapolis.py), [Compose](../local_gpu/compose.yaml) | `analysis_results/local_gpu_smoke/evaluation_full_all/`, `evaluation_tiles_all/` |
| SeaDronesSee 전체 이미지 공개·추가 학습 | [전체 이미지 실행기](../scripts/run_seadronessee_full_images.py), [평가 구현](../scripts/evaluate_seadronessee.py) | `analysis_results/seadronessee_validation/public/full/`, `finetuned/full/` |
| SSPNet 신뢰도 0.05 | [저장 예측 재채점](../scripts/rescore_sspnet_005.py) | `analysis_results/seadronessee_validation/public/full/metrics_score_005.json` |

SeaDronesSee의 [전체 이미지 실행기](../scripts/run_seadronessee_full_images.py)는 `pilot_full_approval.json`의 기존 소량 점검 기록을 확인한다. 일반 평가 스크립트는 기본 소량 점검이며 전체·분할 네 조합을 처리하므로, README의 전체 이미지 두 조합과 범위를 구분해야 한다. Compose의 명령과 로컬 마운트 경로도 목적에 맞게 확인한다.

AnnotationPiriapolis의 Compose는 `data/AnnotationPiriapolis.v4i.coco.zip`을 기본 입력으로 사용한다. 다른 위치에 있다면 `ANNOTATION_PIRIAPOLIS_ZIP` 환경 변수에 ZIP의 절대 경로를 지정한다. 기본 명령은 전체 이미지 전체 635장 평가이며, 분할은 실행기의 `--mode tiles --limit 0`을 사용한다. 공개한 `evaluate_piriapolis.py`는 기존 `run_smoke.py`의 구현을 보존한 파일이다.

## YOLO26m

기록된 환경은 Ultralytics 8.4.215, ONNX Runtime 1.24.4, ONNX Runtime의 DirectML NVIDIA GPU 실행이다. PyTorch 패키지 기록은 `2.15.0+cpu`지만 실제 평가 추론은 ONNX Runtime으로 수행했다.

| 항목 | 설정 |
|---|---|
| 가중치 | 공개 `yolo26m.pt`, 추가 학습 없음 |
| PT SHA-256 | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| ONNX SHA-256 | `b3bcb442be867d5652e38bc8a19692d02aaf88531c261f1de1975140100afc3a` |
| 입력 | `imgsz=1280`, 전체 이미지 |
| 클래스 | COCO person, `classes=[0]` |
| 최대 탐지 | 500 |
| 신뢰도 | 환경 변수 `YOLO_SCORE=0.25` 또는 `0.05` |
| 장치 | ONNX Runtime `DmlExecutionProvider`, 기록된 어댑터 번호 1 |

[평가 실행기](../scripts/evaluate_yolo26_seadronessee.py)는 `.pt`·`.onnx`, 원본 주석, 이미지 목록의 해시를 확인하며 중간 캐시를 재사용한다. 어댑터 번호는 컴퓨터마다 다르므로 경로와 함께 조정해야 한다.

## 미디어와 표의 근거

[결과 파일 안내](results/README.md)에서 각 표의 JSON을 확인할 수 있다. [provenance.json](results/provenance.json)에는 README용 파일의 로컬 원본 경로와 SHA-256을 기록했다. 이미지 원본은 복제했고, 비교 영상은 해상도·길이·프레임률을 유지해 재인코딩했다. 첫 프레임은 영상 링크의 미리보기로 사용했다.

실험 코드에는 당시의 고정 경로와 장치 설정이 남아 있다. 이 문서는 기존 결과를 추적하기 위한 안내이며 새 환경에서 모든 실험을 재실행해 검증했다는 뜻은 아니다. 원본 데이터·가중치는 각 출처에서 확보해야 한다.
