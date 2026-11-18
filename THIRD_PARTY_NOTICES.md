# 외부 코드·모델·데이터 출처

이 저장소는 공개 모델을 실행·추가 학습·평가하기 위한 코드와 실험 기록을 제공한다. 원본 모델·가중치·데이터셋 전체는 포함하지 않는다.

- **SSPNet / MMDetection 기반 구현**: [MingboHong/SSPNet](https://github.com/MingboHong/SSPNet). 실험 소스 커밋은 `f2d0f42273945577a5c6497fd2b384cb545f8f66`이며, 해당 저장소의 Apache-2.0 고지와 이용 조건을 따른다. 학습 시 필요한 수정은 `train_legacy.py` 및 `train.py`에 명시돼 있다.
- **Ultralytics YOLO26m**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). 관련 코드·모델의 AGPL-3.0 또는 별도 Enterprise 이용 조건은 [공식 안내](https://www.ultralytics.com/license)를 따른다.
- **SeaDronesSee**: [공식 저장소](https://github.com/Ben93kie/SeaDronesSee). README의 예시 이미지는 해당 데이터의 swimmer 주석과 모델 예측을 겹쳐 시각화했다.
- **YoloBeachTrain v2 / AnnotationPiriapolis v4**: 데이터 출처 및 배포 조건은 원본 배포 페이지와 [연구팀의 데이터 소개](https://research.wur.nl/en/datasets/object-detection-models-training-datasets-and-scripts-for-watchin/)를 따른다. README의 해변 예시는 AnnotationPiriapolis 이미지에서 파생됐다.
- **비교 영상 원본**: [Pexels 19953301](https://www.pexels.com/video/19953301/), [Pexels 37277130](https://www.pexels.com/video/37277130/). 원본 프레임에 각 모델의 예측 박스를 표시하고 두 화면을 병치했다.

공개 저장소라는 사실이 외부 코드·가중치·데이터·영상에 새로운 이용 권한을 부여하지는 않는다. 각 자료의 출처 고지와 이용 조건은 별도로 유지된다.
