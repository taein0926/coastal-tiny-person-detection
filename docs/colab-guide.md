# Colab 실행 안내

## 추론과 평가

1. [추론·평가 노트북](../notebooks/SSPNet_무료_Colab_추론과_수영자평가.ipynb)을 Google Colab에 업로드한다.
2. GPU 런타임을 선택하고 환경 설치·공개 가중치 준비·예제 추론을 순서대로 실행한다.
3. 사용자 이미지는 업로드 방식으로 지정한다. 데이터셋 평가는 해당 버전의 COCO JSON ZIP을 준비한다.
4. AnnotationPiriapolis 전체 635장 범위는 `SPLIT=all`, `MAX_IMAGES=0`을 사용한다. 전체·분할 추론 설정을 구분한다.
5. 결과를 유지하려면 Google Drive 저장을 설정하거나 마지막 셀에서 결과 ZIP을 내려받는다.

이미 실행 중인 초기 노트북을 갱신하려면 [전체 평가 업데이트 셀](../notebooks/기존_Colab_전체635장_평가_업데이트셀.txt)을 참고한다. [통합 평가 노트북](../notebooks/SSPNet_통합평가_무료Colab.ipynb)도 함께 제공한다.

## README의 5 epoch 학습 재구성

[보존 학습 노트북](../notebooks/SSPNet_해변_5epoch_학습_보존본.ipynb)은 `train_legacy.py`를 내장한다. YoloBeachTrain v2 전체를 학습에 사용한 기존 방식에 대응하며, 별도 검증 분할을 이용하는 후속 학습과 구분한다. 세부 조건은 [실험 안내](experiment-guide.md)를 참고한다.

`SSPNet_추가학습_무료Colab.ipynb`와 `SSPNet_재현율우선_recall_v2_Colab.ipynb`는 검수된 데이터와 검증 기반 선택을 요구하는 후속 학습 절차다. 이 노트북을 실행한 결과가 README의 기존 5 epoch 결과와 같다고 간주하지 않는다.

무료 Colab GPU 할당과 실행 지속 시간은 보장되지 않는다. 설치 검사와 모델 실행 완료 여부는 실제 세션의 출력으로 확인한다.
