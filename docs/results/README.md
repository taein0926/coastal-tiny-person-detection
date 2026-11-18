# README 결과표의 근거

저장된 실험 결과에서 README에 필요한 집계 파일을 모았다. 새로운 추론·추가 학습 결과가 아니다. 지표의 원본 경로와 SHA-256은 [provenance.json](provenance.json)에 기록했다.

## SeaDronesSee OD v2 validation

전체 1,547장, swimmer 정답 6,206개, 매칭 IoU 0.3, 전체 이미지 추론이다.

| 파일 | 모델·설정 |
|---|---|
| [sea-public-025.json](sea-public-025.json) | 공개 SSPNet, 신뢰도 0.25 |
| [sea-public-005.json](sea-public-005.json) | 공개 SSPNet, 신뢰도 0.05 |
| [sea-finetuned-025.json](sea-finetuned-025.json) | 5 epoch 추가 학습 SSPNet, 신뢰도 0.25 |
| [sea-yolo26m-025.json](sea-yolo26m-025.json) | 공개 YOLO26m, 신뢰도 0.25 |
| [sea-yolo26m-005.json](sea-yolo26m-005.json) | 공개 YOLO26m, 신뢰도 0.05 |
| [sspnet-thresholds.json](sspnet-thresholds.json) | 저장된 공개 SSPNet 예측을 0.25·0.20·0.15·0.10·0.05로 재채점한 집계 |

`geometric_2to20`은 원본 배포 이미지에서 `2 ≤ √(박스 가로×세로) < 20`픽셀인 정답 339개의 결과다. `tiny_le20` 등 다른 크기 정의와 혼합하지 않는다. `AP50_saved_predictions_101point`는 이 평가에서 사용하지 않으므로 README를 mAP 결과로 해석하지 않는다.

## AnnotationPiriapolis v4

전체 635장, 전체 사람 정답 8,099개, 신뢰도 0.25, 매칭 IoU 0.3이다.

| 파일 | 모델·추론 |
|---|---|
| [beach-public-full.json](beach-public-full.json) | 공개 SSPNet, 전체 이미지 |
| [beach-public-tiles.json](beach-public-tiles.json) | 공개 SSPNet, 320×320 분할 |
| [beach-finetuned-full.json](beach-finetuned-full.json) | 추가 학습 SSPNet, 전체 이미지 |
| [beach-finetuned-tiles.json](beach-finetuned-tiles.json) | 추가 학습 SSPNet, 320×320 분할 |
| [beach-water-only.json](beach-water-only.json) | 물속 사람 정답 2,904개에 별도로 재매칭한 네 조합 |

추가 학습 모델의 두 파일은 원본 `result.json`의 `all_people` 집계를 추출했다. 원본의 `water_only` 집계와 구분한다. 분할 추론의 겹침은 20%, 병합 NMS IoU는 0.5다.

## 미디어 출처

루트 README는 「YOLO26m과 SSPNet 공개 모델 비교」와 「SSPNet 모델의 해변 미세 객체 탐지 테스트」의 본문·결과표·이미지 캡션을 참고했다. 첨부 파일은 다운로드하지 않고 기존 로컬 자료를 사용했다.

- `public-models-overview.jpg`: 기존 SeaDronesSee 이미지 1018 원본·확대·모델 비교 시각화.
- `small-person-thresholds.jpg`: 기존 SeaDronesSee 이미지 526의 두 모델·신뢰도 비교 시각화.
- `beach-*`: SSPNet 테스트 문서에 사용한 AnnotationPiriapolis 이미지 604의 공개·추가 학습 예측.
- `external-*`: 같은 문서에 사용한 SeaDronesSee 이미지 1018의 공개·추가 학습 예측.
- `public-models-19953301.mp4`, `public-models-37277130.mp4`: 관련 로컬 공개 모델 비교 영상. 전체 길이·해상도를 유지하며 15fps H.264로 압축했고, 각 파일은 10 MB 미만이다. 참고 문서에 업로드했던 파일과의 바이너리 동일성은 확인하지 않았다.
- [video-attachments.json](../assets/video-attachments.json): README 플레이어에 사용하는 GitHub 첨부 주소, 압축 설정, 파일 크기와 SHA-256. 첨부 주소를 단독 줄에 배치하며, 저장소의 MP4 링크도 함께 제공한다.

예시 영상은 왼쪽 YOLO26m, 오른쪽 SSPNet이며 모두 신뢰도 0.25다. 표의 정량 평가 집합과 영상 예시는 별개다. 이미지·영상의 저작권과 이용 조건은 원본 데이터·영상 출처에 따른다.
