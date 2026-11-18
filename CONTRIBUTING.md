# 커밋 작성 규칙

Conventional Commits를 사용하며 타입은 영어, 제목은 한국어로 작성한다.

```text
<type>: <한국어 제목>
```

| 타입 | 용도 |
|---|---|
| feat | 기능 추가 |
| fix | 오류 수정 |
| docs | 문서·실험 결과 설명 |
| test | 검증 코드 추가·수정 |
| refactor | 동작을 유지하는 코드 구조 변경 |
| chore | 저장소·의존성·실행 환경 관리 |

서로 독립적인 기능은 별도 커밋으로 나눈다. 예: `feat: SSPNet 전체 이미지와 분할 추론 구현`, `docs: 공개 모델 비교 결과 정리`.

## CPU 평가 로직 검사

```text
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py"
python scripts/validate_notebook.py
```

위 검사는 매칭·크기 경계·중복 제거·좌표 변환 등의 로직을 확인한다. 실제 GPU 학습·추론 성능을 검증하는 실행과 구분한다.
