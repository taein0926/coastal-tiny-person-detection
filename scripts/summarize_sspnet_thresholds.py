"""기존 SSPNet 예측을 재채점해 README 신뢰도 표의 근거를 저장한다. 추론은 실행하지 않는다."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sspnet_colab'))
from metrics import evaluate


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    truth_path = ROOT / 'analysis_results/seadronessee_validation/ground_truth_swimmer.json'
    prediction_path = ROOT / 'analysis_results/seadronessee_validation/public/full/predictions_full.json'
    truth = json.loads(truth_path.read_text(encoding='utf-8-sig'))
    predictions = json.loads(prediction_path.read_text(encoding='utf-8-sig'))
    rows = []
    for score in (0.25, 0.20, 0.15, 0.10, 0.05):
        metric = evaluate(truth, predictions, score, 0.3)
        rows.append({key: metric[key] for key in (
            'score_threshold', 'iou', 'images', 'positives', 'true_positives',
            'false_positives', 'precision', 'recall', 'false_positives_per_image')})
        print(f"신뢰도 {score:.2f}: TP={metric['true_positives']}, FP={metric['false_positives']}", flush=True)
    assert (rows[0]['images'], rows[0]['positives'], rows[0]['true_positives'], rows[0]['false_positives']) == (1547, 6206, 4839, 5980)
    assert rows[-1]['true_positives'] == 5601
    result = {
        'description': '저장된 공개 SSPNet 전체 이미지 예측의 신뢰도별 재채점',
        'sources': [{'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
                    for path in (truth_path, prediction_path, ROOT / 'sspnet_colab/metrics.py', Path(__file__))],
        'rows': rows,
    }
    output = ROOT / 'docs/results/sspnet-thresholds.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
