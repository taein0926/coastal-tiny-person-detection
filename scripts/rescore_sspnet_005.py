import json, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'sspnet_colab'))
from metrics import evaluate
base=root/'analysis_results/seadronessee_validation'
gt=json.loads((base/'ground_truth_swimmer.json').read_text(encoding='utf-8'))
preds=json.loads((base/'public/full/predictions_full.json').read_text(encoding='utf-8'))
check=evaluate(gt,preds,.25,.3)
assert (check['true_positives'],check['false_positives'],check['positives'])==(4839,5980,6206)
result=evaluate(gt,preds,.05,.3)
result['false_negatives']=result['positives']-result['true_positives']
path=base/'public/full/metrics_score_005.json'
path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
