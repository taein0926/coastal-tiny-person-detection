"""사용자가 선택한 전체 이미지 두 조합만 실행한다. 분할 전체 평가는 보류한다."""
import json
from pathlib import Path
import evaluate_seadronessee as ev

def main():
    gate=json.loads((ev.OUT/'pilot_full_approval.json').read_text())
    assert gate['passed'] and gate['scope']==['public/full','finetuned/full']
    assert gate['pilot_ids']==ev.PILOT
    gt,manifest=ev.load()
    results={}
    for label in ('public','finetuned'):
        results[label+'_full']=ev.run_combo(label,'full',gt,manifest,'full')
    ev.dump(ev.OUT/'comparison_full_images.json',results)
    ev.dump(ev.OUT/'full_images_status.json',dict(state='완료',completed_combinations=2,images_per_combination=1547,tiles='사용자 요청으로 전체 평가 보류'))
    print('전체 이미지 두 조합 평가 완료. 분할 전체는 보류.',flush=True)

if __name__=='__main__': main()
