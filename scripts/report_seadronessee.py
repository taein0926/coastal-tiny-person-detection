"""저장된 결과만 읽어 전체/소량 평가를 구분한 비교 보고서를 만든다."""
import csv
import html
import json
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_results/seadronessee_validation'
sys.path.insert(0,str(ROOT/'sspnet_colab'))
from metrics import match_predictions

LABELS={'public':'공개 SSPNet','finetuned':'추가 학습 SSPNet','full':'전체 이미지','tiles':'분할 추론'}

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def percent(x): return '계산 불가' if x is None else f'{x*100:.2f}%'

def main():
    rows=[]; pilot_rows=[]; details=[]; estimates={}
    scope=read(OUT/'evaluation_scope.json') if (OUT/'evaluation_scope.json').exists() else None
    for label in ('public','finetuned'):
        for mode in ('full','tiles'):
            folder=OUT/label/mode
            for phase,target in [('pilot',pilot_rows),('full',rows)]:
                path=folder/('metrics_'+phase+'.json')
                if not path.exists(): continue
                m=read(path); timing=m['timing']
                target.append(dict(model=LABELS[label],mode=LABELS[mode],images=m['images'],positives=m['positives'],
                    TP=m['true_positives'],FP=m['false_positives'],FN=m['false_negatives'],recall=m['recall'],precision=m['precision'],
                    FP_per_image=m['false_positives_per_image'],mean_seconds=timing['mean_inference_seconds'],
                    mean_wall_seconds=timing['total_measured_wall_seconds']/m['images'],
                    measured_wall_hours=timing['total_measured_wall_seconds']/3600,
                    inference_hours=timing['total_inference_seconds']/3600,peak_vram_mib=timing['maximum_vram_mib'],
                    patches=timing['total_patches']))
                if phase=='full':
                    details.append('<h3>'+LABELS[label]+' / '+LABELS[mode]+'</h3>'+size_table(m))
            if (folder/'metrics_pilot.json').exists():
                recs=[read(folder/'cache'/f'{i}.json') for i in read(OUT/'protocol.json')['pilot_ids']]
                ratios=sorted(r['seconds']/r['patches'] for r in recs)
                median=(ratios[1]+ratios[2])/2
                estimates[label+'_'+mode]=dict(seconds_per_patch_median=median,
                    estimated_inference_hours=(208934 if mode=='tiles' else 1547)*median/3600,
                    note='4장 기반 추정이며 밀도·NMS·해상도별 차이로 실제와 달라질 수 있음.')
    summaries={}; model_comparison=None
    gtpath=OUT/'ground_truth_swimmer.json'
    if gtpath.exists():
        gt=read(gtpath)
        for label in ('public','finetuned'):
            paths=[OUT/label/m/('predictions_full.json') for m in ('full','tiles')]
            if not all(p.exists() for p in paths): continue
            hitsets=[]
            for p in paths:
                hits={a['id'] for pred,a,ignored in match_predictions(gt,read(p),.3)
                      if a is not None and not ignored and pred['score']>=.25}
                hitsets.append(hits)
            a,b=hitsets; allids={x['id'] for x in gt['annotations']}
            summaries[label]=dict(both_hit=len(a&b),tiles_new_hit=len(b-a),tiles_lost_hit=len(a-b),both_missed=len(allids-a-b))
            with (OUT/(label+'_annotation_comparison.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f); writer.writerow(['annotation_id','image_id','full_hit','tiles_hit','status'])
                for item in gt['annotations']:
                    ident=item['id']; status='공통 정탐' if ident in a&b else '분할 신규 정탐' if ident in b-a else '분할에서 놓침' if ident in a-b else '공통 미탐'
                    writer.writerow([ident,item['image_id'],ident in a,ident in b,status])
        paths=[OUT/label/'full/predictions_full.json' for label in ('public','finetuned')]
        if all(p.exists() for p in paths):
            hitsets=[{a['id'] for pred,a,ignored in match_predictions(gt,read(p),.3) if a is not None and not ignored and pred['score']>=.25} for p in paths]
            a,b=hitsets; allids={x['id'] for x in gt['annotations']}
            model_comparison=dict(both_hit=len(a&b),public_only_hit=len(a-b),finetuned_only_hit=len(b-a),both_missed=len(allids-a-b))
            with (OUT/'full_models_annotation_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f); writer.writerow(['annotation_id','image_id','public_hit','finetuned_hit','status'])
                for item in gt['annotations']:
                    ident=item['id']; status='공통 정탐' if ident in a&b else '공개만 정탐' if ident in a-b else '추가 학습만 정탐' if ident in b-a else '공통 미탐'
                    writer.writerow([ident,item['image_id'],ident in a,ident in b,status])
    for name,data in [('comparison.csv',rows),('pilot_comparison.csv',pilot_rows)]:
        if data:
            with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    if scope and scope['active_combinations']==['public/full','finetuned/full']:
        status=('전체 이미지 두 조합 평가 완료' if len(rows)==2 else f'전체 이미지 평가 진행 중: 완료 {len(rows)}/2조합')+' / 분할 전체 보류'
    else:
        status='전체 평가 완료' if len(rows)==4 else f'전체 평가 진행 중: 완료 {len(rows)}/4조합'
    run_clock=None
    clock_path=OUT/'run_clock.json'; done_path=OUT/'full_images_status.json'
    if clock_path.exists() and done_path.exists() and read(done_path)['state']=='완료':
        started=datetime.fromisoformat(read(clock_path)['started_utc'].replace('Z','+00:00'))
        ended=datetime.fromtimestamp(done_path.stat().st_mtime,tz=timezone.utc)
        kst=timezone(timedelta(hours=9))
        run_clock=dict(started_kst=started.astimezone(kst).isoformat(),completed_kst=ended.astimezone(kst).isoformat(),elapsed_hours=(ended-started).total_seconds()/3600)
    (OUT/'summary.json').write_text(json.dumps(dict(state=status,full=rows,pilot=pilot_rows,pilot_estimates=estimates,
        full_tiles_annotation_comparison=summaries,full_models_annotation_comparison=model_comparison,evaluation_scope=scope,run_clock=run_clock),ensure_ascii=False,indent=2),encoding='utf-8')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SeaDronesSee SSPNet 비교 평가</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1300px;margin:40px auto;padding:0 20px;color:#222}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #bbb;padding:8px;text-align:right}th{background:#eee}td:nth-child(-n+2){text-align:left}.percent{background:#fff4bf}img{max-width:100%}small{color:#555}</style>
<h1>SeaDronesSee OD v2 validation SSPNet 비교</h1>'''
    doc+=f'<p><strong>{status}</strong></p><p>공식 validation 원본 PNG 1,547장, swimmer 정답 6,206개. ESOD 없이 공개·추가 학습 모델을 비교한다.</p>'
    if scope:
        doc+='<p>'+html.escape(scope['reason'])+'</p><p>'+html.escape(scope['tiles_runtime_evidence'])+'</p>'
    if run_clock:
        doc+=f'<p>전체 이미지 두 조합 작업 시간: {run_clock["elapsed_hours"]:.2f}시간. 입력 해시 확인·모델 초기화·이미지 읽기·추론·저장·집계 포함. 시작 {run_clock["started_kst"]}, 완료 {run_clock["completed_kst"]}.</p>'
    doc+='<h2>validation 전체 결과</h2>'+(table(rows) if rows else '<p>아직 전체 평가 완료 조합이 없다.</p>')
    if model_comparison:
        doc+='<h2>두 모델의 정답별 대조 — 전체 이미지 추론</h2><table><tr><th>공통 정탐</th><th>공개만 정탐</th><th>추가 학습만 정탐</th><th>공통 미탐</th></tr><tr>'+''.join(f'<td>{v:,}</td>' for v in model_comparison.values())+'</tr></table><p><a href="full_models_annotation_comparison.csv">정답별 대조 CSV</a></p>'
    doc+='<h2>평가 조건과 해석</h2><p>신뢰도 ≥0.25, 매칭 IoU ≥0.3, 신뢰도 순 일대일 매칭. 타일 320×320·겹침 20%·병합 NMS IoU 0.5. 모델 입력은 긴 변 1280·짧은 변 1024 상한으로 비율 유지 리사이즈하며, 전체 추론도 원본 크기를 그대로 입력하는 것은 아니다. 저장 임계값 0.001, 원본 이미지당 최대 탐지 수 500. 다른 클래스와 배 위 인물 등 swimmer 정답 범위 밖의 사람 예측은 오탐에 포함될 수 있다.</p>'
    doc+='<p>이 결과는 IoU 0.3에서의 재현율·정밀도이며 공식 COCO mAP가 아니다. 시간은 이미지 읽기를 제외한 전처리·전송·추론·병합과 GPU 동기화를 포함한다. 워밍업은 제외하고, 재시작 시 재사용한 이미지의 최초 측정 시간도 합산한다. PNG 읽기 포함 시간은 각 metrics JSON의 total_measured_wall_seconds로 별도 기록한다.</p>'
    doc+='<p>원본 주석은 변경하지 않았다. 일부 수중 신체의 박스 범위가 애매한 사례가 있으며, 기존 학습 자료와 동일 픽셀 중복은 0건이지만 크롭·축소 중복과 TinyPerson 사전학습 원본과의 중복은 배제하지 못했다.</p>'
    if summaries:
        doc+='<h2>전체·분할 정답별 대조</h2><table><tr><th>모델</th><th>공통 정탐</th><th>분할 신규 정탐</th><th>분할에서 놓침</th><th>공통 미탐</th></tr>'
        for label,s in summaries.items(): doc+='<tr><td>'+LABELS[label]+'</td>'+''.join(f'<td>{v:,}</td>' for v in s.values())+'</tr>'
        doc+='</table>'
    doc+='<h2>원본 좌표의 크기별 재현율</h2><p>s=√(가로×세로), 단위 픽셀. 전체 정답과 먼저 매칭한 결과를 크기별로 집계한다.</p>'+''.join(details)
    doc+='<h2>4장 소량 점검 결과</h2><p>사전 선정 ID: 3464, 863, 1018, 48. 해상도·방향·음성 이미지를 점검하기 위한 표본으로, validation 전체 성능을 대표하지 않는다.</p>'+table(pilot_rows)
    gate=OUT/'pilot_full_approval.json'
    if gate.exists(): doc+='<pre>'+html.escape(json.dumps(read(gate),ensure_ascii=False,indent=2))+'</pre>'
    doc+='<h2>소량 점검 중첩 그림</h2><p>초록: swimmer 정답, 빨강: 신뢰도 0.25 이상 예측. 전체 이미지의 축소 표시이며 예측 좌표는 원본 기준으로 저장한다.</p>'
    for label in ('public','finetuned'):
        for mode in ('full','tiles'):
            for ident in (3464,863,1018,48):
                image=OUT/label/mode/f'pilot_{ident}.jpg'
                if image.exists(): doc+=f'<details><summary>{LABELS[label]} / {LABELS[mode]} / {ident}</summary><img src="{label}/{mode}/{image.name}"></details>'
    doc+='<p><a href="comparison.csv">전체 결과 CSV</a> · <a href="summary.json">요약 JSON</a> · <a href="protocol.json">실행 조건</a> · <a href="../seadronessee_validation_audit/review.html">원본 주석 검토</a></p></html>'
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    print(status)
    print(json.dumps(estimates,ensure_ascii=False,indent=2))

def size_table(m):
    content='<table><tr><th>크기 구간</th><th>정답</th><th>정탐</th><th>재현율</th></tr>'
    for name,v in m['geometric_size_all_ranges'].items(): content+=f'<tr><td>{html.escape(name)}</td><td>{v["positives"]}</td><td>{v["hits"]}</td><td class="percent">{percent(v["recall"])}</td></tr>'
    return content+'</table>'

def table(rows):
    if not rows: return '<p>계산 전</p>'
    content='<table><tr>'+''.join(f'<th>{s}</th>' for s in ['모델','방식','이미지','정탐','오탐','미탐','재현율','정밀도','오탐/장','추론 초/장','PNG 포함 초/장','PNG 포함 합계(시간)','최대 VRAM(MiB)'])+'</tr>'
    for r in rows:
        cells=[r['model'],r['mode'],str(r['images']),f"{r['TP']:,}",f"{r['FP']:,}",f"{r['FN']:,}",percent(r['recall']),percent(r['precision']),f"{r['FP_per_image']:.2f}",f"{r['mean_seconds']:.2f}",f"{r['mean_wall_seconds']:.2f}",f"{r['measured_wall_hours']:.2f}",f"{r['peak_vram_mib']:.0f}"]
        content+='<tr>'+''.join(f'<td class="{"percent" if i in (6,7) else ""}">{html.escape(v)}</td>' for i,v in enumerate(cells))+'</tr>'
    return content+'</table>'

if __name__=='__main__': main()
