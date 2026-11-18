"""원본 OD v2 validation의 네 조합 평가. 기존 데이터·결과를 수정하지 않는다."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import torch
from engine import make_model, predict, positions, coco_predictions, dump, sha256
from metrics import evaluate, match_predictions, geometric_size

DATA = Path('/workspace/data')
OUT = Path('/workspace/output')
PILOT = [3464, 863, 1018, 48]
CHECKPOINTS = {
    'public': ('/workspace/weights/faster_rcnn_sspnet.pth', '912cc5b6f8a45b0f4521c9bf325868ed10dd4757497fb6ba8843fdbd26b09e4a'),
    'finetuned': ('/workspace/epoch_5.pth', '690f943c0a01fe40954b823942a65f64df94bb2157766156fa731d5a89f249a1'),
}

def load():
    source = DATA/'annotations/instances_val.json'
    assert sha256(source) == '9d64e077cc8c5d8c53e8f106ec292123300a6d93fed7f1c5c1464b600f9ce6a9'
    raw = json.loads(source.read_text())
    assert next(c['id'] for c in raw['categories'] if c['name']=='swimmer') == 1
    assert not any(a['category_id']==0 for a in raw['annotations']), 'ignore 처리 계약을 다시 확인해야 합니다.'
    images=sorted(raw['images'],key=lambda im: im['id'])
    manifest={r['name']:r for r in json.loads((DATA/'manifest.json').read_text())}
    assert len(images)==len(manifest)==1547
    for i,im in enumerate(images):
        path=DATA/'images/val'/im['file_name']
        assert path.parent==DATA/'images/val'
        assert sha256(path)==manifest[im['file_name']]['sha256'], im['file_name']
        if (i+1)%300==0: print('입력 해시 확인',i+1,flush=True)
    anns=[dict(a,iscrowd=0,ignore=False,segmentation=[]) for a in raw['annotations'] if a['category_id']==1]
    assert len(anns)==6206
    gt=dict(info={'description':'SeaDronesSee OD v2 validation swimmer'},licenses=[],images=images,
            annotations=anns,categories=[dict(id=1,name='swimmer')])
    return gt,manifest

def subset(gt,ids):
    return dict(gt,images=[im for im in gt['images'] if im['id'] in ids],
                annotations=[a for a in gt['annotations'] if a['image_id'] in ids])

def settings_for(label,mode,manifest):
    checkpoint,expected=CHECKPOINTS[label]
    assert sha256(checkpoint)==expected
    return dict(repo='/workspace/SSPNet',checkpoint=checkpoint,checkpoint_sha256=expected,
                model_label=label,mode=mode,long_edge=1280,short_edge=1024,tile_size=320,
                tile_overlap=.2,merge_iou=.5,report_score=.25,save_min_score=.001,
                max_detections=500,match_iou=.3,dataset='SeaDronesSee OD v2 validation',
                target_class='swimmer',annotations_sha256=sha256(DATA/'annotations/instances_val.json'),
                manifest_sha256=sha256(DATA/'manifest.json'),engine_sha256=sha256('/workspace/helpers/engine.py'),
                metrics_sha256=sha256('/workspace/helpers/metrics.py'),runner_sha256=sha256(__file__),
                source_commit=os.environ['SSPNET_SOURCE_COMMIT'])

def validate_record(im,rec,mode):
    boxes=np.asarray(rec['boxes'],dtype=np.float64).reshape(-1,5)
    assert np.isfinite(boxes).all()
    if len(boxes):
        assert (boxes[:,0]>=0).all() and (boxes[:,1]>=0).all()
        assert (boxes[:,2]<=im['width']).all() and (boxes[:,3]<=im['height']).all()
        assert (boxes[:,2]>boxes[:,0]).all() and (boxes[:,3]>boxes[:,1]).all()
        assert ((boxes[:,4]>=0)&(boxes[:,4]<=1)).all()
    expected=1 if mode=='full' else len(positions(im['width'],320,.2))*len(positions(im['height'],320,.2))
    assert rec['patches']==expected,(im['id'],rec['patches'],expected)

def preview(im,rec,gt,path):
    with Image.open(DATA/'images/val'/im['file_name']) as original:
        view=original.convert('RGB'); view.thumbnail((1600,1200))
        sx,sy=view.width/im['width'],view.height/im['height']
        pen=ImageDraw.Draw(view)
        for a in gt['annotations']:
            if a['image_id']==im['id']:
                x,y,w,h=a['bbox']; pen.rectangle((x*sx,y*sy,(x+w)*sx,(y+h)*sy),outline='lime',width=2)
        for x,y,r,b,s in rec['boxes']:
            if s>=.25:
                pen.rectangle((x*sx,y*sy,r*sx,b*sy),outline='red',width=2)
                pen.text((x*sx,y*sy),'%.2f'%s,fill='red')
        view.save(path,quality=93)

def summarize(gt,records):
    predictions=coco_predictions(records)
    result=evaluate(gt,predictions,.25,.3)
    result['false_negatives']=result['positives']-result['true_positives']
    result['timing']=dict(total_inference_seconds=sum(r['seconds'] for r in records.values()),
        mean_inference_seconds=float(np.mean([r['seconds'] for r in records.values()])),
        median_inference_seconds=float(np.median([r['seconds'] for r in records.values()])),
        total_measured_wall_seconds=sum(r['wall_seconds'] for r in records.values()),
        maximum_vram_mib=max(r['peak_vram_mib'] for r in records.values()),
        total_patches=sum(r['patches'] for r in records.values()),
        note='wall_seconds는 PNG 읽기를 포함한다. seconds는 기존 engine 정의에 따라 PNG 읽기를 제외하고 전처리·전송·추론·NMS와 GPU 동기화를 포함한다. 워밍업은 제외한다.')
    matches=match_predictions(gt,predictions,.3)
    hits={a['id'] for p,a,ignored in matches if a is not None and not ignored and p['score']>=.25}
    result['geometric_size_all_ranges']={}
    for name,low,high in [('s<8',0,8),('8<=s<12',8,12),('12<=s<20',12,20),('20<=s<32',20,32),('32<=s<50',32,50),('50<=s',50,float('inf'))]:
        selected=[a for a in gt['annotations'] if low<=geometric_size(a['bbox'])<high]
        n=sum(a['id'] in hits for a in selected)
        result['geometric_size_all_ranges'][name]=dict(positives=len(selected),hits=n,recall=n/len(selected) if selected else None)
    return result,predictions

def run_combo(label,mode,gt,manifest,phase):
    folder=OUT/label/mode; folder.mkdir(parents=True,exist_ok=True)
    settings=settings_for(label,mode,manifest)
    identity=folder/'identity.json'
    if identity.exists(): assert json.loads(identity.read_text())==settings,'설정이 다른 기존 결과를 재사용할 수 없습니다.'
    else: dump(identity,settings)
    cache=folder/'cache'; cache.mkdir(exist_ok=True)
    chosen=gt['images'] if phase=='full' else [next(im for im in gt['images'] if im['id']==i) for i in PILOT]
    records={}; model=None
    started=time.perf_counter()
    dump(folder/'status.json',dict(state='실행 중',phase=phase,completed=0,total=len(chosen)))
    for index,im in enumerate(chosen):
        saved=cache/(str(im['id'])+'.json')
        rec=json.loads(saved.read_text()) if saved.exists() else None
        digest=manifest[im['file_name']]['sha256']
        if rec is None or rec.get('image_sha256')!=digest:
            if model is None:
                model,cfg=make_model(settings); cfg.dump(str(folder/'resolved_config.py'))
                warm_image=DATA/'images/val'/chosen[0]['file_name']
                warm=predict(model,warm_image,dict(settings,mode='full'))
                # 정사각 타일 경로도 한 번 예열한다. 타일 전체를 중복 실행하지 않는다.
                if mode=='tiles':
                    with Image.open(warm_image) as picture: picture.crop((0,0,320,320)).save(folder/'warmup.png')
                    warm_tile=predict(model,folder/'warmup.png',dict(settings,mode='full'))
                else: warm_tile=None
                dump(folder/('warmup_'+phase+'.json'),dict(full=warm,tile=warm_tile,gpu=torch.cuda.get_device_name(0)))
            tick=time.perf_counter()
            rec=predict(model,DATA/'images/val'/im['file_name'],settings)
            rec.update(wall_seconds=time.perf_counter()-tick,image_sha256=digest,image_id=im['id'],
                       width=im['width'],height=im['height'])
            validate_record(im,rec,mode)
            dump(saved,rec)
        validate_record(im,rec,mode)
        records[im['id']]=rec
        if phase=='pilot': preview(im,rec,gt,folder/('pilot_%s.jpg'%im['id']))
        dump(folder/'status.json',dict(state='실행 중',phase=phase,completed=index+1,total=len(chosen),
            last_image_id=im['id'],last_seconds=rec['seconds'],last_patches=rec['patches'],
            elapsed_seconds=time.perf_counter()-started))
        print('%s/%s %s %d/%d id=%d %.2fs %d패치 %.0fMiB'%(label,mode,phase,index+1,len(chosen),im['id'],rec['seconds'],rec['patches'],rec['peak_vram_mib']),flush=True)
    chosen_gt=subset(gt,set(records))
    result,predictions=summarize(chosen_gt,records)
    result.update(phase=phase,partial=phase!='full',image_ids=sorted(records),gpu=torch.cuda.get_device_name(0))
    dump(folder/('metrics_'+phase+'.json'),result)
    dump(folder/('predictions_'+phase+'.json'),predictions)
    dump(folder/'status.json',dict(state='완료',phase=phase,completed=len(chosen),total=len(chosen),elapsed_seconds=time.perf_counter()-started))
    del model; gc.collect(); torch.cuda.empty_cache()
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['pilot','full'],default='pilot')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    gt,manifest=load()
    dump(OUT/'ground_truth_swimmer.json',gt)
    dump(OUT/'protocol.json',dict(pilot_ids=PILOT,pilot_selection='사전 고정: 3840 가로·5456 가로·5456 세로·swimmer 없는 이미지. 정확도로 선택하지 않는다.',
        score_threshold=.25,matching_iou=.3,tile_size=320,tile_overlap=.2,merge_nms_iou=.5,
        long_edge=1280,short_edge=1024,max_detections=500,save_min_score=.001,
        total_images=1547,total_swimmers=6206,expected_patches_tiles=208934,
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    if args.phase=='full':
        gate=json.loads((OUT/'pilot_approval.json').read_text())
        assert gate.get('passed') and gate['pilot_ids']==PILOT,'소량 점검 통과 기록이 필요합니다.'
    results={}
    for mode in ['full','tiles']:
        for label in ['public','finetuned']:
            results[label+'_'+mode]=run_combo(label,mode,gt,manifest,args.phase)
    dump(OUT/('comparison_'+args.phase+'.json'),results)
    print('네 조합 평가 완료:',args.phase,flush=True)

if __name__=='__main__': main()
