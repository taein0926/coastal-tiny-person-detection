import argparse
import json
import os
from pathlib import Path
import torch
from workflow import prepare_dataset, digest
from metrics import load_dataset, image_path, evaluate
from engine import make_model, predict, coco_predictions, draw

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=('full','tiles'),default='full')
parser.add_argument('--limit',type=int,default=10)
args=parser.parse_args()
if args.limit<0: raise ValueError('이미지 수는 0 이상이어야 합니다.')
base=Path('/workspace/output')
root=base/('evaluation_'+args.mode+('_all' if args.limit==0 else '_'+str(args.limit)))
root.mkdir(parents=True,exist_ok=True)
work=base/'work'
work.mkdir(exist_ok=True)
state=prepare_dataset(dict(task='dataset',dataset_name='AnnotationPiriapolis_v4',split='all'),work,'/workspace/dataset.zip')
annotations=state['annotations']
settings=dict(repo='/workspace/SSPNet',checkpoint='/workspace/epoch_5.pth',mode=args.mode,long_edge=1280,short_edge=1024,
              tile_size=320,tile_overlap=.2,merge_iou=.5,report_score=.25,save_min_score=.001,max_detections=500,match_iou=.3)
print('GPU:',torch.cuda.get_device_name(0),'가중치 SHA256:',digest(settings['checkpoint']),flush=True)
gt_all,summary=load_dataset(annotations,['person water','person sand'],limit=args.limit)
gt_water,_=load_dataset(annotations,['person water'],limit=args.limit)
identity=dict(settings,checkpoint_sha256=digest(settings['checkpoint']),dataset_sha256=state['zip_sha256'])
identity_file=root/'identity.json'
if identity_file.exists() and json.loads(identity_file.read_text(encoding='utf-8'))!=identity:
    raise RuntimeError('기존 결과의 가중치나 설정이 다릅니다. 결과 폴더를 확인하세요.')
identity_file.write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
cache=root/'cache'
cache.mkdir(exist_ok=True)
model,cfg=make_model(settings)
records={}
for im in gt_all['images']:
    path=image_path(annotations,im['file_name'])
    cache_file=cache/(str(im['id'])+'.json')
    image_sha=digest(path)
    try: rec=json.loads(cache_file.read_text(encoding='utf-8'))
    except (OSError,ValueError): rec=None
    if not rec or rec.get('image_sha256')!=image_sha:
        rec=predict(model,path,settings)
        rec['image_sha256']=image_sha
        tmp=cache_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(rec,ensure_ascii=False),encoding='utf-8')
        os.replace(str(tmp),str(cache_file))
    records[im['id']]=rec
    if len(records)<=3:draw(path,rec,root/('preview_%s.jpg'%im['id']),.25)
    print('완료',len(records),'/',len(gt_all['images']),'VRAM MiB:',round(rec['peak_vram_mib']),'시간:',round(rec['seconds'],2),flush=True)
pred=coco_predictions(records)
result=dict(gpu=torch.cuda.get_device_name(0),checkpoint_sha256=digest(settings['checkpoint']),settings=settings,
            image_ids=[im['id'] for im in gt_all['images']],image_names=[im['file_name'] for im in gt_all['images']],
            all_people=evaluate(gt_all,pred,.25,.3),water_only=evaluate(gt_water,pred,.25,.3),
            mean_seconds=sum(x['seconds'] for x in records.values())/len(records),
            peak_vram_mib=max(x['peak_vram_mib'] for x in records.values()))
(root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'predictions.json').write_text(json.dumps(pred,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
