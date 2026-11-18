"""작은 수영자 재현율과 오탐 예산으로 검증 모델을 선택한다. GPU 의존성 없음."""
import hashlib
import json
import math
from pathlib import Path
from metrics import match_predictions


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_bundle(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'quality_manifest.json').read_text(encoding='utf-8'))
    groups={};hashes={};data={}
    for split in ('train','val'):
        entry=manifest['splits'][split]
        ann=(root/entry['annotation_file']).resolve()
        if root not in ann.parents or digest(ann)!=entry['annotation_sha256']:
            raise ValueError('승인 후 주석이 변경됐거나 데이터 경로가 올바르지 않습니다.')
        raw=json.loads(ann.read_text(encoding='utf-8'))
        images={im['id']:im for im in raw['images']}
        if len(images)!=len(raw['images']) or {r['image_id'] for r in entry['images']}!=set(images):
            raise ValueError('검토 목록과 이미지 목록이 다릅니다.')
        hashes[split]=set();groups[split]=set()
        for item in entry['images']:
            if item.get('review',{}).get('status')!='accepted':raise ValueError('주석 검토 승인 없는 이미지입니다.')
            path=(root/item['file_name']).resolve()
            if root not in path.parents or item['file_name']!=images[item['image_id']]['file_name']:
                raise ValueError('이미지 경로가 데이터 범위를 벗어납니다.')
            if digest(path)!=item['sha256']:raise ValueError('승인 후 이미지가 변경됐습니다.')
            if item['group']!=images[item['image_id']]['capture_group']:raise ValueError('촬영 그룹이 일치하지 않습니다.')
            groups[split].add(item['group']);hashes[split].add(item['sha256'])
        if groups[split]!=set(entry['groups']):raise ValueError('촬영 그룹 목록이 다릅니다.')
        ids=set()
        for a in raw['annotations']:
            if a['id'] in ids or a['image_id'] not in images:raise ValueError('중복 또는 연결되지 않은 주석입니다.')
            ids.add(a['id'])
            im=images[a['image_id']];x,y,w,h=a['bbox']
            if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<1 or min(x,y)<0 or x+w>im['width'] or y+h>im['height']:
                raise ValueError('유효하지 않은 사람 박스입니다.')
            water_valid=isinstance(a.get('water'),bool) or (manifest.get('version')==3 and 'water' in a and a['water'] is None)
            if a['category_id']!=1 or not water_valid:raise ValueError('사람 클래스와 물속 여부 주석을 확인하세요.')
        data[split]=raw
    if groups['train']&groups['val'] or hashes['train']&hashes['val']:
        raise ValueError('학습과 검증에 같은 촬영 또는 이미지가 있습니다.')
    if (groups['train']|groups['val'])&set(manifest['reserved_groups']):
        raise ValueError('보존 평가 촬영이 학습·검증에 포함됐습니다.')
    return manifest,data


def select_operating_point(gt,predictions,fp_per_image=4.,iou=.3,small_low=2.,small_high=20.,
                           thresholds=(.001,.005,.01,.025,.05,.075,.1,.15,.2,.25,.35,.5,.75)):
    if fp_per_image<0 or not 0<iou<=1 or not 0<=small_low<small_high:raise ValueError('검증 기준이 올바르지 않습니다.')
    if not gt['images']:raise ValueError('검증 이미지가 없습니다.')
    for prediction in predictions:
        if not math.isfinite(prediction['score']) or not 0<=prediction['score']<=1 or not all(math.isfinite(v) for v in prediction['bbox']):
            raise ValueError('유효하지 않은 검증 예측입니다.')
    normal=[a for a in gt['annotations'] if not a.get('ignore') and not a.get('iscrowd')]
    small={a['id'] for a in normal if a.get('water') and small_low<=math.sqrt(a['bbox'][2]*a['bbox'][3])<small_high}
    if not small:raise ValueError('검증 데이터에 작은 수영자 정답이 없습니다.')
    matches=match_predictions(gt,predictions,iou)
    points=[]
    for threshold in tuple(thresholds)+(1.000001,):
        selected=[(p,a) for p,a,ignored in matches if not ignored and p['score']>=threshold]
        hit={a['id'] for p,a in selected if a is not None}
        fp=sum(a is None for p,a in selected)
        points.append(dict(threshold=threshold,small_swimmer_positives=len(small),small_swimmer_hits=len(small&hit),
                           small_swimmer_recall=len(small&hit)/len(small),person_recall=len(hit)/len(normal) if normal else 0.,
                           TP=len(hit),FP=fp,FN=len(normal)-len(hit),FP_per_image=fp/len(gt['images'])))
    feasible=[p for p in points if p['FP_per_image']<=fp_per_image]
    best=max(feasible,key=selection_key)
    return dict(best=best,points=points,fp_budget=fp_per_image,match_iou=iou,
                small_definition=dict(space='원본 이미지',low_inclusive=small_low,high_exclusive=small_high,measure='sqrt(width*height)'))


def selection_key(point):
    return point['small_swimmer_recall'],point['person_recall'],-point['FP_per_image']
