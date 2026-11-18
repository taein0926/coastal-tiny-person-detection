import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from mmdet.apis import inference_detector
from engine import make_model

torch.set_num_threads(2)
model, _ = make_model(dict(repo='/workspace/SSPNet', checkpoint='/workspace/weights/faster_rcnn_sspnet.pth', mode='full', long_edge=1280, short_edge=1024, save_min_score=.25, max_detections=500))
for name in ('pexels_19953301', 'pexels_37277130'):
    cap = cv2.VideoCapture('/workspace/videos/' + name + '.mp4')
    if not cap.isOpened(): raise RuntimeError(name)
    out = Path('/workspace/output') / (name + '_sspnet.jsonl')
    start=time.monotonic(); count=0
    with out.open('w') as f:
        while True:
            ok, frame=cap.read()
            if not ok: break
            with torch.no_grad(): result=inference_detector(model, frame)
            boxes=np.asarray(result[0], dtype=np.float32).reshape(-1,5)
            f.write(json.dumps({'frame':count,'boxes':boxes.tolist()},allow_nan=False)+'\n')
            count+=1
            if count%100==0: f.flush(); print(name, count, 'elapsed',round(time.monotonic()-start),flush=True)
    cap.release()
    print(name, '완료',count,flush=True)
