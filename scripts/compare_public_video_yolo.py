import json, os, sys, time
from pathlib import Path
os.environ['YOLO_CONFIG_DIR']=str(Path('analysis_results/public_video_comparison/yolo_config').resolve())
import cv2, onnxruntime as ort
session=ort.InferenceSession
def gpu_session(path, sess_options=None, providers=None, **kwargs):
    opts=ort.SessionOptions(); opts.enable_mem_pattern=False; opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    result=session(path,opts,providers=[('DmlExecutionProvider',{'device_id':1}),'CPUExecutionProvider'])
    assert result.get_providers()[0]=='DmlExecutionProvider'
    return result
ort.InferenceSession=gpu_session
from ultralytics import YOLO
model=YOLO('analysis_results/yolo26_examples/yolo26m.onnx',task='detect')
out=Path('analysis_results/public_video_comparison'); out.mkdir(parents=True,exist_ok=True)
for name in ('pexels_19953301','pexels_37277130'):
    cap=cv2.VideoCapture('data/videos/'+name+'.mp4')
    if not cap.isOpened(): raise RuntimeError(name)
    count=0; started=time.monotonic()
    with (out/(name+'_yolo.jsonl')).open('w') as f:
        while True:
            ok,frame=cap.read()
            if not ok: break
            result=model.predict(frame,imgsz=1280,conf=.25,classes=[0],device='cpu',max_det=500,augment=False,verbose=False)[0]
            f.write(json.dumps({'frame':count,'boxes':result.boxes.data[:,:5].tolist()},allow_nan=False)+'\n')
            count+=1
            if count%100==0: f.flush(); print(name,count,'elapsed',round(time.monotonic()-started),flush=True)
    cap.release(); print(name,'done',count,flush=True)
