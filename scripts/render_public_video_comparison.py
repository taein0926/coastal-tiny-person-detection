import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_results/public_video_comparison'
FF=ROOT/'analysis_results/training_2000_tools/packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v9.0.2.exe'
FONT='C:/Windows/Fonts/malgun.ttf'

def records(name,model):
    with (OUT/(name+'_'+model+'.jsonl')).open() as f:
        for i,line in enumerate(f):
            r=json.loads(line); assert r['frame']==i
            yield r['boxes']

def render(name):
    cap=cv2.VideoCapture(str(ROOT/'data/videos'/ (name+'.mp4')))
    fps=cap.get(cv2.CAP_PROP_FPS); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale=min(1,1280/max(width,height)); w=int(width*scale)//2*2; h=int(height*scale)//2*2
    path=OUT/(name+'_두_모델_비교.mp4')
    cmd=[str(FF),'-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w*2}x{h+48}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    a,b=records(name,'yolo'),records(name,'sspnet'); count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.resize(frame,(w,h)); views=[]
            for model,arr in [('YOLO26m',next(a)),('SSPNet',next(b))]:
                img=frame.copy()
                for x1,y1,x2,y2,score in arr:
                    if score<.25: continue
                    p1=(max(0,round(x1*w/width)),max(0,round(y1*h/height))); p2=(min(w-1,round(x2*w/width)),min(h-1,round(y2*h/height)))
                    cv2.rectangle(img,p1,p2,(40,255,40),2)
                view=np.zeros((h+48,w,3),np.uint8); view[48:]=img
                cv2.putText(view,model+'  |  score >= 0.25', (12,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
                views.append(view)
            proc.stdin.write(np.hstack(views).tobytes()); count+=1
            if count%200==0: print(name,'render',count,flush=True)
    finally:
        cap.release(); proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
    for gen in (a,b):
        try: next(gen); raise RuntimeError('prediction frame count exceeds video')
        except StopIteration: pass
    print(name,'영상 완료',count,flush=True)

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def example(name,region,gt,tag):
    cap=cv2.VideoCapture(str(ROOT/'data/videos'/(name+'.mp4'))); ok,frame=cap.read(); cap.release(); assert ok
    x,y,r,b=region; crop=frame[y:b,x:r]; panels=[]; scores={}
    for model in ('yolo','sspnet'):
        predictions=next(records(name,model)); selected=[p for p in predictions if p[4]>=.25 and x<=(p[0]+p[2])/2<r and y<=(p[1]+p[3])/2<b]
        selected.sort(key=lambda p:-p[4]); unmatched=set(range(len(gt))); matched=[]; false=[]
        for p in selected:
            candidates=[(iou(p,g),j) for j,g in enumerate(gt) if j in unmatched]
            v,j=max(candidates,default=(0,None))
            if v>=.3: matched.append(p); unmatched.remove(j)
            else: false.append(p)
        img=crop.copy()
        def box(q,color):
            cv2.rectangle(img,(int(q[0]-x),int(q[1]-y)),(int(q[2]-x),int(q[3]-y)),color,1)
        for g in gt: box(g,(0,0,255))
        for p in matched: box(p,(0,255,0))
        for p in false: box(p,(0,165,255))
        scores[model]=dict(gt=len(gt),tp=len(matched),fp=len(false),fn=len(unmatched))
        panels.append(img)
    target_h=600; neww=round(crop.shape[1]*target_h/crop.shape[0]); canvas=Image.new('RGB',(neww*2, target_h+115),'white'); draw=ImageDraw.Draw(canvas); font=ImageFont.truetype(FONT,20); small=ImageFont.truetype(FONT,17)
    for n,model in enumerate(('yolo','sspnet')):
        image=Image.fromarray(cv2.cvtColor(panels[n],cv2.COLOR_BGR2RGB)).resize((neww,target_h),Image.Resampling.NEAREST); canvas.paste(image,(neww*n,80)); s=scores[model]
        draw.text((neww*n+8,5),('YOLO26m' if n==0 else 'SSPNet')+f"  정탐 {s['tp']} / 미탐 {s['fn']} / 오탐 {s['fp']}",font=font,fill='black')
        draw.text((neww*n+8,35),'신뢰도 0.25 · IoU 0.3',font=small,fill='black')
    draw.text((8,target_h+85),'빨강: 정답   초록: 정답과 일치한 예측   주황: 정답과 일치하지 않은 예측',font=small,fill='black')
    canvas.save(OUT/(name+'_'+tag+'_주석_비교.png')); return scores

if __name__=='__main__':
    for name in ('pexels_19953301','pexels_37277130'): render(name)
    items=json.loads((ROOT/'analysis_results/recall_v3_review/video_gt_candidates.json').read_text(encoding='utf-8'))
    item=next(q for q in items if q['video_key']=='pexels_19953301/0'); x,y,r,b=item['source_region']
    gt=[[x+a['bbox'][0],y+a['bbox'][1],x+a['bbox'][0]+a['bbox'][2],y+a['bbox'][1]+a['bbox'][3]] for a in item['annotations']]
    first=example('pexels_19953301',[x,y,r,b],gt,'기존_검토영역')
    second=example('pexels_37277130',[1125,520,1205,555],[[1135,540,1145,549],[1148,526,1161,536],[1184,527,1192,536]],'작은_사람_직접_주석영역')
    (OUT/'sample_scores.json').write_text(json.dumps({'pexels_19953301':first,'pexels_37277130':second,'note':'각 영상 첫 프레임의 지정 영역에 한정한 예시. 두 번째 주석은 직접 작성한 잠정 주석이며 독립 검수 전.'},ensure_ascii=False,indent=2),encoding='utf-8')
