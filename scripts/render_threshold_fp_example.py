import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[1]
gt=json.loads((root/'analysis_results/seadronessee_validation/ground_truth_swimmer.json').read_text(encoding='utf-8'))
ident=1157; region=(914,2202,1614,2902); l,t,r,b=region
im=next(i for i in gt['images'] if i['id']==ident)
anns=[a['bbox'] for a in gt['annotations'] if a['image_id']==ident and not a.get('ignore') and not a.get('iscrowd')]
original=json.loads((root/'data/seadronessee_odv2/annotations/instances_val.json').read_text(encoding='utf-8'))
other=[a['bbox'] for a in original['annotations'] if a['image_id']==ident and a['category_id'] in (2,3,4,5)]
preds=json.loads((root/f'analysis_results/seadronessee_validation/public/full/cache/{ident}.json').read_text())['boxes']
def iou(p,a):
    x,y,z,w=p[:4]; u,v,aw,ah=a
    inter=max(0,min(z,u+aw)-max(x,u))*max(0,min(w,v+ah)-max(y,v))
    return inter/max(1e-10,(z-x)*(w-y)+aw*ah-inter)
def center_in(p): return l<=(p[0]+p[2])/2<r and t<=(p[1]+p[3])/2<b
def score(threshold):
    matched=set(); result=[]
    for p in sorted((p for p in preds if p[4]>=threshold),key=lambda p:-p[4]):
        val,j=max([(iou(p,a),i) for i,a in enumerate(anns) if i not in matched],default=(0,None))
        hit=val>=.3
        if hit: matched.add(j)
        if center_in(p): result.append((p,hit))
    inside=[(i,a) for i,a in enumerate(anns) if l<=a[0]+a[2]/2<r and t<=a[1]+a[3]/2<b]
    return result,[a for i,a in inside if i not in matched],len(inside)

src=Image.open(root/'data/seadronessee_odv2/images/val'/im['file_name']).convert('RGB')
whole=src.copy(); ImageDraw.Draw(whole).rectangle(region,outline='#ffdb00',width=12); whole.thumbnail((1250,650),Image.Resampling.LANCZOS)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',25); label=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21); small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
crop=src.crop(region).resize((650,650),Image.Resampling.LANCZOS); ratio=650/700
page=Image.new('RGB',(1400,650+2*720+170),'white'); draw=ImageDraw.Draw(page)
draw.text((30,12),f'전체 원본 {ident}  |  {im["width"]} × {im["height"]} px',font=font,fill='black')
page.paste(whole,((1400-whole.width)//2,50)); y=whole.height+65
draw.text((30,y),'노랑: 비교 영역   초록: 정답   파랑: 정탐 예측   빨강: 오탐 예측',font=label,fill='black'); y+=40
draw.text((30,y),'같은 이미지 · SSPNet 공개 모델 · 물결 주변의 예측 두 곳을 선정',font=small,fill='black'); y+=30
for i,threshold in enumerate((.20,.15,.10,.05)):
    boxes,missed,n=score(threshold); tp=sum(hit for _,hit in boxes)
    def background(p):
        cx=(p[0]+p[2])/2; cy=(p[1]+p[3])/2
        return any(abs(p[0]-x)<1 and abs(p[1]-y)<1 for x,y in ((1307.1,2762.2),(1288.3,2593.2)))
    fp=sum(not hit and background(p) for p,hit in boxes)
    x=30+(i%2)*690; yy=y+(i//2)*720
    draw.text((x,yy),f'신뢰도 {threshold:.2f} 이상   오탐 예측 {fp}개',font=label,fill='black')
    panel=crop.copy(); d=ImageDraw.Draw(panel)
    for p,hit in boxes:
        if hit or background(p):
            x1,y1,x2,y2=p[:4]; d.rectangle(((x1-l)*ratio,(y1-t)*ratio,(x2-l)*ratio,(y2-t)*ratio),outline='#00bfff' if hit else '#ff2929',width=3)
    for a in anns:
        ax,ay,aw,ah=a
        if l<=ax+aw/2<r and t<=ay+ah/2<b:
            d.rectangle(((ax-l)*ratio,(ay-t)*ratio,(ax+aw-l)*ratio,(ay+ah-t)*ratio),outline='#00e929',width=2)
    page.paste(panel,(x,yy+38)); print(threshold, 'gt',n,'tp',tp,'fp',fp,'fn',len(missed))
out=root/'analysis_results/sspnet_threshold_015_020/sspnet_1157_신뢰도별_배경_예측.jpg'
page.save(out,quality=95); print(out)
