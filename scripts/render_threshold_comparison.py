"""기존 두 모델의 전체 이미지 추론 결과를 같은 영역에서 비교한다."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[1]
output=root/'analysis_results/threshold_comparison'
output.mkdir(parents=True,exist_ok=True)
gt=json.loads((root/'analysis_results/seadronessee_validation/ground_truth_swimmer.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',25)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
for ident in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else (5146,1018,3464)):
    im=next(x for x in gt['images'] if x['id']==ident)
    anns=[a for a in gt['annotations'] if a['image_id']==ident]
    left=max(0,int(min(a['bbox'][0] for a in anns)-100)); top=max(0,int(min(a['bbox'][1] for a in anns)-100))
    right=min(im['width'],int(max(a['bbox'][0]+a['bbox'][2] for a in anns)+100)); bottom=min(im['height'],int(max(a['bbox'][1]+a['bbox'][3] for a in anns)+100))
    with Image.open(root/'data/seadronessee_odv2/images/val'/im['file_name']) as image: source=image.convert('RGB')
    original=source.copy(); p=ImageDraw.Draw(original); p.rectangle((left,top,right,bottom),outline='white',width=max(5,round(max(original.size)/700)))
    original.thumbnail((1320,740),Image.Resampling.LANCZOS)
    crop=source.crop((left,top,right,bottom)); scale=min(640/crop.width,550/crop.height)
    size=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))
    crop=crop.resize(size,Image.Resampling.LANCZOS)
    ssp=json.loads((root/f'analysis_results/seadronessee_validation/public/full/cache/{ident}.json').read_text(encoding='utf-8'))['boxes']
    yolo=json.loads((root/f'analysis_results/yolo26_seadronessee_1280_gpu_005/cache/{ident}.json').read_text(encoding='utf-8'))['boxes']
    def render(boxes, threshold):
        panel=crop.copy(); draw=ImageDraw.Draw(panel)
        for a in anns:
            x,y,w,h=a['bbox']; draw.rectangle(((x-left)*scale,(y-top)*scale,(x+w-left)*scale,(y+h-top)*scale),outline='#00ef35',width=3)
        for x1,y1,x2,y2,score,*_ in boxes:
            if score<threshold or x2<=left or x1>=right or y2<=top or y1>=bottom: continue
            x1=max(0,(x1-left)*scale); y1=max(0,(y1-top)*scale)
            x2=min(size[0]-1,(x2-left)*scale); y2=min(size[1]-1,(y2-top)*scale)
            col='#ff3030' if score>=.25 else '#ffd400'
            draw.rectangle((x1,y1,x2,y2),outline=col,width=3)
            label=f'{score:.2f}'; pos=(x1,max(0,y1-18))
            bb=draw.textbbox(pos,label,font=small); draw.rectangle(bb,fill='black'); draw.text(pos,label,font=small,fill=col)
        return panel
    gap=20; pw=max(640,size[0]); ph=size[1]+44
    width=max(original.width,2*pw+gap)+80; height=original.height+2*ph+215
    page=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(page)
    d.text((40,12),f'전체 원본 {ident}  |  {im["width"]} × {im["height"]} px',font=font,fill='black')
    page.paste(original,((width-original.width)//2,50))
    y=original.height+65
    d.text((40,y),'흰 영역 확대  |  초록: 정답   빨강: 0.25 이상   노랑: 0.05 이상 0.25 미만',font=font,fill='black')
    y+=45
    for i,(name,boxes,threshold) in enumerate((('SSPNet · 0.25',ssp,.25),('SSPNet · 0.05',ssp,.05),('YOLO26m · 0.25',yolo,.25),('YOLO26m · 0.05',yolo,.05))):
        x=40+(i%2)*(pw+gap); yy=y+(i//2)*ph
        d.text((x,yy),name,font=font,fill='black'); page.paste(render(boxes,threshold),(x,yy+40))
    dims=', '.join(f'{round(a["bbox"][2])}×{round(a["bbox"][3])}' for a in anns)
    d.text((40,height-42),f'원본 정답 박스 크기 (가로×세로 px): {dims}',font=small,fill='black')
    page.save(output/f'comparison_{ident}.jpg',quality=94)
    print(ident,size)
