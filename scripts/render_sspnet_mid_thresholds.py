import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[1]
out=root/'analysis_results/sspnet_threshold_015_020'
out.mkdir(parents=True,exist_ok=True)
gt=json.loads((root/'analysis_results/seadronessee_validation/ground_truth_swimmer.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',25)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
for ident in (5146,1018,3464,526,622,3001,1073,3716,393):
    im=next(x for x in gt['images'] if x['id']==ident)
    anns=[a for a in gt['annotations'] if a['image_id']==ident]
    left=max(0,int(min(a['bbox'][0] for a in anns)-100)); top=max(0,int(min(a['bbox'][1] for a in anns)-100))
    right=min(im['width'],int(max(a['bbox'][0]+a['bbox'][2] for a in anns)+100)); bottom=min(im['height'],int(max(a['bbox'][1]+a['bbox'][3] for a in anns)+100))
    with Image.open(root/'data/seadronessee_odv2/images/val'/im['file_name']) as image: source=image.convert('RGB')
    whole=source.copy(); d=ImageDraw.Draw(whole)
    d.rectangle((left,top,right,bottom),outline='white',width=max(5,round(max(whole.size)/700)))
    whole.thumbnail((1320,740),Image.Resampling.LANCZOS)
    crop=source.crop((left,top,right,bottom)); scale=min(640/crop.width,550/crop.height)
    size=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))
    crop=crop.resize(size,Image.Resampling.LANCZOS)
    boxes=json.loads((root/f'analysis_results/seadronessee_validation/public/full/cache/{ident}.json').read_text(encoding='utf-8'))['boxes']
    def render(threshold):
        panel=crop.copy(); draw=ImageDraw.Draw(panel)
        for a in anns:
            x,y,w,h=a['bbox']; draw.rectangle(((x-left)*scale,(y-top)*scale,(x+w-left)*scale,(y+h-top)*scale),outline='#00ef35',width=3)
        visible=[b for b in boxes if b[4]>=threshold and b[2]>left and b[0]<right and b[3]>top and b[1]<bottom]
        for x1,y1,x2,y2,score in sorted(visible,key=lambda b:b[4]):
            x1=max(0,(x1-left)*scale); y1=max(0,(y1-top)*scale)
            x2=min(size[0]-1,(x2-left)*scale); y2=min(size[1]-1,(y2-top)*scale)
            col='#ff3030' if score>=.20 else '#ffd400'
            draw.rectangle((x1,y1,x2,y2),outline=col,width=3)
            label=f'{score:.2f}'; pos=(min(x1,max(0,size[0]-40)),max(0,y1-18))
            bb=draw.textbbox(pos,label,font=small); draw.rectangle(bb,fill='black'); draw.text(pos,label,font=small,fill=col)
        return panel
    width=max(whole.width,2*max(640,size[0])+20)+80; height=whole.height+size[1]+195
    page=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(page)
    d.text((40,12),f'전체 원본 {ident}  |  {im["width"]} × {im["height"]} px',font=font,fill='black')
    page.paste(whole,((width-whole.width)//2,50))
    y=whole.height+65
    d.text((40,y),'흰 영역 확대  |  초록: 정답   빨강: 0.20 이상   노랑: 0.15 이상 0.20 미만',font=font,fill='black')
    for i,th in enumerate((.20,.15)):
        x=40+i*(max(640,size[0])+20)
        d.text((x,y+43),f'SSPNet · {th:.2f}',font=font,fill='black'); page.paste(render(th),(x,y+82))
    dims=', '.join(f'{round(a["bbox"][2])}×{round(a["bbox"][3])}' for a in anns)
    d.text((40,height-40),f'원본 정답 박스 크기 (가로×세로 px): {dims}',font=small,fill='black')
    page.save(out/f'comparison_{ident}.jpg',quality=94)
    print(ident, sum(b[4]>=.20 for b in boxes),sum(b[4]>=.15 for b in boxes))
