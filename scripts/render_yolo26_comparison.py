"""완료된 동일 조건 예측으로 문서용 이미지 생성."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[1]
out=root/'analysis_results/yolo26_seadronessee_1280_gpu'
gt=json.loads((root/'analysis_results/seadronessee_validation/ground_truth_swimmer.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 26)
for ident in (5146,1018,3464):
    im=next(x for x in gt['images'] if x['id']==ident)
    anns=[a for a in gt['annotations'] if a['image_id']==ident]
    rec=json.loads((out/'cache'/f'{ident}.json').read_text(encoding='utf-8'))
    with Image.open(root/'data/seadronessee_odv2/images/val'/im['file_name']) as s:
        img=s.convert('RGB')
    draw=ImageDraw.Draw(img)
    for a in anns:
        x,y,w,h=a['bbox']; draw.rectangle((x,y,x+w,y+h),outline='lime',width=5)
    for x1,y1,x2,y2,score,cls in rec['boxes']:
        draw.rectangle((x1,y1,x2,y2),outline='red',width=4)
    img.save(out/f'sea_{ident}_yolo_full.jpg',quality=95)
    left=max(0,int(min(a['bbox'][0] for a in anns)-100)); top=max(0,int(min(a['bbox'][1] for a in anns)-100))
    right=min(im['width'],int(max(a['bbox'][0]+a['bbox'][2] for a in anns)+100)); bottom=min(im['height'],int(max(a['bbox'][1]+a['bbox'][3] for a in anns)+100))
    yolo=img.crop((left,top,right,bottom))
    ssp=Image.open(root/f'analysis_results/annotated_examples/images/sea_{ident}_public_zoom.png').convert('RGB')
    assert ssp.size==yolo.size
    maxw=850
    if yolo.width>maxw:
        size=(maxw,round(yolo.height*maxw/yolo.width))
        yolo=yolo.resize(size); ssp=ssp.resize(size)
    panel=Image.new('RGB',(ssp.width+yolo.width+24,ssp.height+65),'white')
    panel.paste(ssp,(0,55)); panel.paste(yolo,(ssp.width+24,55))
    pen=ImageDraw.Draw(panel)
    pen.text((8,8),'SSPNet 공개 모델',font=font,fill='black')
    pen.text((ssp.width+32,8),'YOLO26m 공개 모델',font=font,fill='black')
    panel.save(out/f'compare_{ident}.png')
    with Image.open(root/'data/seadronessee_odv2/images/val'/im['file_name']) as s:
        original=s.convert('RGB')
    whole=original.copy()
    wd=ImageDraw.Draw(whole)
    line=max(5,round(max(whole.size)/700))
    wd.rectangle((left,top,right,bottom),outline='white',width=line)
    whole.thumbnail((1600,900), Image.Resampling.LANCZOS)
    width=max(whole.width,panel.width,1100)
    page=Image.new('RGB',(width+80,whole.height+panel.height+195),'white')
    p=ImageDraw.Draw(page)
    p.text((40,15),f'전체 원본  |  {im["width"]} × {im["height"]} px  |  이미지 {ident}',font=font,fill='black')
    page.paste(whole,((page.width-whole.width)//2,55))
    y=whole.height+75
    p.text((40,y),'흰 상자 영역 확대  |  초록: 정답  빨강: 예측',font=font,fill='black')
    page.paste(panel,((page.width-panel.width)//2,y+45))
    sizes=', '.join(f'{round(a["bbox"][2])}×{round(a["bbox"][3])}' for a in anns)
    p.text((40,page.height-48),f'원본 정답 박스 크기 (가로×세로 px): {sizes}',font=font,fill='black')
    page.save(out/f'compare_{ident}_full_and_zoom.jpg',quality=94)
    print(ident,len(anns),len(rec['boxes']))
