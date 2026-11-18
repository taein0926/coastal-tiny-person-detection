import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root=Path(__file__).resolve().parents[1]; ident=int(sys.argv[1]) if len(sys.argv)>1 else 3001
out=root/'analysis_results/sspnet_threshold_015_020'; out.mkdir(parents=True,exist_ok=True)
gt=json.loads((root/'analysis_results/seadronessee_validation/ground_truth_swimmer.json').read_text(encoding='utf-8'))
im=next(x for x in gt['images'] if x['id']==ident); anns=[a for a in gt['annotations'] if a['image_id']==ident]
left=max(0,int(min(a['bbox'][0] for a in anns)-100)); top=max(0,int(min(a['bbox'][1] for a in anns)-100))
right=min(im['width'],int(max(a['bbox'][0]+a['bbox'][2] for a in anns)+100)); bottom=min(im['height'],int(max(a['bbox'][1]+a['bbox'][3] for a in anns)+100))
with Image.open(root/'data/seadronessee_odv2/images/val'/im['file_name']) as f: src=f.convert('RGB')
whole=src.copy(); ImageDraw.Draw(whole).rectangle((left,top,right,bottom),outline='white',width=8); whole.thumbnail((1400,780),Image.Resampling.LANCZOS)
crop=src.crop((left,top,right,bottom)); scale=min(700/crop.width,720/crop.height)
size=(round(crop.width*scale),round(crop.height*scale)); crop=crop.resize(size,Image.Resampling.LANCZOS)
boxes=json.loads((root/f'analysis_results/seadronessee_validation/public/full/cache/{ident}.json').read_text(encoding='utf8'))['boxes']
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',26); small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
def panel(threshold, previous):
    result=crop.copy(); d=ImageDraw.Draw(result)
    for a in anns:
        x,y,w,h=a['bbox']; d.rectangle(((x-left)*scale,(y-top)*scale,(x+w-left)*scale,(y+h-top)*scale),outline='#00f030',width=3)
    for x,y,r,b,s in sorted((b for b in boxes if b[4]>=threshold),key=lambda a:a[4]):
        if r<=left or x>=right or b<=top or y>=bottom: continue
        x1=max(0,(x-left)*scale); y1=max(0,(y-top)*scale); x2=min(size[0]-1,(r-left)*scale); y2=min(size[1]-1,(b-top)*scale)
        color='#ffce00' if previous is not None and s < previous else '#ff3030'; d.rectangle((x1,y1,x2,y2),outline=color,width=4)
    return result
pw=max(700,size[0]); ph=size[1]+52; width=max(whole.width,2*pw+30)+80; height=whole.height+2*ph+175
page=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(page)
d.text((40,10),f'전체 원본 {ident}  |  {im["width"]} × {im["height"]} px',font=font,fill='black'); page.paste(whole,((width-whole.width)//2,50))
y=whole.height+68; d.text((40,y),'초록: 정답   빨강: 기존 예측   노랑: 바로 앞 기준에서 새로 포함된 예측',font=font,fill='black'); y+=45
for i,t in enumerate((.20,.15,.10,.05)):
    x=40+(i%2)*(pw+30); yy=y+(i//2)*ph; d.text((x,yy),f'SSPNet · {t:.2f} 이상',font=font,fill='black'); page.paste(panel(t,None if i==0 else (.20,.15,.10)[i-1]),(x,yy+42))
page.save(out/f'comparison_{ident}_stepwise.jpg',quality=94)
print('완료')
