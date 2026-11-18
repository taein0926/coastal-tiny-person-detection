from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
out = root / 'analysis_results/public_video_comparison'
font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 26)
small = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 19)
cases = [
    ('pexels_19953301', '기존_검토영역', (20, 1080, 640, 1600)),
    ('pexels_37277130', '작은_사람_직접_주석영역', (1125, 520, 1205, 555)),
]
for name, tag, region in cases:
    original = Image.open(root / f'analysis_results/recall_v3_review/videos/{name}/000000.png').convert('RGB')
    compare = Image.open(out / f'{name}_{tag}_주석_비교.png').convert('RGB')
    scale = min(1, 1800 / original.width, 900 / original.height)
    full = original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
    if compare.width > 1800:
        compare = compare.resize((1800, round(compare.height * 1800 / compare.width)), Image.Resampling.LANCZOS)
    width = max(full.width, compare.width, 900)
    canvas = Image.new('RGB', (width, full.height + compare.height + 125), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), '원본 전체 화면 · 첫 번째 프레임', font=font, fill='black')
    left = (width - full.width) // 2
    canvas.paste(full, (left, 50))
    x1, y1, x2, y2 = region
    box = (left + round(x1 * scale), 50 + round(y1 * scale), left + round(x2 * scale), 50 + round(y2 * scale))
    draw.rectangle(box, outline='#ffcc00', width=3)
    draw.text((20, full.height + 58), '아래 비교 영역: 노란색 테두리', font=small, fill='black')
    canvas.paste(compare, ((width - compare.width) // 2, full.height + 100))
    canvas.save(out / f'{name}_원본_포함_채점_비교.png')
    print(out / f'{name}_원본_포함_채점_비교.png')
