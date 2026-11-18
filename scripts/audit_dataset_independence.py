"""두 COCO ZIP의 파일·픽셀·원본 이름·촬영시각과 시각적 유사 후보를 점검한다."""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_results/dataset_independence_20261111"
INPUTS = {
    "YoloBeachTrain_v2": Path("C:/Users/admin/Downloads/YoloBeachTrain.v2i.coco.zip"),
    "AnnotationPiriapolis_v4": Path("C:/Users/admin/Downloads/AnnotationPiriapolis.v4i.coco.zip"),
}
N = 32
xx = np.arange(N, dtype=np.float32)
DCT = np.cos(np.pi * (2 * xx[None, :] + 1) * xx[:, None] / (2 * N))
DCT[0] *= 1 / np.sqrt(2)


def features(im):
    gray = np.asarray(im.convert("L").resize((N, N)), dtype=np.float32)
    coeff = (DCT @ gray @ DCT.T)[:8, :8].reshape(-1)[1:]
    bits = coeff > np.median(coeff)
    small = np.asarray(im.convert("L").resize((64, 32)), dtype=np.float32).reshape(-1)
    small -= small.mean()
    small /= max(float(np.linalg.norm(small)), 1e-8)
    return bits, small


def load(name, path):
    records = []
    pictures = []
    with zipfile.ZipFile(path) as z:
        for ann_path in sorted(n for n in z.namelist() if n.endswith(".coco.json")):
            split = ann_path.split("/")[0]
            data = json.loads(z.read(ann_path))
            for item in data["images"]:
                member = str(Path(ann_path).parent / item["file_name"]).replace("\\", "/")
                raw = z.read(member)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                source = re.sub(r"\.(?:jpg|png)$", "", item["file_name"].split(".rf.")[0], flags=re.I)
                date = re.search(r"20\d{2}_\d{2}_\d{2}", source)
                stamp = re.search(r"20\d{2}_\d{2}_\d{2}_\d{2}(?:_\d{2})?", source)
                records.append(dict(
                    split=split, file_name=item["file_name"], member=member,
                    source_key=source, capture_frame_key=re.sub(r"^Part_\d+", "", source), date=date.group() if date else None,
                    capture_time_key=stamp.group() if stamp else None,
                    byte_sha256=hashlib.sha256(raw).hexdigest(),
                    pixel_sha256=hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest(),
                    width=im.width, height=im.height,
                ))
                pictures.append(im.resize((320, 160)))
    print(name, len(records), "이미지 읽기 완료", flush=True)
    return records, pictures


def summary(records):
    dates = sorted({r["date"] for r in records if r["date"]})
    names = defaultdict(set)
    times = defaultdict(set)
    frames = defaultdict(set)
    for r in records:
        names[r["source_key"]].add(r["split"])
        frames[r["capture_frame_key"]].add(r["split"])
        if r["capture_time_key"]:
            times[r["capture_time_key"]].add(r["split"])
    cross = {k: sorted(v) for k, v in times.items() if len(v) > 1}
    return dict(
        images=len(records), splits=dict(Counter(r["split"] for r in records)),
        unique_source_keys=len(names), missing_date_count=sum(r["date"] is None for r in records),
        min_date=dates[0] if dates else None, max_date=dates[-1] if dates else None,
        date_count=len(dates), unique_pixel_hashes=len({r["pixel_sha256"] for r in records}),
        source_keys_across_splits={k: sorted(v) for k, v in names.items() if len(v) > 1},
        unique_capture_frame_keys=len(frames),
        capture_frame_keys_across_splits={k: sorted(v) for k, v in frames.items() if len(v) > 1},
        capture_time_keys_across_splits=cross,
        capture_time_warning="같은 촬영시각 이름의 서로 다른 크롭일 수 있으며 동일 픽셀이라는 뜻은 아님",
    )


def montage_pairs(pairs, a, b, ap, bp, filename):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    sheet = Image.new("RGB", (1000, len(pairs) * 195), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(pairs):
        ai, bi = pair["a"], pair["b"]
        y = row * 195
        sheet.paste(ap[ai], (0, y + 24))
        sheet.paste(bp[bi], (330, y + 24))
        draw.text((0, y), f"YoloBeachTrain {a[ai]['split']} | Piriapolis {b[bi]['split']} | {pair['kind']}={pair['value']:.4f}", fill="black", font=font)
        draw.text((660, y + 30), a[ai]["file_name"][:40], fill="black", font=font)
        draw.text((660, y + 60), b[bi]["file_name"][:40], fill="black", font=font)
        draw.text((660, y + 90), "variant: " + pair["variant"], fill="black", font=font)
    sheet.save(OUT / filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = [load(name, path) for name, path in INPUTS.items()]
    (a, ap), (b, bp) = loaded
    names = list(INPUTS)
    report = dict(inputs={k: dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k, v in INPUTS.items()},
                  datasets={name: summary(records) for name, (records, _) in zip(names, loaded)})
    overlaps = {}
    for key in ("byte_sha256", "pixel_sha256", "source_key", "date", "capture_time_key", "capture_frame_key"):
        common = {r[key] for r in a if r[key]} & {r[key] for r in b if r[key]}
        overlaps[key] = dict(common_unique_values=len(common),
                             a_images=sum(r[key] in common for r in a),
                             b_images=sum(r[key] in common for r in b), examples=sorted(common)[:20])
    report["cross_dataset_overlaps"] = overlaps
    # 각 Piriapolis 이미지를 반전·회전을 포함해 전체 학습 후보와 비교한다.
    variants = ["original", "flip_horizontal", "flip_vertical", "rotate180", "rotate90", "rotate270"]
    transforms = [lambda x: x, ImageOps.mirror, ImageOps.flip,
                  lambda x: x.transpose(Image.Transpose.ROTATE_180),
                  lambda x: x.transpose(Image.Transpose.ROTATE_90),
                  lambda x: x.transpose(Image.Transpose.ROTATE_270)]
    af = [features(im) for im in ap]
    ah = np.asarray([f[0] for f in af], dtype=np.int16)
    av = np.asarray([f[1] for f in af])
    phash_candidates, ncc_candidates = [], []
    for start in range(0, len(b), 64):
        ids, hashes, vectors = [], [], []
        for bi in range(start, min(start + 64, len(b))):
            for vi, trans in enumerate(transforms):
                hh, vv = features(trans(bp[bi]))
                ids.append((bi, vi)); hashes.append(hh); vectors.append(vv)
        bh = np.asarray(hashes, dtype=np.int16)
        bv = np.asarray(vectors)
        distances = ah.sum(1)[:, None] + bh.sum(1)[None, :] - 2 * ah @ bh.T
        correlations = av @ bv.T
        for j, (bi, vi) in enumerate(ids):
            ai = int(np.argmin(distances[:, j]))
            phash_candidates.append(dict(a=ai, b=bi, variant=variants[vi], kind="phash_distance", value=int(distances[ai, j]), ncc=float(correlations[ai, j])))
            ai = int(np.argmax(correlations[:, j]))
            ncc_candidates.append(dict(a=ai, b=bi, variant=variants[vi], kind="normalized_correlation", value=float(correlations[ai, j]), phash_distance=int(distances[ai, j])))
    phash_candidates.sort(key=lambda x: (x["value"], -x["ncc"]))
    ncc_candidates.sort(key=lambda x: -x["value"])
    report["visual_screen"] = dict(
        method="전체 화면 63bit pHash와 평균/대비 정규화 상관계수; Piriapolis에 6가지 방향 적용. 다른 크롭의 부분 겹침을 모두 검출하는 방식은 아님.",
        min_phash_distance=phash_candidates[0]["value"], max_normalized_correlation=ncc_candidates[0]["value"],
        phash_le_6_candidate_count=sum(x["value"] <= 6 for x in phash_candidates),
        ncc_ge_098_candidate_count=sum(x["value"] >= .98 for x in ncc_candidates),
        top_phash=phash_candidates[:30], top_ncc=ncc_candidates[:30],
    )
    montage_pairs(phash_candidates[:8], a, b, ap, bp, "nearest_phash_pairs.png")
    montage_pairs(ncc_candidates[:8], a, b, ap, bp, "nearest_correlation_pairs.png")
    # 각 데이터에서 4행씩, 서로 다른 원본 이름의 표본을 골라 촬영 배경을 확인한다.
    sheet = Image.new("RGB", (960, 8 * 180), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    for dataset_index, (records, pics) in enumerate(loaded):
        seen, candidates = set(), []
        for i, r in enumerate(records):
            if r["source_key"] not in seen:
                seen.add(r["source_key"]); candidates.append(i)
        for j, pos in enumerate(np.linspace(0, len(candidates) - 1, 12).astype(int)):
            idx = candidates[pos]; y = dataset_index * 720 + (j // 3) * 180
            x = (j % 3) * 320
            sheet.paste(pics[idx], (x, y + 20))
            draw.text((x, y), names[dataset_index] + " " + records[idx]["source_key"][:22], fill="black", font=font)
    sheet.save(OUT / "dataset_samples.png")
    (OUT / "image_inventory.json").write_text(json.dumps(dict(zip(names, [a, b])), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "audit_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(overlaps=overlaps, datasets={k: {kk: (len(vv) if kk.endswith("across_splits") else vv) for kk, vv in v.items()} for k, v in report["datasets"].items()}), ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report["visual_screen"].items() if not k.startswith("top_")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
