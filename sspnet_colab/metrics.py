"""원본 이미지 좌표에서 수영자 탐지와 크기별 재현율을 평가한다."""
import json
import math
import re
import os
from collections import Counter, defaultdict
from pathlib import Path


def canonical(value):
    return " ".join(re.sub(r"[_-]+", " ", value.lower()).split())


def image_path(annotation_path, name):
    base = Path(annotation_path).resolve().parent
    path = (base / name).resolve()
    if base != path and base not in path.parents:
        raise ValueError("이미지 경로가 데이터셋 폴더 밖을 가리킵니다: " + name)
    if not path.is_file():
        raise FileNotFoundError("정답 파일과 같은 폴더 아래에 이미지가 필요합니다: " + str(path))
    return path


def size_group(box):
    d = max(float(box[2]), float(box[3]))
    return "0<d<=8" if d <= 8 else "8<d<=12" if d <= 12 else "12<d<=20" if d <= 20 else "20<d"


def geometric_size(box):
    return math.sqrt(float(box[2]) * float(box[3]))


def merge_splits(annotation_paths, output_path):
    """분할별로 겹치는 ID와 파일명을 구분하고 이미지 복사 없이 합친다."""
    output_path = Path(output_path).resolve()
    base = output_path.parent
    merged = dict(info={"description": "전체 분할 통합", "source_splits": []},
                  licenses=[], images=[], annotations=[], categories=[])
    categories = {}
    seen_paths = set()
    for split, source in annotation_paths:
        source = Path(source).resolve()
        if split in merged["info"]["source_splits"]:
            raise ValueError("같은 분할을 두 번 합칠 수 없습니다: " + split)
        raw = json.loads(source.read_text(encoding="utf-8"))
        merged["info"]["source_splits"].append(split)
        catmap = {}
        for category in raw["categories"]:
            name = canonical(category["name"])
            if name not in categories:
                categories[name] = len(categories) + 1
                merged["categories"].append(dict(id=categories[name], name=category["name"]))
            catmap[category["id"]] = categories[name]
        images = {}
        for im in sorted(raw["images"], key=lambda v: v["id"]):
            if im["id"] in images:
                raise ValueError("분할 안에서 이미지 ID가 중복됩니다: " + split)
            path = image_path(source, im["file_name"])
            if path in seen_paths:
                raise ValueError("같은 이미지 경로가 중복됩니다: " + str(path))
            seen_paths.add(path)
            if base not in path.parents:
                raise ValueError("통합 정답은 모든 이미지의 상위 폴더에 저장해야 합니다.")
            new_id = len(merged["images"])
            images[im["id"]] = new_id
            merged["images"].append(dict(im, id=new_id, file_name=path.relative_to(base).as_posix(),
                                         source_split=split, source_image_id=im["id"]))
        seen_annotations = set()
        for a in raw["annotations"]:
            if a["id"] in seen_annotations:
                raise ValueError("분할 안에서 정답 ID가 중복됩니다: " + split)
            seen_annotations.add(a["id"])
            if a["image_id"] not in images or a["category_id"] not in catmap:
                raise ValueError("정답의 이미지 또는 종류 ID를 찾지 못했습니다: " + split)
            merged["annotations"].append(dict(a, id=len(merged["annotations"])+1,
                image_id=images[a["image_id"]], category_id=catmap[a["category_id"]],
                source_split=split, source_annotation_id=a["id"]))
    if not merged["images"]:
        raise ValueError("합칠 이미지가 없습니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(output_path))
    return output_path


def load_dataset(annotation_path, target_names=("person water",), limit=0):
    raw = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    cats = raw.get("categories", [])
    wanted = {canonical(s) for s in target_names}
    selected = [c["id"] for c in cats if canonical(c["name"]) in wanted]
    if not selected:
        raise ValueError("물속 사람 라벨을 찾지 못했습니다. 실제 종류: " + str([c["name"] for c in cats]))
    images = sorted(raw["images"], key=lambda im: im["id"])
    if len({im["id"] for im in images}) != len(images):
        raise ValueError("이미지 ID가 중복되어 있습니다.")
    if limit < 0:
        raise ValueError("이미지 제한은 0 이상이어야 합니다.")
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError("선택한 분할에 이미지가 없습니다.")
    imap = {im["id"]: im for im in images}
    ann_ids = set()
    annotations = []
    for a in raw["annotations"]:
        if a["id"] in ann_ids:
            raise ValueError("정답 ID가 중복되어 있습니다.")
        ann_ids.add(a["id"])
        if a["image_id"] not in imap or a["category_id"] not in selected:
            continue
        x, y, w, h = map(float, a["bbox"])
        im = imap[a["image_id"]]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError("유효하지 않은 정답 박스: " + str(a["id"]))
        if x < -.5 or y < -.5 or x + w > im["width"] + .5 or y + h > im["height"] + .5:
            raise ValueError("이미지 밖의 정답 박스입니다. 좌표 변환을 확인하세요: " + str(a["id"]))
        annotations.append(dict(id=a["id"], image_id=a["image_id"], category_id=1,
                                bbox=[x, y, w, h], area=w*h,
                                iscrowd=int(a.get("iscrowd", 0)),
                                ignore=bool(a.get("ignore", False)), segmentation=[]))
    for im in images:
        path = image_path(annotation_path, im["file_name"])
        from PIL import Image
        with Image.open(path) as pic:
            if pic.size != (im["width"], im["height"]):
                raise ValueError("이미지 크기와 정답 메타데이터가 다릅니다: " + str(path))
    gt = dict(info={"description": "물속 사람 평가용 정답"}, licenses=[], images=images,
              annotations=annotations, categories=[dict(id=1, name="person water")])
    counts = Counter(size_group(a["bbox"]) for a in annotations if not a["ignore"] and not a["iscrowd"])
    normal = [a for a in annotations if not a["ignore"] and not a["iscrowd"]]
    split_images = Counter(im.get("source_split", "single") for im in images)
    image_splits = {im["id"]: im.get("source_split", "single") for im in images}
    split_targets = Counter(image_splits[a["image_id"]] for a in normal)
    return gt, dict(image_count=len(images), available_image_count=len(raw["images"]), target_count=sum(counts.values()),
                    geometric_2to20_count=sum(2 <= geometric_size(a["bbox"]) < 20 for a in normal),
                    geometric_2to8_count=sum(2 <= geometric_size(a["bbox"]) < 8 for a in normal),
                    split_counts={s: dict(images=n, water_targets=split_targets[s]) for s,n in split_images.items()},
                    size_counts=dict(counts), available_classes=[c["name"] for c in cats],
                    target_classes=[c["name"] for c in cats if c["id"] in selected])


def overlap(a, b, crowd=False):
    x, y, w, h = a
    u, v, s, t = b
    inter = max(0., min(x+w, u+s)-max(x, u)) * max(0., min(y+h, v+t)-max(y, v))
    area = w*h if crowd else w*h+s*t-inter
    return inter / area if area > 0 else 0.


def match_predictions(gt, predictions, iou=.5):
    normal, ignored = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        (ignored if a.get("ignore") or a.get("iscrowd") else normal)[a["image_id"]].append(a)
    used = set()
    valid_ids = {im["id"] for im in gt["images"]}
    ranked = sorted((p for p in predictions if p["image_id"] in valid_ids), key=lambda p: -p["score"])
    matches = []
    for p in ranked:
        candidates = [(overlap(p["bbox"], a["bbox"]), a) for a in normal[p["image_id"]] if a["id"] not in used]
        score, best = max(candidates, key=lambda pair: pair[0]) if candidates else (0., None)
        if best is not None and score >= iou:
            used.add(best["id"])
            matches.append((p, best, False))
        elif any(overlap(p["bbox"], a["bbox"], bool(a.get("iscrowd"))) >= iou for a in ignored[p["image_id"]]):
            matches.append((p, None, True))
        else:
            matches.append((p, None, False))
    return matches


def evaluate(gt, predictions, score_threshold=.25, iou=.5):
    matches = match_predictions(gt, predictions, iou)
    counts = Counter(size_group(a["bbox"]) for a in gt["annotations"] if not a.get("ignore") and not a.get("iscrowd"))
    positives = sum(counts.values())
    hits, tp, fp = Counter(), 0, 0
    recalls, precisions = [], []
    total_tp = total_fp = 0
    for p, a, ignored in matches:
        if ignored:
            continue
        total_tp += a is not None
        total_fp += a is None
        recalls.append(total_tp / positives if positives else 0.)
        precisions.append(total_tp / (total_tp + total_fp))
        if p["score"] >= score_threshold:
            tp += a is not None
            fp += a is None
            if a is not None:
                hits[size_group(a["bbox"])] += 1
    ap = sum(max((p for r, p in zip(recalls, precisions) if r >= q/100), default=0.) for q in range(101))/101 if positives else None
    tiny_count = sum(counts[k] for k in ("0<d<=8", "8<d<=12", "12<d<=20"))
    tiny_hits = sum(hits[k] for k in ("0<d<=8", "8<d<=12", "12<d<=20"))
    def geometric_range(low, high):
        selected = {a["id"] for a in gt["annotations"] if not a.get("ignore") and not a.get("iscrowd")
                    and low <= geometric_size(a["bbox"]) < high}
        found = sum(a is not None and a["id"] in selected for p,a,ignored in matches
                    if not ignored and p["score"] >= score_threshold)
        return dict(positives=len(selected), hits=found, recall=found/len(selected) if selected else None)
    return dict(iou=iou, score_threshold=score_threshold, images=len(gt["images"]),
                positives=positives, true_positives=tp, false_positives=fp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/positives if positives else None,
                false_positives_per_image=fp/len(gt["images"]),
                AP50_saved_predictions_101point=ap if iou == .5 else None,
                geometric_size_definition="s=sqrt(width*height), 배포 이미지 픽셀 기준",
                geometric_2to20=geometric_range(2,20),
                geometric_2to8=geometric_range(2,8),
                geometric_size_recall={"2<=s<8": geometric_range(2,8),
                                       "8<=s<12": geometric_range(8,12),
                                       "12<=s<20": geometric_range(12,20)},
                tiny_le20=dict(positives=tiny_count, hits=tiny_hits,
                               recall=tiny_hits/tiny_count if tiny_count else None),
                size_recall={k: dict(positives=counts[k], hits=hits[k],
                                    recall=hits[k]/counts[k] if counts[k] else None)
                             for k in ("0<d<=8", "8<d<=12", "12<d<=20", "20<d")})
