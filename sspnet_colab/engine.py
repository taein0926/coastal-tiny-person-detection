"""SSPNet 추론, 분할 병합, 이미지별 재시작 캐시."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from metrics import evaluate, image_path, load_dataset


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(tmp), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positions(length, size, overlap):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("분할 크기는 양수, 겹침은 0 이상 1 미만이어야 합니다.")
    if length <= size:
        return [0]
    step = max(1, int(size*(1-overlap)))
    values = list(range(0, length-size+1, step))
    if values[-1] != length-size:
        values.append(length-size)
    return values


def merge_nms(boxes, threshold=.5):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if not len(boxes):
        return boxes
    order = np.argsort(-boxes[:, 4], kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        b = boxes[rest]
        inter = np.maximum(0, np.minimum(boxes[i, 2:4], b[:, 2:4])-np.maximum(boxes[i, :2], b[:, :2])).prod(axis=1)
        area = np.maximum(0, boxes[i, 2:4]-boxes[i, :2]).prod()
        areas = np.maximum(0, b[:, 2:4]-b[:, :2]).prod(axis=1)
        union = area+areas-inter
        iou = inter/np.maximum(union, 1e-12)
        order = rest[iou <= threshold]
    return boxes[keep]


def make_model(settings):
    import torch
    from mmcv import Config
    from mmdet.models import build_detector
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 없습니다. 무료 Colab에서 GPU 런타임을 선택하세요.")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(str(Path(settings["repo"])/"configs/_base_/models/faster_rcnn_r50_sspnet.py"))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.test_cfg.rcnn.score_thr = settings["save_min_score"]
    cfg.test_cfg.rcnn.nms.min_score = settings["save_min_score"]
    cfg.test_cfg.rcnn.max_per_img = settings["max_detections"]
    cfg.data = dict(test=dict(pipeline=[
        dict(type="LoadImageFromFile"),
        dict(type="MultiScaleFlipAug", img_scale=(settings["long_edge"], settings["short_edge"]),
             flip=False, transforms=[dict(type="Resize", keep_ratio=True), dict(type="RandomFlip"),
             dict(type="Normalize", mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
             dict(type="Pad", size_divisor=32), dict(type="ImageToTensor", keys=["img"]),
             dict(type="Collect", keys=["img"])])]))
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    checkpoint = torch.load(settings["checkpoint"], map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k,v in state.items()}
    if not any("neck.CAM." in k for k in state):
        raise ValueError("SSPNet 가중치가 아닙니다. Faster R-CNN SSPNet용 파일을 확인하세요.")
    model.load_state_dict(state, strict=True)
    model.CLASSES = ("person",)
    model.cfg = cfg
    model.cuda().eval()
    return model, cfg


def predict(model, image, settings):
    import torch
    from mmdet.apis import inference_detector
    rgb = np.asarray(Image.open(image).convert("RGB"))
    h, w = rgb.shape[:2]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    patches = 0
    boxes = []
    if settings["mode"] == "full":
        regions = [(0, 0, w, h)]
    else:
        size = settings["tile_size"]
        regions = [(x,y,min(x+size,w),min(y+size,h))
                   for y in positions(h,size,settings["tile_overlap"])
                   for x in positions(w,size,settings["tile_overlap"])]
    for x,y,r,b in regions:
        # MMDetection의 배열 입력은 BGR 순서를 사용한다.
        patch = np.ascontiguousarray(rgb[y:b,x:r,::-1])
        result = inference_detector(model, patch)
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError("단일 person 클래스 SSPNet 출력이 아닙니다.")
        out = np.asarray(result[0], dtype=np.float32).reshape(-1,5).copy()
        out[:,[0,2]] += x
        out[:,[1,3]] += y
        boxes.extend(out.tolist())
        patches += 1
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1,5)
    if len(boxes):
        boxes[:,[0,2]] = boxes[:,[0,2]].clip(0,w)
        boxes[:,[1,3]] = boxes[:,[1,3]].clip(0,h)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
        boxes = boxes[valid]
    if settings["mode"] == "tiles":
        boxes = merge_nms(boxes,settings["merge_iou"])
    if len(boxes):
        boxes = boxes[np.argsort(-boxes[:,4],kind="stable")][:settings["max_detections"]]
    torch.cuda.synchronize()
    return dict(boxes=boxes.tolist(), seconds=time.perf_counter()-started, patches=patches,
                peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2)


def draw(image, record, path, threshold):
    picture = Image.open(image).convert("RGB")
    pen = ImageDraw.Draw(picture)
    for x,y,r,b,s in record["boxes"]:
        if s >= threshold:
            pen.rectangle((x,y,r,b),outline="red",width=2)
            pen.text((x,max(0,y-12)),"person %.2f" % s,fill="red")
    picture.save(path)


def coco_predictions(records):
    return [dict(image_id=im_id,category_id=1,bbox=[x,y,r-x,b-y],score=s)
            for im_id,rec in records.items() for x,y,r,b,s in rec["boxes"]]


def report_text(metrics, settings, summary):
    def percent(value):
        return "계산 불가" if value is None else "%.1f%%" % (100*value)
    small = metrics["geometric_2to20"]
    smallest = metrics["geometric_2to8"]
    method = "이미지 전체를 한 번에 검사" if settings["mode"] == "full" else \
        "%d픽셀 패치로 분할 검사(겹침 %.0f%%)" % (settings["tile_size"],settings["tile_overlap"]*100)
    lines = ["SSPNet 수영자 탐지 평가", "",
             "평가 데이터: %s / 분할: %s / 이미지 %d장" %
             (settings.get("dataset_name","사용자 데이터"), settings.get("split","single"), metrics["images"]),
             "평가 방식: " + method,
             "공개 TinyPerson 학습 가중치를 사용했으며 이번 데이터로 추가 학습하지 않았다.",
             "전체 물속 사람 정답 %d개 중 %d개를 찾아 재현율 %s, 정밀도 %s였다." %
             (metrics["positives"],metrics["true_positives"],percent(metrics["recall"]),percent(metrics["precision"])),
             "사람 크기는 배포 이미지에서 s=√(박스 가로×박스 세로)로 정의했다.",
             "2≤s<20픽셀 물속 사람 %d개 중 %d개를 찾아 재현율 %s였다." %
             (small["positives"],small["hits"],percent(small["recall"])),
             "가장 작은 2≤s<8픽셀 구간에서는 %d개 중 %d개를 찾아 재현율 %s였다." %
             (smallest["positives"],smallest["hits"],percent(smallest["recall"])),
             "오탐은 %d개(이미지당 %.2f개)였고, 점수 임계값 %.3f와 IoU %.2f를 적용했다." %
             (metrics["false_positives"],metrics["false_positives_per_image"],metrics["score_threshold"],metrics["iou"])]
    if settings.get("split") == "all":
        lines.append("train·valid·test와 배포된 증강 이미지를 포함한 전체 분할 평가이며, 독립 테스트 분할 성능과 구분한다.")
    if metrics["partial"]:
        lines.append("현재 결과는 일부 이미지만 처리한 표본 평가다.")
    lines += ["", "분할별 구성: " + json.dumps(summary["split_counts"],ensure_ascii=False),
              "입력 해상도 상한: %d×%d, 종횡비 유지 / 최대 탐지 수: %d" %
              (settings["long_edge"],settings["short_edge"],settings["max_detections"]),
              "검출 결과는 전체 물속 사람 정답과 일대일로 매칭한 뒤 크기 구간별로 집계했다.",
              "이 파일은 이번 SSPNet 실행 결과만 정리하며 다른 모델과의 비교는 포함하지 않는다."]
    return "\n".join(lines) + "\n"


def run(args):
    import torch
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    if settings["mode"] not in ("full","tiles"):
        raise ValueError("추론 방식은 full 또는 tiles여야 합니다.")
    if not 0 < settings["save_min_score"] <= settings["report_score"] <= 1:
        raise ValueError("저장 점수와 보고 점수의 범위를 확인하세요.")
    if not 0 < settings.get("match_iou",.5) <= 1:
        raise ValueError("정답 매칭 IoU는 0 초과 1 이하여야 합니다.")
    if settings["mode"] == "tiles":
        positions(1,settings["tile_size"],settings["tile_overlap"])
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    settings["checkpoint_sha256"] = sha256(settings["checkpoint"])
    settings["source_commit"] = subprocess.check_output(["git","-C",settings["repo"],"rev-parse","HEAD"],text=True).strip()
    settings["engine_sha256"] = sha256(__file__)
    settings["metrics_sha256"] = sha256(Path(__file__).with_name("metrics.py"))
    if args.image:
        model,cfg = make_model(settings)
        cfg.dump(str(out/"resolved_config.py"))
        # 첫 호출의 초기화 비용은 별도 기록하고 본 측정에서 제외한다.
        warmup = predict(model,args.image,dict(settings,mode="full"))
        rec = predict(model,args.image,settings)
        rec.update(image_sha256=sha256(args.image), image_file_name=Path(args.image).name)
        dump(out/"predictions.json",rec)
        dump(out/"settings.json",dict(settings,warmup_seconds=warmup["seconds"],gpu=torch.cuda.get_device_name(0)))
        draw(args.image,rec,out/"prediction.jpg",settings["report_score"])
        print("이미지 추론 완료: 박스 %d개, 표시 임계값 %.2f, %.3f초" %
              (len(rec["boxes"]),settings["report_score"],rec["seconds"]))
        print("출력:",out)
        return
    gt,summary = load_dataset(args.annotations,settings["target_names"],args.limit)
    if not summary["target_count"]:
        raise ValueError("선택한 이미지에 평가할 물속 사람 정답이 없습니다.")
    dump(out/"dataset_summary.json",summary)
    settings["annotations_sha256"] = sha256(args.annotations)
    identity = dict(settings)
    # 결과 표시 임계값을 바꿔도 저점수 예측을 재사용할 수 있다.
    identity.pop("report_score",None)
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    cache = out/("cache_"+key)
    cache.mkdir(exist_ok=True)
    dump(cache/"identity.json",identity)
    dump(out/"run_status.json",dict(state="실행 중",cache_key=key,
         image_ids=[im["id"] for im in gt["images"]]))
    records = {}
    pending = []
    for im in gt["images"]:
        path = image_path(args.annotations,im["file_name"])
        image_hash = sha256(path)
        saved = cache/(str(im["id"])+".json")
        try:
            rec = json.loads(saved.read_text(encoding="utf-8"))
        except (FileNotFoundError,json.JSONDecodeError):
            rec = None
        if rec and rec.get("image_sha256") == image_hash:
            records[im["id"]] = rec
        else:
            pending.append((im,path,image_hash,saved))
    warmup_seconds = None
    if pending:
        model,cfg = make_model(settings)
        cfg.dump(str(out/"resolved_config.py"))
        warmup_seconds = predict(model,pending[0][1],dict(settings,mode="full"))["seconds"]
        for im,path,image_hash,saved in pending:
            rec = predict(model,path,settings)
            rec.update(image_sha256=image_hash,file_name=im["file_name"])
            dump(saved,rec)
            records[im["id"]] = rec
            print("완료 %d/%d: %s" % (len(records),len(gt["images"]),im["file_name"]),flush=True)
            # 적은 수의 시각화만 저장하여 무료 저장공간을 아낀다.
            if len(records) <= 5:
                draw(path,rec,out/("preview_%s.jpg" % im["id"]),settings["report_score"])
    predictions = coco_predictions(records)
    dump(out/"ground_truth_water.json",gt)
    dump(out/"predictions_coco.json",predictions)
    metrics = evaluate(gt,predictions,settings["report_score"],settings.get("match_iou",.5))
    metrics.update(partial=len(gt["images"]) < summary["available_image_count"],requested_limit=args.limit,cache_key=key,
                   mean_inference_seconds=sum(r["seconds"] for r in records.values())/len(records),
                   maximum_vram_mib=max(r["peak_vram_mib"] for r in records.values()),
                   warmup_seconds=warmup_seconds)
    metrics["by_split"] = {}
    for split in summary["split_counts"]:
        image_ids = {im["id"] for im in gt["images"] if im.get("source_split","single") == split}
        split_gt = dict(gt, images=[im for im in gt["images"] if im["id"] in image_ids],
                        annotations=[a for a in gt["annotations"] if a["image_id"] in image_ids])
        metrics["by_split"][split] = evaluate(split_gt,predictions,settings["report_score"],settings.get("match_iou",.5))
    dump(out/"metrics.json",metrics)
    (out/"문서용_결과요약.txt").write_text(report_text(metrics,settings,summary),encoding="utf-8")
    dump(out/"run_manifest.json",dict(settings,dataset=summary,
         image_ids=[im["id"] for im in gt["images"]],gpu=torch.cuda.get_device_name(0)))
    dump(out/"run_status.json",dict(state="완료",cache_key=key,
         image_ids=[im["id"] for im in gt["images"]]))
    with open(out/"metrics.csv","w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["지표","값"])
        for name,value in metrics.items():
            if not isinstance(value,dict):
                writer.writerow([name,value])
        for name,value in metrics["size_recall"].items():
            writer.writerow(["크기별 재현율 "+name,value["recall"]])
        for name in ("geometric_2to20","geometric_2to8"):
            for item,value in metrics[name].items():
                writer.writerow([name+" "+item,value])
        for name,value in metrics["geometric_size_recall"].items():
            writer.writerow(["기하평균 크기별 재현율 "+name,value["recall"]])
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="SSPNet 단일 이미지 추론과 물속 사람 평가")
    parser.add_argument("--settings",required=True)
    parser.add_argument("--out",required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image")
    group.add_argument("--annotations")
    parser.add_argument("--limit",type=int,default=0)
    run(parser.parse_args())
