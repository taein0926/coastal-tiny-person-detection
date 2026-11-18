"""한 번 저장한 person 예측을 전체 사람·물속 사람 기준으로 각각 집계한다."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from metrics import geometric_size, load_dataset, match_predictions


def score_scope(gt, predictions, score, iou):
    selected = [p for p in predictions if p["score"] >= score]
    matches = match_predictions(gt, selected, iou)
    normal = [a for a in gt["annotations"] if not a.get("ignore") and not a.get("iscrowd")]
    annotations_by_image = defaultdict(list)
    matches_by_image = defaultdict(list)
    for annotation in normal:
        annotations_by_image[annotation["image_id"]].append(annotation)
    for record in matches:
        matches_by_image[record[0]["image_id"]].append(record)

    def group(images):
        annotations = [a for im in images for a in annotations_by_image[im["id"]]]
        records = [r for im in images for r in matches_by_image[im["id"]]]
        hits = [a for _, a, ignored in records if a is not None and not ignored]
        false_positive = sum(a is None and not ignored for _, a, ignored in records)
        true_positive = len(hits)
        def sizes(low, high):
            positives = sum(low <= geometric_size(a["bbox"]) < high for a in annotations)
            found = sum(low <= geometric_size(a["bbox"]) < high for a in hits)
            return dict(positives=positives, hits=found, recall=found/positives if positives else None)
        return dict(images=len(images), positives=len(annotations), true_positives=true_positive,
                    false_positives=false_positive, false_negatives=len(annotations)-true_positive,
                    precision=true_positive/(true_positive+false_positive) if true_positive+false_positive else None,
                    recall=true_positive/len(annotations) if annotations else None,
                    false_positives_per_image=false_positive/len(images),
                    geometric_2to20=sizes(2,20), geometric_2to8=sizes(2,8))

    result = group(gt["images"])
    result.update(score_threshold=score, iou=iou,
                  geometric_size_definition="s=sqrt(width*height), 배포 이미지 픽셀 기준")
    splits = sorted({im.get("source_split", "single") for im in gt["images"]})
    result["by_split"] = {s:group([im for im in gt["images"] if im.get("source_split", "single")==s]) for s in splits}
    return result


def summarize_scopes(annotations, out, score=.25, iou=.3):
    if not 0 < score <= 1 or not 0 < iou <= 1:
        raise ValueError("점수와 IoU 기준은 0 초과 1 이하여야 합니다.")
    out = Path(out)
    predictions = json.loads((out/"predictions_coco.json").read_text(encoding="utf-8"))
    saved_path = out/"ground_truth_water.json"
    image_ids = {im["id"] for im in json.loads(saved_path.read_text(encoding="utf-8"))["images"]} if saved_path.is_file() else None
    scopes = (("all_people", ["person water","person sand"], 8099, "전체 사람"),
              ("water_only", ["person water"], 2904, "물속 사람"))
    values = {}
    reference = {}
    thresholds = [iou] if iou==.5 else [iou,.5]
    for name, classes, expected, label in scopes:
        gt, _ = load_dataset(annotations, classes)
        available_images = len(gt["images"])
        if image_ids is not None:
            gt["images"] = [im for im in gt["images"] if im["id"] in image_ids]
            gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in image_ids]
        if not gt["images"]:
            raise ValueError("평가할 이미지가 없습니다.")
        gt["info"]["description"] = label+" 평가용 정답"
        gt["categories"] = [dict(id=1,name="person" if name=="all_people" else "person water")]
        if name=="all_people":
            (out/"ground_truth_all_people.json").write_text(json.dumps(gt,ensure_ascii=False),encoding="utf-8")
        for threshold in thresholds:
            result = score_scope(gt, predictions, score, threshold)
            result.update(target_classes=classes,scope_label=label,
                          partial=len(gt["images"])<available_images,available_image_count=available_images)
            primary = threshold==iou
            filename = "metrics_%s%s.json" % (name,"" if primary else "_iou05_reference")
            (out/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
            (values if primary else reference)[name] = result

    def percent(value):
        return "계산 불가" if value is None else "%.1f%%" % (100*value)
    lines = ["SSPNet 전체 사람·물속 사람 동시 평가", "", "동일한 person 예측을 두 가지 정답 범위로 각각 평가했습니다.",
             "이미지: %d장 / 예측 점수: %.2f / 주 결과 IoU: %.2f" % (values["all_people"]["images"],score,iou), ""]
    for name, _, _, label in scopes:
        m = values[name]
        lines += ["[%s 기준]" % label,
                  "정답 %s개 / 정탐 %s개 / 놓친 정답 %s개" % tuple(format(m[k],",") for k in ("positives","true_positives","false_negatives")),
                  "재현율 %s / 정밀도 %s" % (percent(m["recall"]),percent(m["precision"])),
                  "주석 기준 오탐 %s개 / 이미지당 %.2f개" % (format(m["false_positives"],","),m["false_positives_per_image"]), ""]
    lines += ["전체 사람 기준에서는 모래사장 사람도 정답입니다.",
              "물속 사람 기준에서는 모래사장 사람에 연결되는 예측도 오탐에 포함됩니다.",
              "주석 누락·위치 오류는 보정하지 않았습니다. 동일한 예측을 범위별로 다시 매칭하므로 정탐 수는 단순 합산으로 연결되지 않을 수 있습니다.",
              "크기 제한 없는 결과가 주 결과이며 크기별 재현율과 train·valid·test별 지표도 각 JSON에 저장했습니다.",
              "IoU 0.5 참고 결과도 저장했습니다." if reference else "현재 IoU 0.5로 집계했습니다."]
    report = "\n".join(lines)+"\n"
    (out/"전체사람_물속사람_결과요약.txt").write_text(report,encoding="utf-8")
    print(report,flush=True)
    return values


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="전체 사람과 물속 사람의 동시 결과 집계")
    parser.add_argument("--annotations",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--score",type=float,default=.25)
    parser.add_argument("--iou",type=float,default=.3)
    args=parser.parse_args()
    summarize_scopes(args.annotations,args.out,args.score,args.iou)
