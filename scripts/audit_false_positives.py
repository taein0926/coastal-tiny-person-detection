"""사람 전체 기준 오탐을 주석과의 IoU로 상호 배타적으로 분류한다."""
import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sspnet_colab"))
from metrics import canonical, match_predictions, overlap


def audit(results_zip, dataset_zip, output_dir, score=.25, foreground_iou=.5, background_iou=.1):
    if not 0 < background_iou < foreground_iou <= 1:
        raise ValueError("IoU 경계는 0 < 배경 경계 < 정탐 경계 <= 1이어야 합니다.")
    with zipfile.ZipFile(results_zip) as archive:
        prefix = "evaluation/AnnotationPiriapolis_v4/all/full/"
        saved = json.loads(archive.read(prefix + "ground_truth_water.json"))
        predictions = json.loads(archive.read(prefix + "predictions_coco.json"))
        manifest = json.loads(archive.read(prefix + "run_manifest.json"))
    predictions = [dict(p, source_prediction_index=i) for i,p in enumerate(predictions) if p["score"] >= score]
    images = {im["id"]: im for im in saved["images"]}
    image_map = {(im["source_split"],im["source_image_id"]): im["id"] for im in saved["images"]}
    annotations = []
    with zipfile.ZipFile(dataset_zip) as archive:
        for split in ("train","valid","test"):
            raw = json.loads(archive.read(split + "/_annotations.coco.json"))
            names = {c["id"]: canonical(c["name"]) for c in raw["categories"]}
            for a in raw["annotations"]:
                name = names[a["category_id"]]
                if name not in ("person water","person sand"):
                    continue
                annotations.append(dict(a, id=len(annotations)+1,
                    image_id=image_map[(split,a["image_id"])],category_id=1,
                    source_class=name,source_annotation_id=a["id"]))
    ground_truth = dict(images=list(images.values()),annotations=annotations)
    matches = match_predictions(ground_truth,predictions,foreground_iou)
    used_by = {a["id"]: p["source_prediction_index"] for p,a,ignored in matches if a is not None and not ignored}
    by_image = defaultdict(list)
    for a in annotations:
        if not a.get("ignore") and not a.get("iscrowd"):
            by_image[a["image_id"]].append(a)
    rows = []
    counts = Counter()
    split_counts = defaultdict(Counter)
    for p,matched,ignored in matches:
        if ignored or matched is not None:
            continue
        candidates = [(overlap(p["bbox"],a["bbox"]),a) for a in by_image[p["image_id"]]]
        maximum,best = max(candidates,key=lambda pair:pair[0]) if candidates else (0.,None)
        if maximum >= foreground_iou:
            if best["id"] not in used_by:
                raise AssertionError("중복으로 분류한 주석이 정탐에 사용되지 않았습니다.")
            kind = "duplicate"
        elif maximum >= background_iou:
            kind = "localization"
        else:
            kind = "background_candidate"
        im = images[p["image_id"]]
        row = dict(source_prediction_index=p["source_prediction_index"],image_id=p["image_id"],
                   source_split=im["source_split"],file_name=im["file_name"],error_type=kind,
                   score=p["score"],max_person_iou=maximum,
                   prediction_bbox=p["bbox"],nearest_annotation_id=best["id"] if best else None,
                   nearest_source_annotation_id=best["source_annotation_id"] if best else None,
                   nearest_source_class=best["source_class"] if best else None,
                   nearest_person_bbox=best["bbox"] if best else None,
                   accepted_prediction_index=used_by.get(best["id"]) if best else None)
        rows.append(row)
        counts[kind] += 1
        split_counts[im["source_split"]][kind] += 1
    true_positive = sum(a is not None and not ignored for _,a,ignored in matches)
    ignored_count = sum(ignored for _,_,ignored in matches)
    assert true_positive + len(rows) + ignored_count == len(predictions)
    assert len({r["source_prediction_index"] for r in rows}) == len(rows)
    assert sum(counts.values()) == len(rows)
    if (score,foreground_iou) == (.25,.5):
        assert len(images)==635 and len(annotations)==8099
        assert true_positive==1208 and len(rows)==14720
    sensitivity = {}
    for boundary in (.05,.1,.2):
        value = Counter("duplicate" if r["max_person_iou"] >= foreground_iou else
                        "localization" if r["max_person_iou"] >= boundary else
                        "background_candidate" for r in rows)
        sensitivity[str(boundary)] = dict(value)
    summary = dict(images=len(images),person_annotations=len(annotations),score_threshold=score,
                   foreground_iou=foreground_iou,background_iou=background_iou,
                   predictions_at_threshold=len(predictions),true_positives=true_positive,
                   false_positives=len(rows),ignored_predictions=ignored_count,
                   false_negatives=len(annotations)-true_positive,
                   precision=true_positive/(true_positive+len(rows)),
                   recall=true_positive/len(annotations),
                   true_positive_by_class=dict(Counter(a["source_class"] for _,a,ignored in matches if a is not None and not ignored)),
                   counts=dict(counts),
                   percentages={k:100*v/len(rows) for k,v in counts.items()},
                   by_split={s:dict(c) for s,c in split_counts.items()},
                   background_no_overlap=sum(r["max_person_iou"]==0 for r in rows if r["error_type"]=="background_candidate"),
                   background_weak_overlap=sum(0<r["max_person_iou"]<background_iou for r in rows),
                   sensitivity_background_iou=sensitivity,
                   definitions=dict(duplicate="이미 정탐에 사용된 사람 주석과 IoU >= 정탐 경계",
                                    localization="배경 경계 <= 같은 이미지의 모든 사람 주석과의 최대 IoU < 정탐 경계",
                                    background_candidate="모든 사람 주석과의 최대 IoU < 배경 경계. 사람이 없는 곳으로 육안 확정한 결과가 아님"),
                   source_checkpoint_sha256=manifest["checkpoint_sha256"],
                   note="TIDE 오류 분류의 IoU 경계를 참고한 주석 기반 자동 집계이며 TIDE 패키지를 실행한 결과는 아니다.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (output_dir/"false_positives.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in row.items()})
    names = dict(duplicate="중복 탐지",localization="위치 불일치 후보",background_candidate="사람 주석과 거의 겹치지 않는 배경 후보")
    report = ["전체 사람 기준 오탐 자동 분류", "", f"평가: 전체 {len(images)}장, 점수 {score} 이상, 정답 매칭 IoU {foreground_iou}", ""]
    for kind in ("duplicate","localization","background_candidate"):
        report.append("%s: %s개 (%.2f%%)" % (names[kind],format(counts[kind],","),100*counts[kind]/len(rows)))
    report += ["합계: %s개" % format(len(rows),","), "",
               f"중복: 이미 사용된 사람 주석과 IoU {foreground_iou} 이상.",
               f"위치 불일치 후보: 사람 주석과의 최대 IoU가 {background_iou} 이상 {foreground_iou} 미만.",
               f"배경 후보: 사람 주석과의 최대 IoU가 {background_iou} 미만.",
               "배경 후보에는 누락 주석이나 심하게 어긋난 박스가 포함될 수 있어 실제 비인물 영역 탐지 수로 단정할 수 없다."]
    (output_dir/"오탐_분류_요약.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 사람 기준 오탐 3종 자동 분류")
    parser.add_argument("--results",required=True)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--out",default="analysis_results/false_positive_breakdown")
    parser.add_argument("--iou",type=float,default=.5)
    args = parser.parse_args()
    print(json.dumps(audit(args.results,args.dataset,args.out,foreground_iou=args.iou),ensure_ascii=False,indent=2))
