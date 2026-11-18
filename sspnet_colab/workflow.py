"""Colab 설정·데이터 준비·실행·비교·다운로드를 위한 공통 처리."""
import csv
import argparse
import hashlib
import json
import shutil
import subprocess
import os
import zipfile
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def prepare_dataset(config,work,zip_path):
    from metrics import merge_splits,load_dataset
    work=Path(work)
    signature=digest(zip_path)
    root=work/"datasets"/config["dataset_name"]/signature[:12]
    stamp=root/"prepared.json"
    reused=stamp.is_file() and json.loads(stamp.read_text()).get("zip_sha256")==signature
    if not reused:
        root.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            if sum(m.file_size for m in archive.infolist())+1024**3>shutil.disk_usage(work).free:
                raise RuntimeError("데이터 압축 해제에 필요한 공간이 부족합니다.")
            for member in archive.infolist():
                resolved=(root/member.filename).resolve()
                if root.resolve()!=resolved and root.resolve() not in resolved.parents:
                    raise ValueError("ZIP에 데이터 폴더 밖을 가리키는 경로가 있습니다.")
                if (member.external_attr>>16)&0o170000==0o120000:
                    raise ValueError("ZIP의 심볼릭 링크는 지원하지 않습니다.")
            archive.extractall(root)
        dump(stamp,dict(zip_sha256=signature))
    split=config["split"]
    parts=("train","valid","test") if split=="all" else (split,)
    sources=[]
    for part in parts:
        paths=[p for p in root.rglob("_annotations.coco.json") if p.parent.name==part]
        if len(paths)!=1:
            raise ValueError(part+" 분할의 정답 JSON을 한 개 찾을 수 없습니다.")
        sources.append((part,str(paths[0])))
    annotations=root/("merged_%s.coco.json"%split)
    merge_splits(sources,annotations)
    gt,summary=load_dataset(annotations,["person water","person sand"])
    water,water_summary=load_dataset(annotations,["person water"])
    if config["dataset_name"]=="AnnotationPiriapolis_v4" and split=="all":
        if (summary["image_count"],summary["target_count"],water_summary["target_count"])!=(635,8099,2904):
            raise ValueError("AnnotationPiriapolis v4의 전체 635장·전체 사람 8099개·물속 사람 2904개와 다릅니다.")
    state=dict(task="dataset",dataset_name=config["dataset_name"],split=split,
               annotations=str(annotations),zip_sha256=signature,
               images=summary["image_count"],all_people=summary["target_count"],water_only=water_summary["target_count"])
    dump(work/"prepared_input.json",state)
    print("데이터 준비 완료: %d장 / 전체 사람 %d개 / 물속 사람 %d개"%(state["images"],state["all_people"],state["water_only"]),flush=True)
    print("기존 압축 해제 파일을 재사용했습니다." if reused else "데이터 압축을 해제했습니다.",flush=True)
    return state


def import_baseline(zip_path,destination):
    destination=Path(destination)
    needed=("predictions_coco.json","ground_truth_water.json","run_manifest.json")
    with zipfile.ZipFile(zip_path) as archive:
        candidates=[n for n in archive.namelist() if n.endswith("/full/predictions_coco.json")]
        if len(candidates)!=1:
            raise ValueError("기존 전체 검사 결과 ZIP의 full/predictions_coco.json을 한 개 찾을 수 없습니다.")
        prefix=candidates[0].rsplit("/",1)[0]+"/"
        payload={name:archive.read(prefix+name) for name in needed}
        manifest=json.loads(payload["run_manifest.json"])
        if manifest["mode"]!="full":
            raise ValueError("비교용 ZIP은 전체 검사(full) 결과여야 합니다.")
    for name,data in payload.items():
        json.loads(data)
    destination.mkdir(parents=True,exist_ok=True)
    for name,data in payload.items():
        (destination/name).write_bytes(data)
    return destination


def align_baseline(baseline,annotations,checkpoint,settings,destination):
    from metrics import load_dataset
    baseline,destination=Path(baseline),Path(destination)
    manifest=json.loads((baseline/"run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256")!=digest(checkpoint):
        raise ValueError("기존 전체 검사와 현재 가중치가 달라 같은 모델의 분할 효과를 비교할 수 없습니다.")
    for key in ("long_edge","short_edge","max_detections","save_min_score"):
        if manifest.get(key)!=settings.get(key):
            raise ValueError("기존 전체 검사와 설정이 다릅니다: "+key)
    gt,_=load_dataset(annotations,["person water"])
    old=json.loads((baseline/"ground_truth_water.json").read_text(encoding="utf-8"))
    def key(im):
        return (im.get("source_split",""),im.get("source_image_id",im["id"]),Path(im["file_name"]).name)
    old_map={key(im):im["id"] for im in old["images"]}
    if any(key(im) not in old_map for im in gt["images"]):
        raise ValueError("기존 전체 검사 결과에 현재 데이터의 이미지가 모두 포함돼 있지 않습니다.")
    remap={old_map[key(im)]:im["id"] for im in gt["images"]}
    # 이름이 같은 파일의 주석이 변경된 비교를 막는다.
    def boxes(raw,ids=None):
        values=[]
        for a in raw["annotations"]:
            image_id=a["image_id"]
            if ids is not None:
                if image_id not in ids:continue
                image_id=ids[image_id]
            values.append((image_id,tuple(a["bbox"]),bool(a.get("ignore")),bool(a.get("iscrowd"))))
        return sorted(values)
    if boxes(old,remap)!=boxes(gt):
        raise ValueError("기존 전체 검사와 현재 물속 사람 주석이 달라 비교를 중단했습니다.")
    predictions=json.loads((baseline/"predictions_coco.json").read_text(encoding="utf-8"))
    predictions=[dict(p,image_id=remap[p["image_id"]]) for p in predictions if p["image_id"] in remap]
    dump(destination/"predictions_coco.json",predictions)
    dump(destination/"ground_truth_water.json",gt)
    dump(destination/"run_manifest.json",dict(manifest,comparison_image_ids=list(remap.values())))
    return destination


def cached_predictions_valid(out,settings,annotations,limit):
    from metrics import load_dataset,image_path
    out=Path(out)
    try:
        old=json.loads((out/"run_manifest.json").read_text(encoding="utf-8"))
        status=json.loads((out/"run_status.json").read_text(encoding="utf-8"))
        if status["state"]!="완료" or not (out/"predictions_coco.json").is_file():return False
        keys=("mode","long_edge","short_edge","tile_size","tile_overlap","merge_iou","save_min_score","max_detections")
        if any(old.get(k)!=settings.get(k) for k in keys):return False
        if old["checkpoint_sha256"]!=digest(settings["checkpoint"]) or old["annotations_sha256"]!=digest(annotations):return False
        if old.get("engine_sha256")!=digest(Path(settings["repo"]).parent/"helpers/engine.py"):return False
        commit=subprocess.check_output(["git","-C",settings["repo"],"rev-parse","HEAD"],text=True).strip()
        if old.get("source_commit")!=commit:return False
        gt,_=load_dataset(annotations,["person water"],limit)
        if old["image_ids"]!=[im["id"] for im in gt["images"]]:return False
        cache=out/("cache_"+status["cache_key"])
        return all(json.loads((cache/(str(im["id"])+".json")).read_text(encoding="utf-8"))["image_sha256"]==digest(image_path(annotations,im["file_name"])) for im in gt["images"])
    except (OSError,ValueError,KeyError):
        return False


def run_dataset(config,work,store,checkpoint,env):
    from summarize_scopes import summarize_scopes
    work,store=Path(work),Path(store)
    state=json.loads((work/"prepared_input.json").read_text(encoding="utf-8"))
    if any(state.get(k)!=config[k] for k in ("task","dataset_name","split")):
        raise ValueError("데이터 선택이 변경됐습니다. 3단계부터 다시 실행하세요.")
    mode=config["mode"]
    label="full" if mode=="full" else "tiles_%d_overlap%g"%(config["tile_size"],config["tile_overlap"])
    out=store/"evaluation"/config["dataset_name"]/config["split"]/label
    if config.get("weight_source", "public") != "public":
        out=store/"evaluation_models"/digest(checkpoint)[:16]/config["dataset_name"]/config["split"]/label
    settings=dict(repo=str(work/"SSPNet"),checkpoint=str(checkpoint),mode=mode,
                  long_edge=config["long_edge"],short_edge=config["short_edge"],tile_size=config["tile_size"],
                  tile_overlap=config["tile_overlap"],merge_iou=.5,report_score=config["report_score"],
                  save_min_score=.001,max_detections=500,target_names=["person water"],
                  dataset_name=config["dataset_name"],split=config["split"],match_iou=config["match_iou"])
    dump(work/"evaluation_settings.json",settings)
    python=work/"venv/bin/python"
    if cached_predictions_valid(out,settings,state["annotations"],config["max_images"]):
        print("동일한 추론 결과를 재사용하고 점수만 다시 집계합니다.",flush=True)
    else:
        subprocess.run([str(python),str(work/"helpers/engine.py"),"--settings",str(work/"evaluation_settings.json"),
                        "--annotations",state["annotations"],"--limit",str(config["max_images"]),"--out",str(out)],env=env,check=True)
    summarize_scopes(state["annotations"],out,config["report_score"],config["match_iou"])
    outputs=[out]
    baseline=Path(config.get("baseline_path") or store/"baselines"/config["dataset_name"])
    if config.get("compare_with_full") and mode!="full":
        if not (baseline/"predictions_coco.json").is_file():
            baseline=store/"evaluation"/config["dataset_name"]/config["split"]/"full"
        if not (baseline/"predictions_coco.json").is_file():
            raise ValueError("기존 전체 검사 결과가 없습니다. 3단계에서 비교용 ZIP을 준비하세요.")
        destination=out/"baseline_full_rescored"
        align_baseline(baseline,state["annotations"],checkpoint,settings,destination)
        # 부분 평가에서는 양쪽 모두 현재 처리한 이미지로만 비교한다.
        current=json.loads((out/"ground_truth_water.json").read_text(encoding="utf-8"))
        aligned=json.loads((destination/"ground_truth_water.json").read_text(encoding="utf-8"))
        ids={im["id"] for im in current["images"]}
        aligned["images"]=[im for im in aligned["images"] if im["id"] in ids]
        aligned["annotations"]=[a for a in aligned["annotations"] if a["image_id"] in ids]
        dump(destination/"ground_truth_water.json",aligned)
        summarize_scopes(state["annotations"],destination,config["report_score"],config["match_iou"])
        rows=[]
        for scope in ("all_people","water_only"):
            for method,folder in (("전체 검사",destination),("분할 검사",out)):
                m=json.loads((folder/("metrics_"+scope+".json")).read_text(encoding="utf-8"))
                rows.append(dict(평가범위=m["scope_label"],방식=method,이미지=m["images"],정답=m["positives"],정탐=m["true_positives"],
                                 재현율=m["recall"],정밀도=m["precision"],오탐=m["false_positives"],이미지당오탐=m["false_positives_per_image"],
                                 작은사람재현율=m["geometric_2to20"]["recall"],최소구간재현율=m["geometric_2to8"]["recall"]))
        dump(out/"전체검사_분할검사_비교.json",rows)
        with (out/"전체검사_분할검사_비교.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print("전체 검사·분할 검사 비교표를 같은 점수·IoU·이미지·정답 범위로 저장했습니다.",flush=True)
    dump(work/"last_output.json",dict(paths=[str(p) for p in outputs],task="dataset",label=label,config=config))
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="SSPNet 통합 평가 처리")
    parser.add_argument("action",choices=["prepare","baseline","run"])
    parser.add_argument("--config",required=True)
    parser.add_argument("--zip")
    parser.add_argument("--work",default="/content/sspnet_free")
    parser.add_argument("--store",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.action=="prepare":
        prepare_dataset(config,args.work,args.zip)
    elif args.action=="baseline":
        import_baseline(args.zip,Path(args.store)/"baselines"/config["dataset_name"])
        print("기존 전체 검사 결과를 준비했습니다. 다시 GPU로 추론하지 않습니다.",flush=True)
    else:
        run_dataset(config,args.work,args.store,Path(config.get("checkpoint") or Path(args.work)/"faster_rcnn_sspnet.pth"),os.environ.copy())
