import sys
import tempfile
import unittest
import json
import types
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sspnet_colab"))
from metrics import evaluate, load_dataset, image_path, merge_splits
from engine import positions, merge_nms, predict, run
from PIL import Image, ImageDraw
import numpy as np


def fake_torch():
    return types.SimpleNamespace(cuda=types.SimpleNamespace(
        reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 1024**2, get_device_name=lambda _: "시험용 가상 GPU"))


def ann(n,box,image=1,**kw):
    return dict(id=n,image_id=image,category_id=1,bbox=box,**kw)


def pred(box,score=.9,image=1):
    return dict(image_id=image,bbox=box,score=score)


class EvaluationTests(unittest.TestCase):
    def test_matches_duplicates_and_negative_frames(self):
        gt=dict(images=[dict(id=1),dict(id=2)],annotations=[ann(1,[0,0,8,8]),ann(2,[50,50,40,40])])
        ps=[pred([0,0,8,8]),pred([0,0,8,8],.8),pred([50,50,40,40],.7),pred([0,0,8,8],.6,image=2)]
        m=evaluate(gt,ps)
        self.assertEqual(m["true_positives"],2)
        self.assertEqual(m["false_positives"],2)
        self.assertEqual(m["tiny_le20"]["recall"],1)
        self.assertEqual(m["precision"],.5)
        self.assertEqual(m["false_positives_per_image"],1)
        self.assertGreater(m["AP50_saved_predictions_101point"],.8)

    def test_threshold_and_size_boundary(self):
        gt=dict(images=[dict(id=1)],annotations=[ann(1,[0,0,20,20]),ann(2,[50,50,21,2])])
        m=evaluate(gt,[pred([0,0,20,20],.1),pred([50,50,21,2],.9)],score_threshold=.25)
        self.assertEqual(m["tiny_le20"]["positives"],1)
        self.assertEqual(m["tiny_le20"]["hits"],0)
        self.assertEqual(m["false_positives"],0)
        self.assertAlmostEqual(m["AP50_saved_predictions_101point"],1)

    def test_ignored_and_crowd(self):
        gt=dict(images=[dict(id=1)],annotations=[ann(1,[0,0,8,8]),ann(2,[40,40,50,50],iscrowd=1)])
        m=evaluate(gt,[pred([0,0,8,8]),pred([45,45,5,5])])
        self.assertEqual(m["false_positives"],0)
        self.assertEqual(m["positives"],1)

    def test_empty_predictions_and_no_positive(self):
        self.assertIsNone(evaluate(dict(images=[dict(id=1)],annotations=[]),[])["recall"])
        m=evaluate(dict(images=[dict(id=1)],annotations=[ann(1,[0,0,4,4])]),[])
        self.assertEqual(m["recall"],0)
        self.assertEqual(m["AP50_saved_predictions_101point"],0)

    def test_target_mapping_and_image_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new("RGB",(100,60)).save(root/"a.jpg")
            raw=dict(images=[dict(id=1,file_name="a.jpg",width=100,height=60)],
                categories=[dict(id=2,name="person_water"),dict(id=3,name="person sand")],
                annotations=[ann(1,[0,0,10,10]),ann(2,[20,20,10,10])])
            raw["annotations"][0]["category_id"]=2
            raw["annotations"][1]["category_id"]=3
            path=root/"_annotations.coco.json"
            path.write_text(json.dumps(raw))
            gt,s=load_dataset(path)
            self.assertEqual(len(gt["annotations"]),1)
            self.assertEqual(gt["annotations"][0]["category_id"],1)
            self.assertEqual(s["target_count"],1)
            with self.assertRaises(ValueError):
                load_dataset(path,["swimmer"])
            with self.assertRaises(ValueError):
                image_path(path,"../outside.jpg")
            raw["images"][0]["width"]=99
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_tiles_cover_edges_and_nms(self):
        xs=positions(1500,640,.25)
        self.assertEqual(xs[0],0)
        self.assertEqual(xs[-1]+640,1500)
        self.assertTrue(all(b <= a+640 for a,b in zip(xs,xs[1:])))
        self.assertEqual(positions(200,640,.25),[0])
        with self.assertRaises(ValueError):
            positions(1500,640,1)
        boxes=merge_nms([[0,0,10,10,.8],[0,0,10,10,.9],[50,50,60,60,.7]])
        self.assertEqual(len(boxes),2)
        self.assertAlmostEqual(float(boxes[0,4]),.9,places=5)
        self.assertEqual(merge_nms([]).shape,(0,5))

    def test_bgr_tiles_return_original_coordinates(self):
        def detector(model,bgr):
            yy,xx=np.where((bgr[:,:,0]==0) & (bgr[:,:,1]==0) & (bgr[:,:,2]==255))
            result=[[xx.min(),yy.min(),xx.max()+1,yy.max()+1,.9]] if len(xx) else []
            return [np.asarray(result,dtype=np.float32).reshape(-1,5)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"red.png"
            picture=Image.new("RGB",(1000,600))
            ImageDraw.Draw(picture).rectangle((610,100,620,110),fill="red")
            picture.save(path)
            settings=dict(mode="tiles",tile_size=640,tile_overlap=.25,merge_iou=.5,max_detections=500)
            with patch.dict(sys.modules,{"torch":fake_torch(),"mmdet.apis":types.SimpleNamespace(inference_detector=detector)}):
                record=predict(None,path,settings)
            self.assertEqual(record["patches"],2)
            self.assertEqual(len(record["boxes"]),1)
            self.assertEqual(record["boxes"][0][:4],[610.,100.,621.,111.])

    def test_resume_reuses_only_matching_images_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/"a.png"
            Image.new("RGB",(30,30)).save(image)
            annotations=root/"labels.json"
            annotations.write_text(json.dumps(dict(
                images=[dict(id=1,file_name="a.png",width=30,height=30)],
                categories=[dict(id=1,name="person water")],annotations=[ann(1,[0,0,8,8])])),encoding="utf-8")
            checkpoint=root/"weights.pth"
            checkpoint.write_bytes("가중치 파일 대신 해시만 검사".encode("utf-8"))
            settings=root/"settings.json"
            config=dict(mode="full",save_min_score=.001,report_score=.25,checkpoint=str(checkpoint),
                        repo=str(root),target_names=["person water"],long_edge=1280,short_edge=1024,max_detections=500)
            settings.write_text(json.dumps(config))
            args=types.SimpleNamespace(settings=str(settings),out=str(root/"out"),image=None,
                                       annotations=str(annotations),limit=0)
            result=dict(boxes=[[0,0,8,8,.9]],seconds=.1,patches=1,peak_vram_mib=1)
            fake_cfg=types.SimpleNamespace(dump=lambda path: Path(path).write_text("# 시험 설정"))
            with patch.dict(sys.modules,{"torch":fake_torch()}), patch("engine.subprocess.check_output",return_value="abc\n"), \
                 patch("engine.make_model",return_value=(None,fake_cfg)), patch("engine.predict",side_effect=lambda *a: dict(result)) as inference, \
                 patch("builtins.print"):
                run(args)
                self.assertEqual(inference.call_count,2)
                run(args)
                self.assertEqual(inference.call_count,2)
                config["report_score"]=.95
                settings.write_text(json.dumps(config))
                run(args)
                self.assertEqual(inference.call_count,2)
                metric=json.loads((root/"out/metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metric["true_positives"],0)
                Image.new("RGB",(30,30),"blue").save(image)
                run(args)
                self.assertEqual(inference.call_count,4)
                config["save_min_score"]=.01
                settings.write_text(json.dumps(config))
                run(args)
                self.assertEqual(inference.call_count,6)

    def test_geometric_ranges_and_matching_outside_range(self):
        gt=dict(images=[dict(id=1)],annotations=[
            ann(1,[0,0,2,2]), ann(2,[40,0,8,8]), ann(3,[80,0,20,20]),
            ann(4,[0,40,1,1]), ann(5,[40,40,30,10])])
        ps=[pred(a["bbox"],.9-i*.1) for i,a in enumerate(gt["annotations"])]
        m=evaluate(gt,ps,.65)
        self.assertEqual(m["geometric_2to20"]["positives"],3)
        self.assertEqual(m["geometric_2to20"]["hits"],2)
        self.assertEqual(m["geometric_2to8"],dict(positives=1,hits=1,recall=1))
        self.assertEqual(m["false_positives"],0)
        self.assertEqual(m["tiny_le20"]["positives"],4)

    def test_merge_renumbers_ids_preserves_names_and_maps_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sources=[]
            for split,cid,name in [("train",7,"person water"),("valid",3,"person_water"),("test",1,"person water")]:
                folder=root/split
                folder.mkdir()
                Image.new("RGB",(30,30)).save(folder/"same.png")
                raw=dict(images=[dict(id=0,file_name="same.png",width=30,height=30)],
                         categories=[dict(id=cid,name=name)],annotations=[ann(0,[0,0,4,4],image=0)])
                raw["annotations"][0]["category_id"]=cid
                path=folder/"_annotations.coco.json"
                path.write_text(json.dumps(raw))
                sources.append((split,path))
            merged=merge_splits(sources,root/"all_annotations.coco.json")
            gt,summary=load_dataset(merged)
            self.assertEqual(summary["image_count"],3)
            self.assertEqual(summary["target_count"],3)
            self.assertEqual(summary["geometric_2to20_count"],3)
            self.assertEqual(len({im["file_name"] for im in gt["images"]}),3)
            self.assertEqual([im["source_image_id"] for im in gt["images"]],[0,0,0])
            self.assertEqual([a["id"] for a in gt["annotations"]],[1,2,3])
            self.assertEqual(set(summary["split_counts"]),{"train","valid","test"})
            with self.assertRaises(ValueError):
                merge_splits([sources[0],sources[0]],root/"bad.json")


if __name__ == "__main__":
    unittest.main()
