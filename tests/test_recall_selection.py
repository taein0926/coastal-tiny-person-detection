import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sspnet_colab'))
from recall_selection import digest,select_operating_point,selection_key,verify_bundle


def truth():
    return dict(images=[dict(id=1)],annotations=[dict(id=1,image_id=1,bbox=[0,0,8,8],water=True),
                                                dict(id=2,image_id=1,bbox=[50,50,40,40],water=True)])


def pred(box,score):return dict(image_id=1,bbox=box,score=score)


class RecallSelectionTests(unittest.TestCase):
    def test_small_recall_has_priority_and_budget_limits_threshold(self):
        ps=[pred([50,50,40,40],.95),pred([100,100,8,8],.6),pred([0,0,8,8],.1)]
        strict=select_operating_point(truth(),ps,fp_per_image=0)
        relaxed=select_operating_point(truth(),ps,fp_per_image=1)
        self.assertEqual(strict['best']['small_swimmer_hits'],0)
        self.assertEqual(relaxed['best']['small_swimmer_hits'],1)
        self.assertEqual(relaxed['best']['FP'],1)
        self.assertGreater(selection_key(relaxed['best']),selection_key(strict['best']))

    def test_duplicate_prediction_is_false_positive(self):
        result=select_operating_point(truth(),[pred([0,0,8,8],.8),pred([0,0,8,8],.7)],fp_per_image=1)
        self.assertEqual(result['best']['small_swimmer_hits'],1)
        # 같은 재현율이면 중복 박스 없는 높은 임계값을 선택한다.
        self.assertEqual(result['best']['FP'],0)

    def test_missing_small_truth_is_rejected(self):
        gt=truth();gt['annotations']=gt['annotations'][1:]
        with self.assertRaises(ValueError):select_operating_point(gt,[])

    def test_land_person_does_not_count_as_small_swimmer(self):
        gt=truth();gt['annotations'].append(dict(id=3,image_id=1,bbox=[100,100,8,8],water=False))
        result=select_operating_point(gt,[pred([100,100,8,8],.9)],fp_per_image=0)
        self.assertEqual(result['best']['small_swimmer_positives'],1)
        self.assertEqual(result['best']['small_swimmer_hits'],0)
        self.assertEqual(result['best']['TP'],1)

    def test_unknown_water_remains_person_but_not_swimmer(self):
        gt=truth();gt['annotations'].append(dict(id=3,image_id=1,bbox=[100,100,8,8],water=None))
        point=select_operating_point(gt,[pred([100,100,8,8],.9)],fp_per_image=0)['best']
        self.assertEqual((point['TP'],point['small_swimmer_positives'],point['small_swimmer_hits']),(1,1,0))

    def test_only_version_three_allows_explicit_unknown_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.make_bundle(root)
            raw=json.loads((root/'train.json').read_text());raw['annotations'][0]['water']=None
            (root/'train.json').write_text(json.dumps(raw));manifest['splits']['train']['annotation_sha256']=digest(root/'train.json')
            manifest['version']=2;(root/'quality_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_bundle(root)
            manifest['version']=3;(root/'quality_manifest.json').write_text(json.dumps(manifest))
            verify_bundle(root)

    def make_bundle(self,root):
        splits={}
        for s in ('train','val'):
            image=root/(s+'.bin');image.write_bytes(s.encode())
            raw=dict(images=[dict(id=1,file_name=image.name,width=32,height=32,capture_group=s)],
                     annotations=[dict(id=1,image_id=1,category_id=1,bbox=[1,1,8,8],water=True)])
            path=root/(s+'.json');path.write_text(json.dumps(raw))
            splits[s]=dict(annotation_file=path.name,annotation_sha256=digest(path),groups=[s],
                           images=[dict(image_id=1,file_name=image.name,sha256=digest(image),group=s,review=dict(status='accepted'))])
        manifest=dict(splits=splits,reserved_groups=['reserved'])
        (root/'quality_manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_quality_gate_and_image_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_bundle(root)
            verify_bundle(root)
            (root/'train.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_bundle(root)

    def test_unreviewed_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.make_bundle(root)
            manifest['splits']['train']['images'][0]['review']['status']='pending'
            (root/'quality_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_bundle(root)

    def test_group_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=self.make_bundle(root)
            raw=json.loads((root/'val.json').read_text());raw['images'][0]['capture_group']='train'
            (root/'val.json').write_text(json.dumps(raw))
            manifest['splits']['val']['annotation_sha256']=digest(root/'val.json')
            manifest['splits']['val']['groups']=['train'];manifest['splits']['val']['images'][0]['group']='train'
            (root/'quality_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_bundle(root)


if __name__=='__main__':unittest.main()
