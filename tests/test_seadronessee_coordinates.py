"""320 타일의 양축 오프셋·끝단 보정·중복 병합을 원본 표식으로 검증한다."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sspnet_colab'))
from engine import predict

class OriginalCoordinateTests(unittest.TestCase):
    def test_landscape_and_portrait_320_tiles(self):
        fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(
            reset_peak_memory_stats=lambda:None,synchronize=lambda:None,max_memory_allocated=lambda:0))
        colors=[(255,0,0),(0,255,0),(0,0,255)]
        for size,expected in [((1001,701),[(270,270,278,278),(600,500,608,508),(990,690,998,698)]),
                              ((701,1001),[(270,270,278,278),(500,600,508,608),(690,990,698,998)])]:
            with self.subTest(size=size),tempfile.TemporaryDirectory() as tmp:
                image=Image.new('RGB',size)
                pen=ImageDraw.Draw(image)
                for box,color in zip(expected,colors):
                    x,y,r,b=box; pen.rectangle((x,y,r-1,b-1),fill=color)
                path=Path(tmp)/'coordinate_markers.png'; image.save(path)
                def detect(model,bgr):
                    boxes=[]
                    for rgb in colors:
                        yy,xx=np.where((bgr==np.array(rgb[::-1])).all(axis=2))
                        if len(xx): boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1,.9])
                    return [np.asarray(boxes,dtype=np.float32).reshape(-1,5)]
                with patch.dict(sys.modules,{'torch':fake_torch,'mmdet.apis':types.SimpleNamespace(inference_detector=detect)}):
                    result=predict(None,path,dict(mode='tiles',tile_size=320,tile_overlap=.2,merge_iou=.5,max_detections=500))
                self.assertEqual(result['patches'],12)
                self.assertEqual(sorted(tuple(v[:4]) for v in result['boxes']),sorted(expected))

if __name__=='__main__': unittest.main()
