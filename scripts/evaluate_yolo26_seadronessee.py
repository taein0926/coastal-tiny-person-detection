"""SeaDronesSee OD v2 validation: 범용 YOLO26m 전체 이미지 평가."""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE = float(os.environ.get('YOLO_SCORE', '0.25'))
assert SCORE in (0.05, 0.25)
OUT = ROOT / ('analysis_results/yolo26_seadronessee_1280_gpu_005' if SCORE == 0.05 else 'analysis_results/yolo26_seadronessee_1280_gpu')
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(OUT / 'config'))
sys.path.insert(0, str(ROOT / 'sspnet_colab'))
from metrics import evaluate
import onnxruntime as ort
real_session = ort.InferenceSession
def gpu_session(path, sess_options=None, providers=None, **kwargs):
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = real_session(path, opts, providers=[('DmlExecutionProvider', {'device_id': 1}), 'CPUExecutionProvider'])
    assert session.get_providers()[0] == 'DmlExecutionProvider'
    return session
ort.InferenceSession = gpu_session
from ultralytics import YOLO, __version__
import torch

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

data = ROOT / 'data/seadronessee_odv2'
gtfile = ROOT / 'analysis_results/seadronessee_validation/ground_truth_swimmer.json'
weights = ROOT / 'analysis_results/yolo26_examples/yolo26m.pt'
onnx = ROOT / 'analysis_results/yolo26_examples/yolo26m.onnx'
manifestfile = data / 'manifest.json'
gt = json.loads(gtfile.read_text(encoding='utf-8'))
manifest = {x['name']: x for x in json.loads(manifestfile.read_text(encoding='utf-8'))}
assert len(gt['images']) == len(manifest) == 1547 and len(gt['annotations']) == 6206
assert sha(data/'annotations/instances_val.json') == '9d64e077cc8c5d8c53e8f106ec292123300a6d93fed7f1c5c1464b600f9ce6a9'
assert sha(manifestfile) == '253034e5e4c23d25f8476739ece6e4c48d1714412d32f2a2e24f79b5c74f46dd'
assert sha(weights) == '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'
identity = dict(weights_sha256=sha(weights), manifest_sha256=sha(manifestfile), ground_truth_sha256=sha(gtfile),
                ultralytics=__version__, torch=torch.__version__, onnx_sha256=sha(onnx), onnxruntime=ort.__version__, device='DirectML adapter 1: NVIDIA GeForce GTX 1660 Ti with Max-Q Design', imgsz=1280, conf=SCORE,
                classes=[0], max_det=500, match_iou=0.3, mode='full', augment=False)
identity_path = OUT/'identity.json'
if identity_path.exists(): assert json.loads(identity_path.read_text(encoding='utf-8')) == identity
else: write(identity_path, identity)
cache=OUT/'cache'; cache.mkdir(exist_ok=True)
model=YOLO(str(onnx), task='detect')
images=sorted(gt['images'], key=lambda x:x['id'])
started=time.monotonic()
for index, im in enumerate(images, 1):
    path=data/'images/val'/im['file_name']; saved=cache/(str(im['id'])+'.json')
    digest=sha(path); assert digest == manifest[im['file_name']]['sha256']
    if saved.exists():
        rec=json.loads(saved.read_text(encoding='utf-8')); assert rec['sha256']==digest
    else:
        t=time.monotonic()
        result=model.predict(str(path), imgsz=1280, conf=SCORE, classes=[0], device='cpu', max_det=500, augment=False, verbose=False)[0]
        assert result.names[0] == 'person'
        assert tuple(result.orig_shape)==(im['height'],im['width'])
        boxes=result.boxes.data.tolist()
        for x1,y1,x2,y2,score,cls in boxes:
            assert 0<=x1<x2<=im['width'] and 0<=y1<y2<=im['height'] and score>=SCORE and cls==0
        write(saved, dict(image_id=im['id'],sha256=digest,boxes=boxes,seconds=time.monotonic()-t))
    if index%25==0 or index==len(images):
        write(OUT/'status.json',dict(completed=index,total=len(images),seconds=time.monotonic()-started))
        print(f'{index}/{len(images)} elapsed={time.monotonic()-started:.0f}s',flush=True)
preds=[]
for im in images:
    rec=json.loads((cache/(str(im['id'])+'.json')).read_text(encoding='utf-8'))
    for x1,y1,x2,y2,score,cls in rec['boxes']:
        preds.append(dict(image_id=im['id'],category_id=1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
metrics=evaluate(gt,preds,SCORE,.3)
metrics['false_negatives']=metrics['positives']-metrics['true_positives']
write(OUT/'predictions.json',preds)
write(OUT/'metrics.json',metrics)
print(json.dumps(metrics,ensure_ascii=False),flush=True)
