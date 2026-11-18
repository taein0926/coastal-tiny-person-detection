"""이전 전체 데이터 학습 노트북 생성기 보존본. 새 실행에는 build_training_notebook.py를 사용한다."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source, form=False):
    source = source.strip() + '\n'
    cell = dict(cell_type=kind, metadata={'cellView':'form'} if form else {},
                id=hashlib.sha256(source.encode()).hexdigest()[:12], source=source.splitlines(keepends=True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''# SSPNet 추가 학습 · 무료 Colab

YoloBeachTrain v2의 train·valid·test 전체 2,961장을 학습에 사용하고 `person water`와 `person sand`를 `person`으로 합칩니다. Piriapolis는 학습과 가중치 선택에 사용하지 않습니다.

무료 GPU를 선택하고 1→2→3→4→5 순서로 실행하세요. Drive 인증은 직접 완료합니다. 먼저 시험 학습의 시간과 메모리를 확인하세요. 본 학습 횟수는 Piriapolis 결과를 보기 전에 고정하며 최종 epoch 가중치를 평가에 사용합니다. 기본 5 epoch는 시작 제안이며 성능을 보장하지 않습니다.

중단되면 같은 설정·실행 이름으로 1→2→3을 실행하고 5단계의 `RESUME`을 켭니다. epoch가 끝날 때마다 optimizer를 포함한 체크포인트를 Drive에 저장하므로 마지막으로 완료한 epoch부터 재개합니다. 시험 학습은 본 학습 가중치에 반영되지 않습니다. 데이터에 이미 포함된 증강 외에 추가 증강은 적용하지 않습니다.''')
add('code', '''#@title 1. 설정
import os, sys, json, shutil, subprocess
from pathlib import Path
EPOCHS = 5 #@param {type:"integer"}
SMOKE_STEPS = 50 #@param {type:"integer"}
RUN_NAME = "beach_v2_5epochs" #@param {type:"string"}
DATASET_ZIP_PATH = "" #@param {type:"string"}
if EPOCHS < 1 or SMOKE_STEPS < 1 or not RUN_NAME or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_NAME):
    raise ValueError('횟수와 영문·숫자·밑줄·하이픈으로 된 실행 이름을 확인하세요.')
WORK=Path('/content/sspnet_free')
WORK.mkdir(exist_ok=True)
print('설정:', RUN_NAME, EPOCHS, 'epoch / 시험', SMOKE_STEPS, '회')''', True)
helpers = {n:(ROOT/'sspnet_colab'/('train_legacy.py' if n == 'train.py' else n)).read_text(encoding='utf-8') for n in ('setup_environment.py','metrics.py','workflow.py','summarize_scopes.py','train.py')}
add('code', '#@title 2. 환경·공개 가중치 준비\nHELPER_SOURCES = '+repr(helpers)+'''\nfrom google.colab import drive
drive.mount('/content/drive')
STORE=Path('/content/drive/MyDrive/SSPNet_free_colab')
STORE.mkdir(parents=True,exist_ok=True)
HELPERS=WORK/'helpers'
HELPERS.mkdir(exist_ok=True)
for name,source in HELPER_SOURCES.items(): (HELPERS/name).write_text(source,encoding='utf-8')
subprocess.run([sys.executable,str(HELPERS/'setup_environment.py')],check=True)
PYTHON=WORK/'venv/bin/python'
REPO=WORK/'SSPNet'
ENV=os.environ.copy()
ENV.pop('PYTHONHOME',None)
ENV.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(REPO)+os.pathsep+str(HELPERS),MPLBACKEND='Agg')
def checked(args):
    process=subprocess.Popen([str(x) for x in args],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout: print(line,end='',flush=True)
    code=process.wait()
    if code: raise subprocess.CalledProcessError(code,args)
CHECKPOINT=STORE/'weights/faster_rcnn_sspnet.pth'
CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
if not CHECKPOINT.is_file():
    checked([PYTHON,'-c',"import gdown; p=gdown.download(id='1IfPCt5xZqqBJ3sYVIuD5F9l29Jcy2Hn1',output=%r,quiet=False); assert p"%str(CHECKPOINT)])
import hashlib
def filehash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
if filehash(CHECKPOINT) != '912cc5b6f8a45b0f4521c9bf325868ed10dd4757497fb6ba8843fdbd26b09e4a':
    raise ValueError('공개 가중치 해시가 기준과 다릅니다. 원본 가중치를 확인하세요.')
print('환경과 공개 가중치 준비 완료')''', True)
add('code', '''#@title 3. 학습 데이터 준비
DATA_ZIP=Path(DATASET_ZIP_PATH) if DATASET_ZIP_PATH else WORK/'YoloBeachTrain_v2.zip'
if not DATA_ZIP.is_file():
    from google.colab import files
    print('YoloBeachTrain.v2i.coco.zip을 선택하세요.')
    uploaded=files.upload()
    if len(uploaded)!=1: raise ValueError('ZIP 한 개를 선택하세요.')
    DATA_ZIP.write_bytes(next(iter(uploaded.values())))
CONFIG=dict(task='dataset',dataset_name='YoloBeachTrain_v2',split='all')
config_path=WORK/'training_data_config.json'
config_path.write_text(json.dumps(CONFIG),encoding='utf-8')
checked([PYTHON,HELPERS/'workflow.py','prepare','--config',config_path,'--zip',DATA_ZIP,'--store',STORE])
prepared=json.loads((WORK/'prepared_input.json').read_text(encoding='utf-8'))
ANNOTATIONS=Path(prepared['annotations'])
print('학습 데이터 준비:',prepared['images'],'장')''', True)
add('markdown', '''## 4. 시험 학습

50회 업데이트가 기본입니다. 오류·소요 시간·GPU 메모리를 확인합니다. 시험 결과로 Piriapolis를 평가하지 않습니다. 다시 시험하려면 실행 이름을 바꾸세요.''')
add('code', '''#@title 4. 시험 학습
SMOKE_DIR=STORE/'training'/RUN_NAME/'smoke'
checked([PYTHON,HELPERS/'train.py','--repo',REPO,'--annotations',ANNOTATIONS,'--checkpoint',CHECKPOINT,
         '--output',SMOKE_DIR,'--mode','smoke','--epochs',EPOCHS,'--smoke-steps',SMOKE_STEPS])
print('시험 학습 완료. 본 학습 횟수를 확정한 뒤 5단계를 실행하세요.')''', True)
add('markdown', '''## 5. 본 학습과 최종 가중치

시험 학습 후 횟수를 확정하고 실행합니다. 재개할 때는 설정을 바꾸지 마세요. 본 학습이 끝나면 표시되는 `epoch_N.pth` 경로를 평가 노트북의 `trained_path`에 입력하고 `full`을 선택합니다. Drive가 연결되어 있어야 합니다.''')
add('code', '''#@title 5. 본 학습
RESUME = False #@param {type:"boolean"}
if not (STORE/'training'/RUN_NAME/'smoke'/('iter_%d.pth'%SMOKE_STEPS)).is_file():
    raise RuntimeError('같은 실행 이름과 설정으로 4단계 시험 학습을 먼저 완료하세요.')
TRAIN_DIR=STORE/'training'/RUN_NAME/'main'
args=[PYTHON,HELPERS/'train.py','--repo',REPO,'--annotations',ANNOTATIONS,'--checkpoint',CHECKPOINT,
      '--output',TRAIN_DIR,'--mode','train','--epochs',EPOCHS,'--smoke-steps',SMOKE_STEPS]
if RESUME: args.append('--resume')
checked(args)
print('평가 노트북에 입력할 최종 가중치 경로:',TRAIN_DIR/('epoch_%d.pth'%EPOCHS))''', True)
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(colab=dict(name='SSPNet_해변_5epoch_학습_보존본.ipynb',provenance=[]),accelerator='GPU',
                        kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3')))
path=ROOT/'notebooks/SSPNet_해변_5epoch_학습_보존본.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print(path)
