"""승인된 데이터 묶음을 사용하는 재현율 우선 Colab 노트북을 생성한다."""
import hashlib
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--version',choices=('recall_v1','recall_v2'),default='recall_v1')
args=p.parse_args()
cells=[]


def add(kind,source):
    source=source.strip()+'\n'
    cell=dict(cell_type=kind,metadata={},id=hashlib.sha256(source.encode()).hexdigest()[:12],source=source.splitlines(keepends=True))
    if kind=='code':cell.update(execution_count=None,outputs=[])
    cells.append(cell)


add('markdown','''# SSPNet 작은 수영자 재현율 우선 추가 학습

프로젝트에서 검토·승인한 `recall_v1.zip`을 사용합니다. YoloBeachTrain 전체 ZIP은 사용하지 않습니다.
원본(epoch 0)과 매 epoch를 검증합니다. 이미지당 FP 예산 안에서 원본 좌표 √면적 2~20픽셀 수영자의 재현율을 최우선으로 선택합니다.
동점이면 전체 재현율, 낮은 FP 순서입니다. 학습이 원본보다 나쁘면 원본을 유지합니다.
현재 데이터는 소규모 초기 실험 묶음입니다. 일반화 성능을 보장하지 않으며 추가 이미지는 검토 후 별도 버전으로 준비합니다.
무료 GPU를 선택하고 순서대로 실행하세요. Drive 인증은 직접 완료합니다.''')
add('code','''import os,sys,json,subprocess,zipfile
from pathlib import Path
from google.colab import drive,files
drive.mount('/content/drive')
RUN_NAME='recall_v1_3epochs'
EPOCHS=3
LEARNING_RATE=0.00001
FP_PER_IMAGE=4.0
DATA_ZIP_PATH='' # Drive의 recall_v1.zip 경로. 비우면 업로드합니다.
WORK=Path('/content/sspnet_free');WORK.mkdir(exist_ok=True)
STORE=Path('/content/drive/MyDrive/SSPNet_free_colab');STORE.mkdir(parents=True,exist_ok=True)
if not RUN_NAME or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_NAME):
    raise ValueError('실행 이름은 영문·숫자·밑줄·하이픈으로 지정하세요.')''')
helpers={n:(ROOT/'sspnet_colab'/n).read_text(encoding='utf-8') for n in ('setup_environment.py','metrics.py','recall_selection.py','train.py')}
add('code','HELPER_SOURCES = '+repr(helpers)+'''
HELPERS=WORK/'helpers';HELPERS.mkdir(exist_ok=True)
for name,source in HELPER_SOURCES.items():(HELPERS/name).write_text(source,encoding='utf-8')
subprocess.run([sys.executable,str(HELPERS/'setup_environment.py')],check=True)
PYTHON=WORK/'venv/bin/python';REPO=WORK/'SSPNet'
ENV=os.environ.copy();ENV.pop('PYTHONHOME',None)
ENV.update(PYTHONNOUSERSITE='1',PYTHONPATH=str(HELPERS),MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
def checked(args):subprocess.run([str(v) for v in args],env=ENV,check=True)
CHECKPOINT=STORE/'weights/faster_rcnn_sspnet.pth';CHECKPOINT.parent.mkdir(exist_ok=True)
if not CHECKPOINT.exists():
    checked([PYTHON,'-c',"import gdown; assert gdown.download(id='1IfPCt5xZqqBJ3sYVIuD5F9l29Jcy2Hn1',output=%r,quiet=False)"%str(CHECKPOINT)])
import hashlib
if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!='912cc5b6f8a45b0f4521c9bf325868ed10dd4757497fb6ba8843fdbd26b09e4a':
    raise ValueError('원본 가중치 해시가 다릅니다.')''')
add('code','''DATA_ZIP=Path(DATA_ZIP_PATH) if DATA_ZIP_PATH else WORK/'recall_v1.zip'
if not DATA_ZIP.is_file():
    print('검토 승인된 recall_v1.zip 한 개를 선택하세요.')
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('ZIP 한 개가 필요합니다.')
    DATA_ZIP.write_bytes(next(iter(uploaded.values())))
DATA=WORK/'recall_v1';DATA.mkdir(exist_ok=True)
with zipfile.ZipFile(DATA_ZIP) as archive:
    for item in archive.infolist():
        target=(DATA/item.filename).resolve()
        if target!=DATA.resolve() and DATA.resolve() not in target.parents:raise ValueError('ZIP 경로가 잘못됐습니다.')
    archive.extractall(DATA)
checked([PYTHON,'-c',"from recall_selection import verify_bundle; verify_bundle(%r); print('검토 승인·해시·촬영 그룹 분리 확인 완료')"%str(DATA)])''')
add('markdown','''시험 학습은 실제 2회 업데이트와 원본·학습 후 검증을 실행합니다. 본 학습은 원본에서 별도로 시작합니다.
재개할 때는 같은 실행 이름·설정으로 환경과 데이터를 준비하고 시험 셀을 건너뛴 뒤 본 학습 셀의 RESUME을 켜세요.''')
add('code','''COMMON=[PYTHON,HELPERS/'train.py','--repo',REPO,'--data',DATA,'--checkpoint',CHECKPOINT,
        '--epochs',EPOCHS,'--learning-rate',LEARNING_RATE,'--fp-per-image',FP_PER_IMAGE]
checked(COMMON+['--output',STORE/'training'/RUN_NAME/'smoke','--mode','smoke','--smoke-steps',2])''')
add('code','''RESUME=False
OUTPUT=STORE/'training'/RUN_NAME/'main'
args=COMMON+['--output',OUTPUT]
if RESUME:args+=['--resume']
checked(args)
result=json.loads((OUTPUT/'best_selection.json').read_text())
print('최적 가중치:',OUTPUT/'best_small_swimmer.pth')
print('최적 epoch·임계값:',result['epoch'],result['point']['threshold'])
print('원본 유지 여부:',result['epoch']==0)''')
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(accelerator='GPU',
    kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3')))
if args.version=='recall_v2':
    for cell in cells:
        cell['source']=[line.replace('recall_v1','recall_v2') for line in cell['source']]
dest=ROOT/'notebooks'/('SSPNet_추가학습_무료Colab.ipynb' if args.version=='recall_v1' else 'SSPNet_재현율우선_recall_v2_Colab.ipynb')
dest.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('재현율 우선 Colab 노트북 생성 완료:',dest)
