"""무료 Colab의 기본 Python과 분리하여 SSPNet 실행 환경을 만든다."""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/sspnet_free")
REPO = ROOT / "SSPNet"
VENV = ROOT / "venv"
PYTHON = VENV / "bin/python"
COMMIT = "f2d0f42273945577a5c6497fd2b384cb545f8f66"
STAMP = "python38-torch110-cu111-mmcv1318-v1"


def command(*args, env=None):
    subprocess.run([str(a) for a in args], check=True, env=env)


def setup():
    if platform.system() != "Linux" or platform.machine() != "x86_64" or not Path("/content").is_dir():
        raise RuntimeError("이 설치 셀은 Google Colab Linux 런타임에서 실행하세요.")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("무료 Colab에서 런타임 유형 변경 → GPU를 선택한 뒤 다시 실행하세요.")
    command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv")
    free = shutil.disk_usage("/content").free / 1024**3
    print("Colab 임시 디스크 여유: %.1f GiB" % free, flush=True)
    if free < 8:
        raise RuntimeError("환경 설치를 시작하려면 임시 디스크에 최소 8 GiB 여유가 필요합니다.")
    ROOT.mkdir(exist_ok=True)
    if not PYTHON.is_file():
        command(sys.executable,"-m","pip","install","--quiet","--no-cache-dir","uv==0.9.7")
        command("uv","python","install","3.8.20")
        command("uv","venv","--python","3.8.20","--seed",VENV)
    env = os.environ.copy()
    env.pop("PYTHONPATH",None)
    env.pop("PYTHONHOME",None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    marker = ROOT / "environment_ready.json"
    ready = json.loads(marker.read_text()).get("stamp") == STAMP if marker.exists() else False
    if not ready:
        command(PYTHON,"-m","pip","install","--no-cache-dir","pip==24.3.1",
                "setuptools==59.5.0","wheel==0.45.1",env=env)
        command(PYTHON,"-m","pip","install","--no-cache-dir",
                "torch==1.10.0+cu111","torchvision==0.11.1+cu111",
                "--find-links","https://download.pytorch.org/whl/cu111/torch_stable.html",env=env)
        command(PYTHON,"-m","pip","install","--no-cache-dir",
                "numpy==1.21.6","scipy==1.7.3","matplotlib==3.5.3","Pillow==9.5.0",
                "opencv-python-headless==4.5.5.64","pycocotools==2.0.7",
                "addict==2.4.0","yapf==0.31.0","PyYAML==6.0.1","packaging==23.2",
                "terminaltables==3.1.10","six==1.16.0","tqdm==4.66.5","gdown==5.2.0",env=env)
        # 사전 빌드된 연산자만 설치한다. 무료 GPU 시간으로 CUDA 소스 빌드를 하지 않는다.
        command(PYTHON,"-m","pip","install","--no-cache-dir","--no-deps",
                "https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/"
                "mmcv_full-1.3.18-cp38-cp38-manylinux1_x86_64.whl",env=env)
    if not REPO.exists():
        command("git","init",REPO)
        command("git","-C",REPO,"remote","add","origin","https://github.com/MingboHong/SSPNet.git")
        command("git","-C",REPO,"fetch","--depth=1","origin",COMMIT)
        command("git","-C",REPO,"checkout","--detach","FETCH_HEAD")
    actual = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    if actual != COMMIT:
        raise RuntimeError("기존 SSPNet 코드 버전이 다릅니다. 새 무료 런타임에서 설치 셀부터 실행하세요.")
    env["PYTHONPATH"] = str(REPO)
    check = '''
import json, sys, torch, torchvision, mmcv, mmdet
from mmcv.ops import nms, roi_align
assert torch.__version__ == '1.10.0+cu111'
assert mmcv.__version__ == '1.3.18'
assert torch.cuda.is_available(), 'GPU를 사용할 수 없습니다.'
a = torch.randn(32,32,device='cuda'); (a @ a).sum().item()
b = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda')
s = torch.tensor([.9,.8],device='cuda')
nms(b,s,.5)
roi_align(torch.ones(1,1,16,16,device='cuda'),torch.tensor([[0.,1.,1.,8.,8.]],device='cuda'),(3,3))
torch.cuda.synchronize()
print(json.dumps(dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
    cuda=torch.version.cuda,mmcv=mmcv.__version__,mmdet=mmdet.__version__,
    gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0)),ensure_ascii=False,indent=2))
'''
    command(PYTHON,"-c",check,env=env)
    marker.write_text(json.dumps(dict(stamp=STAMP,source_commit=actual)),encoding="utf-8")
    freeze = subprocess.check_output([str(PYTHON),"-m","pip","freeze"],env=env,text=True)
    (ROOT/"installed_packages.txt").write_text(freeze,encoding="utf-8")
    print("환경 확인 완료: CUDA 행렬 연산, NMS, RoIAlign, SSPNet 패키지 가져오기 성공",flush=True)


if __name__ == "__main__":
    setup()
