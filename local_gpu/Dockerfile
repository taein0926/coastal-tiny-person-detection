FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN python -m pip install --no-cache-dir 'numpy==1.21.6' 'scipy==1.7.3' 'matplotlib==3.5.3' 'Pillow==9.5.0' 'opencv-python-headless==4.5.5.64' 'pycocotools==2.0.7' 'addict==2.4.0' 'yapf==0.31.0' 'PyYAML==6.0.1' 'packaging==23.2' 'terminaltables==3.1.10' 'six==1.16.0' 'tqdm==4.66.5'
RUN python -m pip install --no-cache-dir --no-deps mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
