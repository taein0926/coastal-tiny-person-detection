"""이전 실험 재현용: 검증 없이 YoloBeachTrain 전체로 학습하던 코드. 새 학습에는 train.py를 사용한다."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_annotations(source, dest):
    raw = json.loads(Path(source).read_text(encoding='utf-8'))
    names = {c['id']: c['name'].lower().replace('_', ' ').strip() for c in raw['categories']}
    wanted = {i for i, name in names.items() if name in ('person water', 'person sand')}
    if set(names[i] for i in wanted) != {'person water', 'person sand'}:
        raise ValueError('두 사람 라벨을 찾지 못했습니다: ' + repr(names))
    raw['images'] = [dict(im, corner=im.get('corner', [0, 0, im['width'], im['height']])) for im in raw['images']]
    raw['annotations'] = [dict(a, category_id=1, ignore=int(bool(a.get('ignore', False) or a.get('iscrowd', False))),
                               uncertain=int(bool(a.get('uncertain', False))), segmentation=a.get('segmentation', []))
                          for a in raw['annotations'] if a['category_id'] in wanted]
    raw['categories'] = [dict(id=1, name='person', supercategory='person')]
    Path(dest).write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    return len(raw['images']), len(raw['annotations'])


def main():
    p = argparse.ArgumentParser()
    for arg in ('repo', 'annotations', 'checkpoint', 'output', 'mode'):
        p.add_argument('--' + arg, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--smoke-steps', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    # 원본 ICNegSampler는 정답이 없는 이미지에서 빈 행렬의 max를 호출한다.
    # 이 경우 모든 후보의 정답과의 겹침은 0이다.
    sampler = Path(a.repo) / 'mmdet/core/bbox/samplers/ic_base_sampler.py'
    sampler_text = sampler.read_text(encoding='utf-8')
    old = '        iod_max_overlaps, _ = iod_overlaps.max(dim=0)'
    new = '        iod_max_overlaps = iod_overlaps.max(dim=0)[0] if gt_bboxes.numel() else bboxes.new_zeros((bboxes.shape[0],))'
    if old in sampler_text:
        sampler.write_text(sampler_text.replace(old, new), encoding='utf-8')
    elif new not in sampler_text:
        raise RuntimeError('ICNegSampler 코드가 예상과 다릅니다.')
    heatmap = Path(a.repo) / 'mmdet/models/module/heatmap.py'
    heatmap_text = heatmap.read_text(encoding='utf-8')
    old_threshold = 'threshold = -bg_score_sorted[bg_num - 1]'
    new_threshold = 'threshold = -bg_score_sorted[min(bg_num, len(bg_score_sorted)) - 1] if bg_num > 0 and len(bg_score_sorted) else float("inf")'
    if heatmap_text.count(old_threshold) == 2:
        heatmap.write_text(heatmap_text.replace(old_threshold, new_threshold), encoding='utf-8')
    elif heatmap_text.count(new_threshold) != 2:
        raise RuntimeError('Heatmap 코드가 예상과 다릅니다.')
    import torch
    from mmcv import Config
    from mmdet.apis import set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    if not torch.cuda.is_available():
        raise RuntimeError('GPU 런타임이 필요합니다.')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    ann = Path(a.annotations).with_name('training_person.coco.json')
    counts = make_annotations(a.annotations, ann)
    if counts[0] != 2961:
        raise ValueError('YoloBeachTrain v2 전체 2961장이 아닙니다: ' + str(counts))
    cfg = Config.fromfile(str(Path(a.repo) / 'configs/sspnet/faster_rcnn_r50_sspnet_1x_coco.py'))
    cfg.model.pretrained = None
    pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1280, 1024), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(type='Normalize', mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32), dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img','gt_bboxes','gt_labels','gt_bboxes_ignore'])]
    cfg.data = dict(samples_per_gpu=1, workers_per_gpu=1,
                    train=dict(type='CocoDataset', ann_file=str(ann), img_prefix=str(ann.parent) + '/',
                               classes=('person',), filter_empty_gt=False, pipeline=pipeline))
    cfg.optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=None)
    cfg.lr_config = dict(policy='fixed')
    cfg.workflow = [('train', 1)]
    cfg.gpu_ids = [0]
    cfg.seed = 0
    cfg.work_dir = str(output)
    cfg.load_from = None
    cfg.resume_from = None
    cfg.log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
    cfg.runner = dict(type='IterBasedRunner', max_iters=a.smoke_steps) if a.mode == 'smoke' else dict(type='EpochBasedRunner', max_epochs=a.epochs)
    cfg.pop('total_epochs', None)
    cfg.checkpoint_config = dict(interval=a.smoke_steps if a.mode == 'smoke' else 1, by_epoch=a.mode != 'smoke', max_keep_ckpts=2)
    signature = dict(dataset_sha256=digest(a.annotations), initial_sha256=digest(a.checkpoint),
                     mode=a.mode, epochs=a.epochs, smoke_steps=a.smoke_steps,
                     learning_rate=0.0001, batch_size=1, image_scale=[1280,1024], seed=0)
    signature_path = output / 'experiment.json'
    if signature_path.exists() and json.loads(signature_path.read_text(encoding='utf-8')) != signature:
        raise ValueError('기존 실행의 데이터 또는 설정이 다릅니다. 새 실행 폴더를 사용하세요.')
    latest = output / 'latest.pth'
    if a.resume:
        if not signature_path.is_file() or not latest.is_file():
            raise ValueError('재개할 설정 또는 latest.pth가 없습니다.')
        cfg.resume_from = str(latest)
    elif signature_path.exists() or latest.exists():
        raise ValueError('기존 실행 폴더입니다. 재개하거나 새 실행 폴더를 지정하세요.')
    signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
    cfg.dump(str(output / 'resolved_config.py'))
    set_random_seed(0, deterministic=True)
    dataset = build_dataset(cfg.data.train)
    print('학습 이미지/사람 주석:', counts, '실제 데이터셋:', len(dataset), flush=True)
    if len(dataset) != counts[0]:
        raise ValueError('데이터셋에서 제외된 이미지가 있습니다.')
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    checkpoint = torch.load(a.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.CLASSES = ('person',)
    train_detector(model, dataset, cfg, distributed=False, validate=False)
    if a.mode == 'train':
        final = output / ('epoch_%d.pth' % a.epochs)
        if not final.is_file():
            raise RuntimeError('마지막 epoch 가중치가 없습니다.')
        print('최종 학습 가중치:', final, flush=True)


if __name__ == '__main__':
    main()
