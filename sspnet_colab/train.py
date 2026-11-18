"""검토 승인된 수영자 자료로 학습하고 작은 수영자 재현율 최적 모델을 보존한다."""
import argparse
import json
import shutil
from pathlib import Path
from recall_selection import digest,verify_bundle,select_operating_point,selection_key


def patch_training_source(repo):
    replacements=[('mmdet/core/bbox/samplers/ic_base_sampler.py',
        '        iod_max_overlaps, _ = iod_overlaps.max(dim=0)',
        '        iod_max_overlaps = iod_overlaps.max(dim=0)[0] if gt_bboxes.numel() else bboxes.new_zeros((bboxes.shape[0],))'),
        ('mmdet/models/module/heatmap.py','threshold = -bg_score_sorted[bg_num - 1]',
         'threshold = -bg_score_sorted[min(bg_num, len(bg_score_sorted)) - 1] if bg_num > 0 and len(bg_score_sorted) else float("inf")')]
    for file,old,new in replacements:
        path=repo/file;text=path.read_text(encoding='utf-8')
        if old in text:path.write_text(text.replace(old,new),encoding='utf-8')
        elif new not in text:raise RuntimeError('학습 소스가 예상과 다릅니다: '+file)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','data','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--fp-per-image',type=float,default=4.)
    p.add_argument('--match-iou',type=float,default=.3)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--mode',choices=('train','smoke'),default='train')
    p.add_argument('--smoke-steps',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or args.smoke_steps<1 or args.patience<1 or args.learning_rate<=0 or args.fp_per_image<0:
        raise ValueError('학습·검증 설정의 범위를 확인하세요.')
    root=Path(args.data).resolve();output=Path(args.output).resolve()
    manifest,raw=verify_bundle(root)
    print('승인 데이터 확인:',{s:len(r['images']) for s,r in raw.items()},flush=True)
    output.mkdir(parents=True,exist_ok=True)
    source=Path(args.repo).resolve();repo=output/'training_source'
    if not repo.exists():shutil.copytree(source,repo,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    patch_training_source(repo)
    import sys
    sys.path.insert(0,str(repo))
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import Hook,build_runner,build_optimizer
    from mmdet.apis import set_random_seed,single_gpu_test
    from mmdet.datasets import build_dataset,build_dataloader
    from mmdet.models import build_detector
    from mmdet.utils import get_root_logger
    if not torch.cuda.is_available():raise RuntimeError('GPU 런타임이 필요합니다.')
    torch.set_num_threads(2)
    cfg=Config.fromfile(str(repo/'configs/sspnet/faster_rcnn_r50_sspnet_1x_coco.py'))
    cfg.model.pretrained=None;cfg.model.backbone.frozen_stages=3
    cfg.test_cfg.rcnn.score_thr=.001;cfg.test_cfg.rcnn.nms.min_score=.001;cfg.test_cfg.rcnn.max_per_img=500
    normalize=dict(type='Normalize',mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True)
    train_pipeline=[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
        dict(type='Resize',img_scale=[(640,512),(960,768),(1280,1024)],multiscale_mode='value',keep_ratio=True),
        dict(type='RandomFlip',flip_ratio=.5),normalize,dict(type='Pad',size_divisor=32),dict(type='DefaultFormatBundle'),
        dict(type='Collect',keys=['img','gt_bboxes','gt_labels','gt_bboxes_ignore'])]
    val_pipeline=[dict(type='LoadImageFromFile'),dict(type='MultiScaleFlipAug',img_scale=(1280,1024),flip=False,
        transforms=[dict(type='Resize',keep_ratio=True),dict(type='RandomFlip'),normalize,dict(type='Pad',size_divisor=32),
                    dict(type='ImageToTensor',keys=['img']),dict(type='Collect',keys=['img'])])]
    dataset_cfg=lambda split,pipeline:dict(type='CocoDataset',ann_file=str(root/(split+'.json')),img_prefix=str(root)+'/',
                                         classes=('person',),filter_empty_gt=False,pipeline=pipeline)
    cfg.data=dict(samples_per_gpu=1,workers_per_gpu=0,train=dataset_cfg('train',train_pipeline),val=dataset_cfg('val',val_pipeline))
    cfg.optimizer=dict(type='SGD',lr=args.learning_rate,momentum=.9,weight_decay=.0001,
                       paramwise_cfg=dict(custom_keys={'backbone':dict(lr_mult=.1),'neck':dict(lr_mult=.1)}))
    cfg.optimizer_config=dict(grad_clip=dict(max_norm=35,norm_type=2));cfg.lr_config=dict(policy='fixed')
    cfg.workflow=[('train',1)];cfg.gpu_ids=[0];cfg.seed=0;cfg.work_dir=str(output)
    cfg.runner=dict(type='EpochBasedRunner',max_epochs=1 if args.mode=='smoke' else args.epochs)
    cfg.pop('total_epochs',None)
    cfg.checkpoint_config=dict(interval=1,by_epoch=True,max_keep_ckpts=2,create_symlink=False)
    cfg.log_config=dict(interval=1 if args.mode=='smoke' else 10,hooks=[dict(type='TextLoggerHook')])
    signature=dict(quality_manifest_sha256=digest(root/'quality_manifest.json'),initial_sha256=digest(args.checkpoint),
        train_code_sha256=digest(__file__),selection_code_sha256=digest(Path(__file__).with_name('recall_selection.py')),
        mode=args.mode,epochs=args.epochs,learning_rate=args.learning_rate,fp_per_image=args.fp_per_image,match_iou=args.match_iou,
        patience=args.patience,smoke_steps=args.smoke_steps,frozen_stages=3,seed=0)
    signature_path=output/'experiment.json'
    if args.resume:
        if not signature_path.exists() or json.loads(signature_path.read_text())!=signature or not (output/'latest.pth').exists():
            raise ValueError('재개할 실험의 설정·최신 체크포인트를 확인하세요.')
    elif signature_path.exists():raise ValueError('기존 실행 폴더입니다. --resume 또는 새 출력 폴더를 사용하세요.')
    signature_path.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    cfg.dump(str(output/'resolved_config.py'))
    set_random_seed(0,deterministic=True)
    trainset=build_dataset(cfg.data.train);valset=build_dataset(cfg.data.val,dict(test_mode=True))
    if args.mode=='smoke':
        trainset.data_infos=trainset.data_infos[:args.smoke_steps];trainset.img_ids=trainset.img_ids[:args.smoke_steps]
        trainset.flag=trainset.flag[:args.smoke_steps]
    trainloader=build_dataloader(trainset,1,0,num_gpus=1,dist=False,seed=0)
    valloader=build_dataloader(valset,1,0,num_gpus=1,dist=False,shuffle=False)
    model=build_detector(cfg.model,train_cfg=cfg.train_cfg,test_cfg=cfg.test_cfg)
    checkpoint=torch.load(args.checkpoint,map_location='cpu');state=checkpoint.get('state_dict',checkpoint)
    if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model.CLASSES=('person',)
    model=MMDataParallel(model.cuda(),device_ids=[0])
    runner=build_runner(cfg.runner,default_args=dict(model=model,optimizer=build_optimizer(model,cfg.optimizer),
        work_dir=str(output),logger=get_root_logger(log_level='INFO'),meta=dict(config=cfg.pretty_text,CLASSES=('person',),quality_manifest_sha256=signature['quality_manifest_sha256'])))
    runner.register_training_hooks(cfg.lr_config,cfg.optimizer_config,cfg.checkpoint_config,cfg.log_config)

    class SmallSwimmerValidation(Hook):
        """원본과 매 epoch를 실제 작은 수영자 재현율로 검증한다."""
        def __init__(self):
            self.best=None;self.stale=0;self.history=[]
            if args.resume:
                self.best=json.loads((output/'best_selection.json').read_text())
                self.history=json.loads((output/'validation_history.json').read_text())
                self.stale=self.best.get('stale_epochs',0)

        def before_run(self,runner):self.evaluate(runner,runner.epoch)

        def after_train_epoch(self,runner):
            self.evaluate(runner,runner.epoch+1)
            # Windows 바인드 마운트에서도 재개 파일을 사용할 수 있도록 심볼릭 링크 대신 저장한다.
            runner.save_checkpoint(str(output),filename_tmpl='latest.pth',create_symlink=False)
            if self.stale>=args.patience:
                print('검증 재현율 개선이 없어 학습을 조기 종료합니다.',flush=True)
                runner._max_epochs=runner.epoch+1

        def evaluate(self,runner,epoch):
            print(f'작은 수영자 검증 시작: epoch {epoch}',flush=True)
            results=single_gpu_test(runner.model,valloader)
            predictions=[]
            for im,result in zip(valset.data_infos,results):
                if len(result)!=1:raise ValueError('단일 사람 클래스 출력이 아닙니다.')
                for x,y,r,b,score in result[0]:
                    if r>x and b>y:
                        predictions.append(dict(image_id=im['id'],category_id=1,bbox=[float(x),float(y),float(r-x),float(b-y)],score=float(score)))
            result=select_operating_point(raw['val'],predictions,args.fp_per_image,args.match_iou)
            point=result['best'];improved=self.best is None or selection_key(point)>selection_key(self.best['point'])
            if improved:
                self.stale=0
                self.best=dict(epoch=epoch,point=point,definition=result['small_definition'],fp_budget=args.fp_per_image,
                               match_iou=args.match_iou,quality_manifest_sha256=signature['quality_manifest_sha256'])
                runner.save_checkpoint(str(output),filename_tmpl='best_small_swimmer.pth',create_symlink=False,
                                       meta=dict(recall_selection=self.best))
            elif not self.history or self.history[-1]['epoch']!=epoch:self.stale+=1
            self.best['stale_epochs']=self.stale
            row=dict(epoch=epoch,selected=point,improved=improved,threshold_sweep=result['points'])
            self.history=[v for v in self.history if v['epoch']!=epoch]+[row]
            for name,value in [('best_selection.json',self.best),('validation_history.json',self.history)]:
                tmp=output/(name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(output/name)
            print(f'epoch {epoch}: 작은 수영자 {point["small_swimmer_hits"]}/{point["small_swimmer_positives"]} '
                  f'재현율 {point["small_swimmer_recall"]:.1%}, 전체 {point["person_recall"]:.1%}, '
                  f'FP/이미지 {point["FP_per_image"]:.2f}, 임계값 {point["threshold"]:.4f}, 최적 갱신 {improved}',flush=True)
            runner.model.train()

    runner.register_hook(SmallSwimmerValidation(),priority='LOW')
    if args.resume:runner.resume(str(output/'latest.pth'))
    runner.run([trainloader],cfg.workflow)
    best=json.loads((output/'best_selection.json').read_text())
    print('선택한 최적 epoch:',best['epoch'],'가중치:',output/'best_small_swimmer.pth',flush=True)
    if best['epoch']==0:print('추가 학습이 원본 재현율을 개선하지 못해 원본 모델을 유지했습니다.',flush=True)


if __name__=='__main__':main()
