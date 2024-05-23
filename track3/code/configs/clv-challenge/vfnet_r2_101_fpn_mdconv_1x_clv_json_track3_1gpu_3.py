_base_ = [
    '../_base_/datasets/ego_ins.py',
    '../vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
]

load_from='/youtu/fuxi-team2-2/persons/niceliu/CLVision2022/models/vfnet_r2_101_dcn_ms_2x_51.1.pth'
#load_from='/youtu/fuxi-team2-2/persons/niceliu/CLVision2022/models/SWA-VFNetX-1-18-53.4_VFNetX-R2101-41e-0.01-0.0001-52.2.pth'

model = dict(
    backbone=dict(frozen_stages=2),
    #backbone=dict(
    #    norm_cfg=dict(type='SyncBN', requires_grad=True),
    #    norm_eval=False),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1110,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    test_cfg=dict(
        score_thr=0.005,
        #nms=dict(type='soft_nms', iou_threshold=0.6, min_score=0.06),
        max_per_img=300))

replay = dict(
    replay_flag=True,
    replay_img_num=5000,
    replay_mode='splitImagesByVid')

knowledge_distill = dict(
    kd_flag=True)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        #img_scale=[(1920, 1080), (968, 728)],
        img_scale=(533, 400),
        #img_scale=(2000, 1200),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(533, 400),
        #img_scale=(2000, 1200),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

usingClassBalanced = False
if usingClassBalanced:
    # ClassBalance
    dataset_type = 'EgoInsDetDataset'
    data_root = '/youtu/fuxi-team2-2/CLVision/submit_datasets/'
    data = dict(
        samples_per_gpu=12,
        workers_per_gpu=4,
        train=dict(
            _delete_=True,
            type='ClassBalancedDataset',
            oversample_thr=1e-3,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root + 'ego_objects_challenge_train_track3Submit_exp0.json',
                img_prefix=data_root + 'images/',
                pipeline=train_pipeline)),
        val=dict(pipeline=test_pipeline),
        test=dict(pipeline=test_pipeline))
else:
    data = dict(
        samples_per_gpu=12,
        workers_per_gpu=4,
        train=dict(pipeline=train_pipeline),
        val=dict(pipeline=test_pipeline),
        test=dict(pipeline=test_pipeline))



# optimizer
optimizer = dict(
    lr=0.005, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)
evaluation = dict(interval=6, metric='bbox')
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=6)
