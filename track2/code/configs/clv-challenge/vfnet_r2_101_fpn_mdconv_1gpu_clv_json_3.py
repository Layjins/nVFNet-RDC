_base_ = [
    '../_base_/datasets/ego.py',
    '../vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco.py'
]

load_from='/youtu/fuxi-team2-2/persons/niceliu/CLVision2022/models/vfnet_r2_101_dcn_ms_2x_51.1.pth'

model = dict(
    backbone=dict(frozen_stages=3),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=277,
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
    replay_img_num=3500,
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
        #img_scale=[(1433, 900), (1920, 1080), (3999,1920)],
        img_scale=(1033,600),
        multiscale_mode='value',
        keep_ratio=True),
    #dict(
    #    type='RandomCrop',
    #    crop_type='absolute_range',
    #    crop_size=(500, 833),
    #    allow_negative_crop=True),
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
        #img_scale=[(533, 400),(633, 500),(833, 500),(1333, 800),(1433, 900)],
        #img_scale=(1433, 900),
        img_scale=(1033, 600),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
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
    step=[6, 8])
runner = dict(type='EpochBasedRunner', max_epochs=9)
evaluation = dict(interval=9, metric='bbox')
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=9)
