_base_ = [
    '../_base_/datasets/ego.py',
    '../vfnet/vfnet_r50_fpn_1x_coco.py',
]

load_from='/youtu/fuxi-team2-2/persons/niceliu/CLVision2022/models/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth'
pretrained=load_from

model = dict(
    backbone=dict(
        type='CBResNet',
        cb_del_stages=1,
        cb_inplanes=[64, 256, 512, 1024, 2048], 
    ),
    neck=dict(
        type='CBFPN',
    ),
    bbox_head=dict(num_classes=277),
    test_cfg=dict(
        score_thr=0.005,
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
        #img_scale=[(1920, 1080), (968, 728)],
        img_scale=(1333, 800),
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
        img_scale=(1333, 800),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12, metric='bbox')
checkpoint_config = dict(interval=12)
