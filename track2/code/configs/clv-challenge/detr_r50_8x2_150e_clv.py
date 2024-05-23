
_base_ = [
    '../detr/detr_r50_8x2_150e_coco.py',
]

model = dict(
    bbox_head=dict(num_classes=277),
    test_cfg=dict(max_per_img=300))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,)
runner = dict(type='EpochBasedRunner', max_epochs=50)
fp16 = dict(loss_scale=dict(init_scale=512))
