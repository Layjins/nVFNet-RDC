_base_ = [
    '../deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
]

model = dict(bbox_head=dict(num_classes=277),
             test_cfg=dict(max_per_img=300))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,)

