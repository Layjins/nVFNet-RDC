

_base_ = [
    '../fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py',
]

model = dict(
    bbox_head=dict(num_classes=277),
    test_cfg=dict(
        score_thr=0.005,
        max_per_img=300))

data = dict(workers_per_gpu=0)
evaluation = dict(interval=12, metric='bbox')
checkpoint_config = dict(interval=12)
