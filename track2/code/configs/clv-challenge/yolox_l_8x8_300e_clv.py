_base_ = [
    '../yolox/yolox_l_8x8_300e_coco.py',
]
model = dict(
        bbox_head=dict(num_classes=277))

data = dict(workers_per_gpu=0)
