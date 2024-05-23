#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : jsonCompress.py
# @Author: niceliu
# @Date  : 2022/4/11 11:28
# @Desc  :

import json
import os, glob,sys
import numpy as np
import os.path as osp
ROOT = sys.argv[1]
jsons = glob.glob(osp.join(ROOT,'track2_output_eval_exp*.json'))

print(jsons)
for json_file in jsons:
    print(json_file,'compressing...')
    data = json.load(open(json_file))
    new_json = {}
    for img_id, detection in data.items():
        boxes = np.array(detection['boxes'])
        scores =  np.array(detection['scores'])
        labels =  np.array(detection['labels'])

        n_boxes = np.around(boxes, 3)
        n_scores = np.around(scores, 3)
        new_json[img_id] = {
            "boxes":n_boxes.tolist(),
            "scores":n_scores.tolist(),
            "labels":labels.tolist(),
        }
    name = json_file.split('/')[-1]
    #name = name.replace('track2','track3') 
    with open(name, 'w') as f:
        json.dump(new_json,f)

