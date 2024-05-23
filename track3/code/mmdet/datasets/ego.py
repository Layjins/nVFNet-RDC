#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ego.py
# @Author: niceliu
# @Date  : 2022/5/23 16:41
# @Desc  :
import json
import os
from collections import defaultdict
import numpy as np
from .coco import CustomDataset
from .builder import DATASETS
from ego_objects import EgoObjects
from .ego_utils import make_compact_category_ids_mapping,\
    remap_category_ids,EgoEvaluator,make_instance_based

import os.path as osp
import tempfile
from terminaltables import AsciiTable
from mmcv.utils import print_log



@DATASETS.register_module()
class EgoCatDetDataset(CustomDataset):


    def load_annotations(self, ann_file):

        self.ego_api = self.load_ego_api(ann_file)

        data = json.load(open(ann_file))
        images,annotations = data['images'],data['annotations']
        data_info, categories = data['info'],data['categories']
        categories = sorted(categories,key=lambda x: x['id'])
        cat2label = {cat_id['id']: i+1 for i, cat_id in enumerate(categories)}
        self.CLASSES = [cat['name'] for cat in categories]
        data_infos = []
        self.img_ids = []
        anno_dicts = defaultdict(dict)
        for ann in annotations:
            img_id = ann['image_id']
           
            label = cat2label[ann['category_id']]
            box = ann['bbox']
            if img_id not in anno_dicts.keys():
                _ann = dict(bboxes=[box],
                           labels=[label])
                anno_dicts[img_id]=_ann
            else:
                anno_dicts[img_id]['bboxes'].append(box)
                anno_dicts[img_id]['labels'].append(label)

        for img in images:
            self.img_ids.append(img['id'])
            box_ann = anno_dicts[img['id']]['bboxes']
            box_ann = np.array(box_ann).astype(np.float32)

            box_ann[:,2] = box_ann[:,0] + box_ann[:,2]
            box_ann[:,3] = box_ann[:,1] + box_ann[:,3]

            label_ann = anno_dicts[img['id']]['labels']
            label_ann = np.array(label_ann).astype(np.int64)

            data_ann=dict(
                filename=img['url'].split('/')[-1],
                width=img['width'],
                height=img['height'],
                ann=dict(bboxes = box_ann,
                    labels = label_ann))
            data_infos.append(data_ann)

        return data_infos

    def load_ego_api(self,ann_file):
        ego_api = EgoObjects(ann_file)
        categories_id_mapping = make_compact_category_ids_mapping(
            ego_api)
        remap_category_ids(ego_api, categories_id_mapping)

        return ego_api
    def format_results(self, results, jsonfile_prefix=None, **kwargs):

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        json_results = self.detection_exporter(results)

        with open('./test.json', 'w') as f:
            json.dump(json_results, f)
            #json.dump(json_results, f, cls=TensorEncoder)

        return {}, tmp_dir

    def detection_exporter(self,results):
        json_results = {}
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]

            bboxes = np.vstack(result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32)
                       for i, bbox in enumerate(result)]
            labels = np.concatenate(labels)
            if not len(labels): continue
            sb = np.around(bboxes[:,:-1], 3)
            ss = np.around(bboxes[:,-1], 3)
            json_results[img_id]={"boxes": sb.tolist(),
                                  "scores":ss.tolist(),
                                  "labels":labels.tolist()}
        return json_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        self.iou_types = metric if isinstance(metric, list) else [metric]
        self.lvis_evaluator = EgoEvaluator(self.ego_api, self.iou_types)

        assert isinstance(results, list), 'results must be a list'
        #print(results[0])
        json_results = self._det2json(results)

        result_dict = self.lvis_evaluator.evaluate(json_results)
        self.lvis_evaluator.summarize()

        key_list,value_list = [],[]
        for key,value in result_dict['bbox'].items():
            value = '{:.3f}'.format(value*100)
            key_list.append(key)
            value_list.append(value)
        table_data = [key_list]
        table_data.append(value_list)
        table = AsciiTable(table_data)

        print_log('\n'+table.table, logger=logger)


        return result_dict


    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] =self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = label #self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def xyxy2xywh(self, bbox):

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
            ]

@DATASETS.register_module()
class EgoInsDetDataset(EgoCatDetDataset):

    def get_all_cats(self,ann_file):
         path = os.path.dirname(ann_file)
         #ann_file = os.path.join(path,'ego_objects_challenge_valSplit_submit.json')
         #ann_file = '/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_valSplit_submit.json'
         DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
         ann_file = DATASET_PATH + '/' + 'ego_objects_challenge_valSplit_submit.json'
         data = json.load(open(ann_file))
         images,annotations = data['images'],data['annotations']
         # get main_id
         main_ids = []
         for img in images:
             main_cat_ids = img['main_category_instance_ids']
             if len(main_cat_ids)>1: continue
             main_ids.append(main_cat_ids[0])

         id2annos = defaultdict(dict)
         for ann in annotations:
             id2annos[ann['id']] = ann

         categories = [id2annos[ind]['instance_id'] for ind in main_ids]
         categories = sorted(list(set(categories)))
         return categories

    def load_annotations(self, ann_file):

        self.categories = self.get_all_cats(ann_file) 
        file = open('cats.txt','w')
        for x in self.categories: file.write(x+'\n')
        file.close()
        self.ego_api = self.load_ego_api(ann_file)
 
        self.img_ids = []
        data = json.load(open(ann_file))
        images,annotations = data['images'],data['annotations']

        id2annos = defaultdict(dict)
        for ann in annotations:
            id2annos[ann['id']] = ann

        self.CLASSES = self.categories
        cat2label = {cat: i+1 for i, cat in enumerate(self.categories)}

        data_infos = []
        for img in images:
            self.img_ids.append(img['id'])
            main_cat_ids = img['main_category_instance_ids']
            if len(main_cat_ids)>1: continue
            main_ann = id2annos[main_cat_ids[0]]

            box,ins_label = main_ann['bbox'],main_ann['instance_id']
            box = np.array([box]).astype(np.float32)
            box[:,2] = box[:,0] + box[:,2]
            box[:,3] = box[:,1] + box[:,3]

            ins_label = cat2label[ins_label] if ins_label in cat2label.keys() else -1
            ins_label = np.array([ins_label]).astype(np.int64)

            data_ann=dict(
                filename=img['url'].split('/')[-1],
                width=img['width'],
                height=img['height'],
                ann=dict(bboxes = box,
                         labels = ins_label))
            data_infos.append(data_ann)

        return data_infos

    def load_ego_api(self,ann_file):

        ego_api = EgoObjects(ann_file)
        make_instance_based(
            ego_api,self.categories
        )

        return ego_api



















