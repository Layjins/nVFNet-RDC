#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ego_utils.py
# @Author: niceliu
# @Date  : 2022/5/24 9:57
# @Desc  :

import copy
from collections import defaultdict
from ego_objects import EgoObjects, EgoObjectsResults, EgoObjectsEval

def make_compact_category_ids_mapping(
        ego_api, test_ego_api=None):
    train_category_ids = set(ego_api.get_cat_ids())

    if test_ego_api is not None:
        if train_category_ids != set(test_ego_api.get_cat_ids()):
            raise ValueError(
                'Train and test datasets must contain the same categories!')

    return list(sorted(train_category_ids)) # the cat Id start 0

def remap_category_ids(ego_api, categories_id_mapping):
    """
    Remaps the category IDs by modifying the API object in-place.

    :param ego_api: The API object to adapt.
    :param categories_id_mapping: The category mapping. It must define a
        mapping from the to-be-used-id to the real category id so that:
        `real_cat_id = categories_id_mapping[mapped_id]`.
    """
    reversed_mapping = dict()
    for mapped_id, real_id in enumerate(categories_id_mapping):
        reversed_mapping[real_id] = mapped_id+1

    dataset_json = ego_api.dataset

    for cat_dict in dataset_json['categories']:
        cat_dict['id'] = reversed_mapping[cat_dict['id']]

    for ann_dict in dataset_json['annotations']:
        ann_dict['category_id'] = reversed_mapping[ann_dict['category_id']]

    ego_api.recreate_index()

def make_instance_based(ego_api,categories):
    main_annotations_ids = set()
    main_annotations_dicts = []
    unique_object_ids = set()

    ego_dataset = ego_api.dataset
    for img_dict in ego_dataset['images']:
        main_category_instance_ids = img_dict['main_category_instance_ids']
        assert len(main_category_instance_ids) == 1

        main_annotations_ids.add(main_category_instance_ids[0])

    for ann_dict in ego_dataset['annotations']:
        if ann_dict['id'] in main_annotations_ids:
            main_annotations_dicts.append(ann_dict)
            unique_object_ids.add(ann_dict['instance_id'])
    ego_dataset['annotations'] = main_annotations_dicts

    unique_object_ids_sorted = categories
    reversed_mapping = dict()
    for mapped_id, real_id in enumerate(unique_object_ids_sorted):
        reversed_mapping[real_id] = mapped_id+1

    img_count = defaultdict(int)
    for ann_dict in ego_dataset['annotations']:
        inst_id = ann_dict['instance_id']
        if inst_id not in reversed_mapping.keys(): continue
        new_id = reversed_mapping[inst_id]
        ann_dict['category_id'] = new_id
        img_count[new_id] += 1

    new_categories = []

    for cat_id in unique_object_ids_sorted:  # Exclude the background
        new_cat_dict = dict(
            id=reversed_mapping[cat_id],
            name=f'Object{cat_id}',
            image_count=img_count[cat_id],
            instance_count=img_count[cat_id]
        )
        new_categories.append(new_cat_dict)
    ego_dataset['categories'] = new_categories
    ego_api._fix_frequencies()
    ego_api._create_index()


class EgoEvaluator:
    def __init__(self, ego_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.ego_gt = ego_gt

        self.iou_types = iou_types
        self.img_ids = []
        self.predictions = []
        self.ego_eval_per_iou = dict()

    def evaluate(self, results, max_dets_per_image=None):
        #all_preds, main_process = self.synchronize_between_processes()
        all_preds, main_process = results, True
        if main_process:
            if max_dets_per_image is None:
                max_dets_per_image = 300

            eval_imgs = [ego_res['image_id'] for ego_res in all_preds]

            gt_subset = EgoEvaluator._make_ego_subset(self.ego_gt, eval_imgs)

            for iou_type in self.iou_types:
                print('Evaluating for iou', iou_type)
                if iou_type == "segm":
                    # See:
                    # https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/lvis_evaluation.html
                    ego_results = copy.deepcopy(all_preds)
                    for c in ego_results:
                        c.pop("bbox", None)
                else:
                    ego_results = all_preds

                ego_results = EgoObjectsResults(
                    gt_subset,
                    ego_results,
                    max_dets=max_dets_per_image)
                ego_eval = EgoObjectsEval(gt_subset, ego_results, iou_type)
                ego_eval.params.img_ids = list(set(eval_imgs))
                ego_eval.run()
                self.ego_eval_per_iou[iou_type] = ego_eval
        else:
            self.ego_eval_per_iou = None

        result_dict = None
        if self.ego_eval_per_iou is not None:
            result_dict = dict()
            for iou, eval_data in self.ego_eval_per_iou.items():
                result_dict[iou] = dict()
                for key in eval_data.results:
                    value = eval_data.results[key]
                    result_dict[iou][key] = value

        return result_dict

    def summarize(self):
        if self.ego_eval_per_iou is not None:
            for iou_type, ego_eval in self.ego_eval_per_iou.items():
                print(f"IoU metric: {iou_type}")
                ego_eval.print_results()

    @staticmethod
    def _make_ego_subset(ego_gt, img_ids):
        img_ids = set(img_ids)

        subset = dict()
        subset['categories'] = list(ego_gt.dataset["categories"])

        subset_imgs = []
        for img in ego_gt.dataset["images"]:
            if img["id"] in img_ids:
                subset_imgs.append(img)
        subset['images'] = subset_imgs

        subset_anns = []
        for ann in ego_gt.dataset["annotations"]:
            if ann["image_id"] in img_ids:
                subset_anns.append(ann)
        subset['annotations'] = subset_anns

        return EgoObjects('', annotation_dict=subset)
