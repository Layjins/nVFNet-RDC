from collections import defaultdict
import json
from ego_objects import EgoObjects, EgoObjectsJson

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def make_instance_based(ego_api: EgoObjects, full_main_ids=False):
    if full_main_ids:
        DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/track2_datasets'
        train_anno_path = DATASET_PATH + '/' + 'ego_objects_challenge_train.json'
        full_train_ego_api = EgoObjects(train_anno_path)

        full_main_annotations_ids = set()
        full_unique_object_ids = set()

        for img_dict in full_train_ego_api.dataset['images']:
            main_category_instance_ids = img_dict['main_category_instance_ids']
            assert len(main_category_instance_ids) == 1
            full_main_annotations_ids.add(main_category_instance_ids[0])

        for ann_dict in full_train_ego_api.dataset['annotations']:
            if ann_dict['id'] in full_main_annotations_ids:
                full_unique_object_ids.add(ann_dict['instance_id'])
        

    ##########
    main_annotations_ids = set()
    main_annotations_dicts = []
    unique_object_ids = set()

    ego_dataset: EgoObjectsJson = ego_api.dataset
    for img_dict in ego_dataset['images']:
        main_category_instance_ids = img_dict['main_category_instance_ids']
        assert len(main_category_instance_ids) == 1

        main_annotations_ids.add(main_category_instance_ids[0])

    for ann_dict in ego_dataset['annotations']:
        if ann_dict['id'] in main_annotations_ids:
            main_annotations_dicts.append(ann_dict)
            unique_object_ids.add(ann_dict['instance_id'])
    ego_dataset['annotations'] = main_annotations_dicts

    if full_main_ids:
        unique_object_ids_sorted = ['background'] + list(sorted(full_unique_object_ids))
        #unique_object_ids_sorted = list(sorted(full_unique_object_ids)) + ['background'] # mmdetection
    else:
        unique_object_ids_sorted = ['background'] + list(sorted(unique_object_ids))
        #unique_object_ids_sorted = list(sorted(unique_object_ids)) + ['background'] # mmdetection
    reversed_mapping = dict()
    for mapped_id, real_id in enumerate(unique_object_ids_sorted):
        reversed_mapping[real_id] = mapped_id

    img_count = defaultdict(int)
    for ann_dict in ego_dataset['annotations']:
        inst_id = ann_dict['instance_id']
        new_id = reversed_mapping[inst_id]
        ann_dict['category_id'] = new_id
        img_count[new_id] += 1

    new_categories = []

    for cat_id in unique_object_ids_sorted[1:]:  # Exclude the background
    #for cat_id in unique_object_ids_sorted[:-1]:  # Exclude the background
        if cat_id in img_count.keys():
            new_cat_dict = dict(
                id=reversed_mapping[cat_id],
                name=f'Object{cat_id}',
                image_count=img_count[cat_id],
                instance_count=img_count[cat_id]
            )
        else:
            new_cat_dict = dict(
                id=reversed_mapping[cat_id],
                name=f'Object{cat_id}',
                image_count=0,
                instance_count=0
            )
        new_categories.append(new_cat_dict)
    ego_dataset['categories'] = new_categories
    ego_api._fix_frequencies()
    ego_api._create_index()


__all__ = [
    'make_instance_based'
]
