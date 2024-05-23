import json
import copy
from pathlib import Path
from collections import defaultdict


def merge_img_anno(data1, data2):
    merged_images = copy.deepcopy(data1['images'])
    merged_annotations = copy.deepcopy(data1['annotations'])

    for img in data2['images']:
        merged_images.append(copy.deepcopy(img))

    for anno in data2['annotations']:
        merged_annotations.append(copy.deepcopy(anno))

    return merged_images, merged_annotations


def gen_cat_info(merged_annotations, data1, data2):
    # obtain unique 'id' and 'name' in data1['categories']+data2['categories']
    merged_categories = copy.deepcopy(data1['categories'])
    data1_cat_id = []
    for cat in data1['categories']:
        data1_cat_id.append(cat['id'])
    for cat in data2['categories']:
        if cat['id'] in data1_cat_id: continue
        merged_categories.append(cat)

    # collect annos for each image
    images_annos = {}
    for anno in merged_annotations:
        image_id = anno['image_id']
        if image_id in images_annos.keys():
            images_annos[image_id]['annos'].append(anno)
        else:
            images_annos[image_id] = {'annos':[anno]}

    # generate category info
    cat_img_cnt, cat_ins_cnt = defaultdict(int),defaultdict(int)
    for image_id, image_info in images_annos.items():
        cat_ind = []
        for ann in image_info['annos']:
            cat_ind.append(ann['category_id'])

        for ind1 in cat_ind:
            cat_ins_cnt[ind1] += 1

        set_cat_ind = list(set(cat_ind))
        for ind2 in set_cat_ind:
            cat_img_cnt[ind2] += 1

    new_merged_categories = []
    for cat in merged_categories:
        ind = cat['id']
        img_cnt,ins_cnt = cat_img_cnt[ind],cat_ins_cnt[ind]
        #if img_cnt==0: continue
        if img_cnt<10: frequency = 'r'
        elif img_cnt<100: frequency = 'c'
        else: frequency = 'f'
        cat_info = {'id':ind,
                    'name':cat['name'],
                    'image_count':img_cnt,
                    'instance_count':ins_cnt,
                    'frequency':frequency}

        new_merged_categories.append(copy.deepcopy(cat_info))

    return new_merged_categories


def merge_json_clv(json_path1, json_path2, merged_json_path=None):
    print(f'loading {json_path1}.')
    print(f'loading {json_path2}.')
    data1 = json.load(open(json_path1))
    data2 = json.load(open(json_path2))

    # re-order: 'main_category_instance_ids' in data2['images']
    # re-order: 'id' in data2['annotations']
    data1_max_anno_id = 0
    for anno in data1['annotations']:
        if anno['id'] > data1_max_anno_id:
            data1_max_anno_id = anno['id']
    data1_max_anno_id = data1_max_anno_id + 10
    print(f'data1_max_anno_id={data1_max_anno_id}')
    for idx in range(len(data2['annotations'])):
        data2['annotations'][idx]['id'] += data1_max_anno_id
    for idx in range(len(data2['images'])):
        for main_idx in range(len(data2['images'][idx]['main_category_instance_ids'])):
            data2['images'][idx]['main_category_instance_ids'][main_idx] += data1_max_anno_id

    merged_images, merged_annotations = merge_img_anno(data1, data2)
    merged_categories = gen_cat_info(merged_annotations, data1, data2)

    # generate json info
    merged_json_info = {'info': data1['info'],
                        'categories': merged_categories,
                        'images':merged_images,
                        'annotations':merged_annotations}
    if merged_json_path!=None:
        json.dump(merged_json_info, open(merged_json_path, 'w'), indent=4)

    return merged_json_info


if __name__ == "__main__":
    json_path1 = '/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_trainSplit_submit_exp0.json'
    json_path2 = '/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_trainSplit_submit_exp1.json'
    merged_json_path = '/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_trainSplit_submit_exp01.json'
    merged_json_info = merge_json_clv(json_path1, json_path2, merged_json_path=merged_json_path)
