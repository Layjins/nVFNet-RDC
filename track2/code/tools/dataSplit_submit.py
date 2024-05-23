#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : dataSplit.py
# @Author: niceliu
# @Date  : 2022/4/6 9:55
# @Desc  :

import json
import  random
from collections import defaultdict
from devkit_tools.challenge_constants import \
    DEFAULT_DEMO_TEST_JSON, \
    DEFAULT_DEMO_TRAIN_JSON, DEMO_DETECTION_EXPERIENCES, \
    CHALLENGE_DETECTION_EXPERIENCES, CHALLENGE_DETECTION_FORCED_TRANSFORMS, \
    DEFAULT_CHALLENGE_TRAIN_JSON, DEFAULT_CHALLENGE_TEST_JSON, \
    DEFAULT_CHALLENGE_CLASS_ORDER_SEED, DATASET_PATH

def splitImagesByVid(images,annotations,ratio):

    video_ids = defaultdict(list)
    for img_info in images:
        group_id = img_info['group_id']
        video_id = img_info['video_id']
        vid = '_'.join([group_id,video_id])
        video_ids[vid].append(img_info)

    train_images,val_images = {}, {}
    ann_id_dicts = {}
    for ann in annotations: ann_id_dicts[ann['id']] = ann

    all_main_ids = []
    for vid,image_dicts in video_ids.items():
        random.shuffle(image_dicts)
        size = len(image_dicts)
        #print(vid,size)
        train,val = image_dicts[:int(size*ratio)],\
                    image_dicts[int(size*ratio):]

        # train
        for img in train:
            id = img['id']
            main_id = img['main_category_instance_ids']
            if len(main_id) != 1: continue
            main_id = main_id[0]
            all_main_ids.append(main_id)
            tem_main_ann = ann_id_dicts[main_id]
            main_ann = {key:value for key,value in tem_main_ann.items()
                        if key != 'id'}
            train_images[id] = {'info':img,
                                'annos':[main_ann]}

        # val
        for img in val:
            id = img['id']
            main_id = img['main_category_instance_ids']
            if len(main_id) != 1: continue
            main_id = main_id[0]
            all_main_ids.append(main_id)
            tem_main_ann = ann_id_dicts[main_id]
            main_ann = {key:value for key,value in tem_main_ann.items()
                        if key != 'id'}
            val_images[id] = {'info':img,
                                'annos':[main_ann]}
    new_annotations = []
    for ann in annotations: 
        if ann['id'] in all_main_ids: continue
        new_annotations.append(ann)
        
    return train_images,val_images,new_annotations



def gen_cat_info(image_dicts,categories):


    cat_img_cnt, cat_ins_cnt = defaultdict(int),defaultdict(int)

    for image_id,image_info in image_dicts.items():
        cat_ind = []
        for ann in image_info['annos']:
            cat_ind.append(ann['category_id'])

        for ind1 in cat_ind:
            cat_ins_cnt[ind1] += 1

        set_cat_ind = list(set(cat_ind))
        for ind2 in set_cat_ind:
            cat_img_cnt[ind2] += 1


    new_categories = []

    for cat in categories:
        ind = cat['id']
        img_cnt,ins_cnt = cat_img_cnt[ind],cat_ins_cnt[ind]
        if img_cnt<10: frequency = 'r'
        elif img_cnt<100: frequency = 'c'
        else: frequency = 'f'
        cat_info = {'id':ind,
                    'name':cat['name'],
                    'image_count':img_cnt,
                    'instance_count':ins_cnt,
                    'frequency':frequency}

        new_categories.append(cat_info)

    return new_categories


def gen_json_info(image_dicts,new_categories,
                  data_info,save_file):

    images = []
    annotations = []

    cnt = 1
    for image_id,image_info in image_dicts.items():
        img_info = image_info['info']
        img_info['main_category_instance_ids'] = [cnt]
        images.append(img_info)
        for ann in image_info['annos']:
            ann.update({'id':cnt})
            annotations.append(ann)
            cnt += 1

    json_info = {'info': data_info,
                 'categories': new_categories,
                 'images':images,
                 'annotations':annotations}

    json.dump(json_info,
              open(save_file, 'w'), indent=4)




if __name__ == '__main__':
    # path to ego_objects_challenge_train.json
    #DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
    DEFAULT_CHALLENGE_TRAIN_JSON = 'ego_objects_challenge_train.json'
    json_file = DATASET_PATH + '/' + DEFAULT_CHALLENGE_TRAIN_JSON
    print(f"train: {json_file}")
   
    data = json.load(open(json_file))
    images,annotations = data['images'],data['annotations']
    data_info, categories = data['info'],data['categories']

    train_images,val_images,new_annotations = splitImagesByVid(images,
                                               annotations,
                                               ratio = 0.9)

    #print(len(train_images),len(val_images))
    print(f"train_images={len(train_images)},val_images={len(val_images)}")
    
    for ann in new_annotations:
        image_id = ann['image_id']
        _ann = {key:value for key,value in ann.items()
                if key != 'id'}
        if image_id in train_images.keys():
            train_images[image_id]['annos'].append(_ann)
        else:
            val_images[image_id]['annos'].append(_ann)

    train_new_categories = gen_cat_info(train_images,categories)
    train_json_file = json_file.replace('ego_objects_challenge_train.json', 'ego_objects_challenge_trainSplit_track2.json')
    gen_json_info(train_images,train_new_categories, data_info, train_json_file)
    print(f"trainSplit: {train_json_file}")

    val_new_categories = gen_cat_info(val_images,categories)
    val_json_file = json_file.replace('ego_objects_challenge_train.json', 'ego_objects_challenge_valSplit_track2.json')
    gen_json_info(val_images,val_new_categories, data_info, val_json_file)
    print(f"valSplit: {val_json_file}")















