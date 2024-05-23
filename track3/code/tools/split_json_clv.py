import json
import  random
from collections import defaultdict

def splitImagesByRandom(images,annotations,sampled_img_num):
    images_list = []
    for img in images:
        images_list.append(img)
    random.shuffle(images_list)
    images_sampled = images_list[:int(sampled_img_num)]

    splited_images = {}
    ann_id_dicts = {}
    for ann in annotations: ann_id_dicts[ann['id']] = ann
    all_main_ids = []

    for img in images_sampled:
        id = img['id']
        main_id = img['main_category_instance_ids']
        if len(main_id) != 1: continue
        main_id = main_id[0]
        all_main_ids.append(main_id)
        tem_main_ann = ann_id_dicts[main_id]
        main_ann = {key:value for key,value in tem_main_ann.items()
                    if key != 'id'}
        splited_images[id] = {'info':img,
                            'annos':[main_ann]}

    new_annotations = []
    for ann in annotations: 
        if ann['id'] in all_main_ids: continue
        new_annotations.append(ann)
        
    return splited_images,new_annotations


def splitImagesByVid(images,annotations,sampled_img_num):
    video_ids = defaultdict(list)
    for img_info in images:
        group_id = img_info['group_id']
        video_id = img_info['video_id']
        vid = '_'.join([group_id,video_id])
        video_ids[vid].append(img_info)
    video_num = len(video_ids)
    print(f'video_num={video_num}')

    # shuffle videos
    v_ids = []
    for vid,image_dicts in video_ids.items():
        v_ids.append(vid)
    random.shuffle(v_ids)
    video_ids_shuffle = {}
    for v_id in v_ids:
        for vid,image_dicts in video_ids.items():
            if v_id == vid:
                video_ids_shuffle[vid] = image_dicts

    shuffle_video_num = len(video_ids_shuffle)
    print(f'shuffle_video_num={shuffle_video_num}')
    
    splited_images = {}
    ann_id_dicts = {}
    for ann in annotations: ann_id_dicts[ann['id']] = ann

    all_main_ids = []
    # splitImagesByVid
    video_sampled_img_num = int(sampled_img_num) // len(video_ids_shuffle)
    if video_sampled_img_num < 1:
        # sample one img per video
        video_sampled_count = 0
        for vid,image_dicts in video_ids_shuffle.items():
            video_sampled_count += 1
            if video_sampled_count > sampled_img_num: break
            random.shuffle(image_dicts)
            images_all = image_dicts[:1]

            for img in images_all:
                id = img['id']
                main_id = img['main_category_instance_ids']
                if len(main_id) != 1: continue
                main_id = main_id[0]
                all_main_ids.append(main_id)
                tem_main_ann = ann_id_dicts[main_id]
                main_ann = {key:value for key,value in tem_main_ann.items()
                            if key != 'id'}
                splited_images[id] = {'info':img,
                                    'annos':[main_ann]}
    else:
        # sample multiple imgs per video
        sampled_nums = video_sampled_img_num * len(video_ids_shuffle)
        video_sampled_count = 0
        for vid,image_dicts in video_ids_shuffle.items():
            video_sampled_count += 1
            random.shuffle(image_dicts)
            if video_sampled_count <= (sampled_img_num - sampled_nums):
                images_all = image_dicts[:int(video_sampled_img_num)+1]
            else:
                images_all = image_dicts[:int(video_sampled_img_num)]

            for img in images_all:
                id = img['id']
                main_id = img['main_category_instance_ids']
                if len(main_id) != 1: continue
                main_id = main_id[0]
                all_main_ids.append(main_id)
                tem_main_ann = ann_id_dicts[main_id]
                main_ann = {key:value for key,value in tem_main_ann.items()
                            if key != 'id'}
                splited_images[id] = {'info':img,
                                    'annos':[main_ann]}

    new_annotations = []
    for ann in annotations: 
        if ann['id'] in all_main_ids: continue
        new_annotations.append(ann)
        
    return splited_images,new_annotations



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
                  data_info,save_file=None):
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
    if save_file!=None:
        json.dump(json_info,
                open(save_file, 'w'), indent=4)
    
    return json_info


def split_json_clv(json_file, sampled_img_num, splited_json_path=None, split_mode='splitImagesByRandom'):
    data = json.load(open(json_file))
    images,annotations = data['images'],data['annotations']
    data_info, categories = data['info'],data['categories']

    if split_mode=='splitImagesByRandom':
        splited_images,new_annotations = splitImagesByRandom(images,annotations,sampled_img_num=sampled_img_num)
    elif split_mode=='splitImagesByVid':
        splited_images,new_annotations = splitImagesByVid(images,annotations,sampled_img_num=sampled_img_num)

    img_num = len(splited_images)
    print(f'sampled_img_num={img_num}')
    
    for ann in new_annotations:
        image_id = ann['image_id']
        _ann = {key:value for key,value in ann.items()
                if key != 'id'}
        if image_id in splited_images.keys():
            splited_images[image_id]['annos'].append(_ann)

    splited_new_categories = gen_cat_info(splited_images,categories)
    splited_json_info = gen_json_info(splited_images,splited_new_categories,data_info,save_file=splited_json_path)
    return splited_json_info


if __name__ == '__main__':
    json_file = '/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_trainSplit_submit_exp0.json'
    sampled_img_num = 3500
    splited_json_path = '/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_trainSplit_submit_exp0_split.json'
    splited_json_info = split_json_clv(json_file, sampled_img_num, splited_json_path=splited_json_path, split_mode='splitImagesByRandom')














