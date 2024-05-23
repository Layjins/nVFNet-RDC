import json
from pathlib import Path
from collections import defaultdict
from ego_objects import EgoObjectsVis
from devkit_tools import ChallengeDetectionDataset
from devkit_tools.benchmarks import challenge_category_detection_benchmark


DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
DEFAULT_CHALLENGE_TRAIN_JSON = 'ego_objects_challenge_trainSplit_submit.json'
DEFAULT_CHALLENGE_TEST_JSON = 'ego_objects_challenge_valSplit_submit.json'
CHALLENGE_DETECTION_EXPERIENCES = 5


def gen_images_from_dataset(images, annotations, dataset):
    dataset_images = {}

    img_id_dicts = {}
    for img in images: img_id_dicts[img['id']] = img

    ann_id_dicts = {}
    for ann in annotations: ann_id_dicts[ann['id']] = ann

    all_main_ids = []
    for idx in range(len(dataset)):
        dataset_img_info = dataset.__getitem__(idx)
        img = img_id_dicts[dataset_img_info["image_id"][0]]
        id = img['id']
        main_id = img['main_category_instance_ids']
        if len(main_id) != 1: continue
        main_id = main_id[0]
        all_main_ids.append(main_id)
        tem_main_ann = ann_id_dicts[main_id]
        main_ann = {key:value for key,value in tem_main_ann.items()
                    if key != 'id'}
        dataset_images[id] = {'info':img,
                            'annos':[main_ann]}

    new_annotations = []
    for ann in annotations:
        if ann['id'] in all_main_ids: continue
        new_annotations.append(ann)

    return dataset_images, new_annotations

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
        #if img_cnt==0: continue
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

def main():
    print(f'loading benchmark ...')
    benchmark = challenge_category_detection_benchmark(
        dataset_path=DATASET_PATH,
        train_transform=None,
        eval_transform=None,
        train_json_name=DEFAULT_CHALLENGE_TRAIN_JSON,
        test_json_name=DEFAULT_CHALLENGE_TEST_JSON,
        n_exps=CHALLENGE_DETECTION_EXPERIENCES
    )

    cls_num = benchmark.n_classes
    print(f'Benchmark class number : {cls_num}') 
    
    # TRAINING LOOP
    print(f'Starting experiment...')
    #val_dataset = benchmark.test_stream[0].dataset

    print(f'loading annotations ...')
    json_file = DATASET_PATH + '/' + DEFAULT_CHALLENGE_TRAIN_JSON
    data = json.load(open(json_file))
    images, annotations = data['images'], data['annotations']
    data_info, categories = data['info'], data['categories']

    for experience in benchmark.train_stream:
        exp_id = experience.current_experience
        dataset_size = len(experience.dataset)
        print(f'Start of experience : {exp_id}, train dataset: {dataset_size}') 
        #print(experience.dataset.__getitem__(0))
    
        dataset_images, new_annotations = gen_images_from_dataset(images, annotations, experience.dataset)                                          
        print('generated train dataset: ', len(dataset_images))

        for ann in new_annotations:
            image_id = ann['image_id']
            _ann = {key:value for key,value in ann.items()
                    if key != 'id'}
            if image_id in dataset_images.keys():
                dataset_images[image_id]['annos'].append(_ann)

        dataset_new_categories = gen_cat_info(dataset_images,categories)
        dataset_json_file = json_file.replace('.json', '_exp' + str(exp_id)+'.json')
        gen_json_info(dataset_images,dataset_new_categories,
                    data_info,dataset_json_file)



if __name__ == "__main__":
    main()
