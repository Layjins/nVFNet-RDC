import json
from pathlib import Path
from ego_objects import EgoObjectsVis
from devkit_tools import ChallengeDetectionDataset

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    # analysis  flag
    wh_ratio_flag = False
    train_test_group_overlap_flag = True

    ############### ego_objects api #######################    
    DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
    sample_dataset = ChallengeDetectionDataset(root=DATASET_PATH) # DEFAULT_CHALLENGE_TRAIN_JSON = 'ego_objects_challenge_train.json'
    #sample_dataset = ChallengeDetectionDataset(root=DATASET_PATH, train=False) # DEFAULT_CHALLENGE_TEST_JSON = 'ego_objects_challenge_test.json'
    ego_api = sample_dataset.ego_api

    print('Categories:', len(ego_api.get_cat_ids()))
    print('Images:', len(ego_api.get_img_ids()))
    print('Annotations:', len(ego_api.get_ann_ids()))

    print('Categories id:')
    print(ego_api.get_cat_ids())

    print('Start analysis ...')
    # wh_ratio
    if wh_ratio_flag:
        box_wh = []
        for img_idx in range(len(ego_api.get_img_ids())):
            image, target = sample_dataset[img_idx]
            for box in target['boxes']:
                box = box.tolist()
                boxw = box[2]
                boxh = box[3]
                wh = round(boxw/boxh, 0)
                if wh <1 :
                    wh = round(boxh/boxw, 0)
                box_wh.append(wh)
        box_wh_unique = list(set(box_wh))
        box_wh_count=[box_wh.count(i) for i in box_wh_unique]
        print('Box wh_ratio:')
        print(box_wh_count)
    
    ############### loading json annotations #######################  
    train_anno_path = DATASET_PATH + '/' + 'ego_objects_challenge_train.json'
    test_anno_path = DATASET_PATH + '/' + 'ego_objects_challenge_test.json'
    # train_test_group_overlap
    if train_test_group_overlap_flag:
        # train set
        print("Loading train annotations...")
        train_anno = _load_json(train_anno_path)
        train_group_id = []
        train_img_wh = []
        for img_anno in train_anno["images"]:
            img_group_id = img_anno["group_id"]
            img_url = img_anno["url"]
            img_file_name = img_url.split("/")[-1]
            img_h = img_anno["height"]
            img_w = img_anno["width"]
            img_wh = '(' + str(img_w) + ',' + str(img_h) + ')'

            train_group_id.append(img_group_id)
            train_img_wh.append(img_wh)
        train_group_id_unique = list(set(train_group_id))
        print('len of train_group_id_unique:')
        print(len(train_group_id_unique))
        #print('train_group_id_unique:')
        #print(train_group_id_unique)
        train_img_wh_unique = list(set(train_img_wh))
        train_img_wh_count=[train_img_wh.count(i) for i in train_img_wh_unique]
        print('train_img_wh_unique:')
        print(train_img_wh_unique)
        print('train_img_wh_count:')
        print(train_img_wh_count)

        # test set
        print("Loading test annotations...")
        test_anno = _load_json(test_anno_path)
        test_group_id = []
        for img_anno in test_anno["images"]:
            img_url = img_anno["url"]
            img_file_name = img_url.split("/")[-1]
            img_group_id = img_file_name.split("_")[0]

            test_group_id.append(img_group_id)
        test_group_id_unique = list(set(test_group_id))
        print('len of test_group_id_unique:')
        print(len(test_group_id_unique))
        #print('test_group_id_unique:')
        #print(test_group_id_unique)

        # train and test overlap
        train_test_group_overlap_id = []
        for test_group in test_group_id_unique:
            if test_group in train_group_id_unique:
                train_test_group_overlap_id.append(test_group)
        print('len of train_test_group_overlap_id:')
        print(len(train_test_group_overlap_id))
        #print('train_test_group_overlap_id:')
        #print(train_test_group_overlap_id)

    print('Dataset analysis finished.')

    
if __name__ == "__main__":
    main()
