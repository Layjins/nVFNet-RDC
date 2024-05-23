
from pathlib import Path
import json

#from avalanche.benchmarks import NCScenario
from devkit_tools.benchmarks.classification_benchmark import \
    challenge_classification_benchmark
from devkit_tools.challenge_constants import DEFAULT_CHALLENGE_CLASS_ORDER_SEED
from devkit_tools.challenge_constants import \
    DEFAULT_CHALLENGE_TEST_JSON, DEFAULT_CHALLENGE_TRAIN_JSON, CHALLENGE_DETECTION_EXPERIENCES
from ego_objects import EgoObjects

from pycocotools.coco import COCO

# TODO: change this to the path where you downloaded (and extracted) the dataset along with your annotations
DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/track2_datasets'
#DATASET_PATH = '/youtu/fuxi-team2-2/CLVision/demo_dataset'
train_json_name = DEFAULT_CHALLENGE_TRAIN_JSON
test_json_name = DEFAULT_CHALLENGE_TEST_JSON

# Don't change this (unless you want to experiment with different class orders)
# Note: it won't be possible to change the class order in the real challenge
CLASS_ORDER_SEED = DEFAULT_CHALLENGE_CLASS_ORDER_SEED

def main():
    print('Loading Dataset......')
    cls_benchmark =challenge_classification_benchmark(
        DATASET_PATH,
        class_order_seed=CLASS_ORDER_SEED,
        train_json_name=train_json_name,
        test_json_name=test_json_name,
        instance_level=False,
        n_exps=CHALLENGE_DETECTION_EXPERIENCES,
        unlabeled_test_set=False
    )
    # Create aligned datasets
    train_ego_api = EgoObjects(str(Path(DATASET_PATH) / train_json_name))
    test_ego_api = EgoObjects(str(Path(DATASET_PATH) / test_json_name))
    print('Start analyzing......')
    # Dataset analsis
    rare_cat_train = 0
    common_cat_train = 0
    frequent_cat_train = 0
    common_cat_test = 0
    rare_cat_test = 0
    frequent_cat_test = 0
    for cat in train_ego_api.dataset["categories"]:
        if cat['frequency'] == 'r':
            rare_cat_train += 1
        elif cat['frequency'] == 'c':
            common_cat_train += 1
        else:
            frequent_cat_train += 1
    for cat in test_ego_api.dataset["categories"]:
        if cat['frequency'] == 'r':
            rare_cat_test += 1
        elif cat['frequency'] == 'c':
            common_cat_test += 1
        else:
            frequent_cat_test += 1
    print('rare_cat_train: ' + str(rare_cat_train))
    print('common_cat_train: ' + str(common_cat_train))
    print('frequent_cat_train: ' + str(frequent_cat_train))
    print('rare_cat_test: ' + str(rare_cat_test))
    print('common_cat_test: ' + str(common_cat_test))
    print('frequent_cat_test: ' + str(frequent_cat_test))

    print('Total number of experiences: ', cls_benchmark.n_experiences)
    print('Number of main classes per experience: ', cls_benchmark.n_classes_per_exp)
    print('Original main classes per experience: ', cls_benchmark.original_classes_in_exp)
    for i, assignment in enumerate(cls_benchmark.train_exps_patterns_assignment):
        print('Number of training images in experience %d: %d' % (i+1, len(assignment)))
    
    for i, assignment in enumerate(cls_benchmark.test_exps_patterns_assignment):
        print('Number of test images in experience %d: %d' % (i+1, len(assignment)))

    coco = COCO(str(Path(DATASET_PATH) / train_json_name))
    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    imgIds = coco.getImgIds()

    coco_test = COCO(str(Path(DATASET_PATH) / test_json_name))
    imgIds_test = coco_test.getImgIds()

    print('Number of classes (including background)', len(cats) + 1)
    print('Number of images in training set: ', len(imgIds))
    print('Number of images in test set: ', len(imgIds_test))

    cat_num2name = {}
    for cat in cats:
        cat_num2name[cat['id']] = cat['name']

    cate_num = {}
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        main_category = img['main_category']
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for k in range(len(anns)):
            cat_id = anns[k]['category_id']
            cat_name = cat_num2name[cat_id]
            if cat_name != main_category:
                cate_num[main_category] = cate_num.get(main_category, {})
                cate_num[main_category][cat_name] = cate_num[main_category].get(cat_name, 0) + 1
    with open('./relation_train.json', "w") as f:
        json.dump(cate_num, f, indent=4)

    cate_num = {}
    for i in range(len(imgIds_test)):
        img = coco_test.loadImgs(imgIds_test[i])[0]
        main_category = img['main_category']
        annIds = coco_test.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_test.loadAnns(annIds)
        for k in range(len(anns)):
            cat_id = anns[k]['category_id']
            cat_name = cat_num2name[cat_id]
            if cat_name != main_category:
                cate_num[main_category] = cate_num.get(main_category, {})
                cate_num[main_category][cat_name] = cate_num[main_category].get(cat_name, 0) + 1
    with open('./relation_test.json', "w") as f:
        json.dump(cate_num, f, indent=4)

    print('Dataset analysis finished. Class corelation has been saved in current directory.')

    
if __name__ == "__main__":
    main()
 
