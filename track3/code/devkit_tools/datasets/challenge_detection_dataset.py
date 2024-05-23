###
# Adapted from Avalanche LvisDataset
# https://github.com/ContinualAI/avalanche/tree/detection/avalanche/benchmarks/datasets/lvis
#
# Released under the MIT license, see:
# https://github.com/ContinualAI/avalanche/blob/master/LICENSE
###

from pathlib import Path
from typing import List, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset
from mmdet.datasets.custom import CustomDataset
from torchvision.datasets.folder import default_loader

from devkit_tools.challenge_constants import DEFAULT_CHALLENGE_TRAIN_JSON, \
    DEFAULT_CHALLENGE_TEST_JSON
from ego_objects import EgoObjects, EgoObjectsAnnotation, \
    EgoObjectsImage
import torch
import numpy as np
import json

class ChallengeDetectionDataset(Dataset):
    """
    The sample dataset. For internal use by challenge organizers only.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        train=True,
        transform=None,
        loader=default_loader,
        ego_api=None,
        img_ids: List[int] = None,
        bbox_format: str = 'ltwh',
        categories_id_mapping: List[int] = None,
        eval_submit = False,
    ):
        """
        Instantiates the sample dataset.

        :param root: The path to the images and annotation file.
        :param transform: The transformation to apply.
        :param loader: The image loader. Defaults to PIL Image open.
        :param ego_api: An EgoObjects object. If not provided, annotations
            will be loaded from the json file found in the root. Defaults to
            None.
        :param img_ids: A list of image ids to use. If not None, only those
            images (a subset of the original dataset) will be used. Defaults
            to None.
        :param bbox_format: The bounding box format. Defaults to "ltwh"
            (Left, Top, Width, Height).
        :param categories_id_mapping: If set, it must define a mapping from
            the to-be-used-id to the real category id so that:
            real_cat_id = categories_id_mapping[mapped_id].
        """
        self.root: Path = Path(root)
        self.train = train
        self.transform = transform
        self.loader = loader
        self.bbox_crop = True
        self.img_ids = img_ids
        self.bbox_format = bbox_format
        self.categories_id_mapping = categories_id_mapping

        self.ego_api = ego_api

        must_load_api = self.ego_api is None
        must_load_img_ids = self.img_ids is None

        # Load metadata
        if must_load_api:
            if self.train:
                ann_json_path = str(self.root / DEFAULT_CHALLENGE_TRAIN_JSON)
            else:
                ann_json_path = str(self.root / DEFAULT_CHALLENGE_TEST_JSON)
            if eval_submit:
                ann_json_path = str(self.root / 'ego_objects_challenge_test.json')

            self.ego_api = EgoObjects(ann_json_path)

        if must_load_img_ids:
            self.img_ids = list(sorted(self.ego_api.get_img_ids()))

        self.targets = EgoObjectsDetectionTargets(
            self.ego_api, self.img_ids,
            categories_id_mapping=categories_id_mapping)

        # Try loading an image
        if len(self.img_ids) > 0:
            img_id = self.img_ids[0]
            img_dict = self.ego_api.load_imgs(ids=[img_id])[0]
            assert self._load_img(img_dict) is not None

        if self.train: 
            self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.ego_api.load_imgs(ids=[self.img_ids[i]])[0]

            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        annotation_dicts = self.targets[index]
        num_objs = len(annotation_dicts)
        category_ids = []
        for i in range(num_objs):
            category_ids.append(annotation_dicts[i]['category_id'])
        return set(category_ids)

    def __getitem__(self, index):
        """
        Loads an instance given its index.

        :param index: The index of the instance to retrieve.

        :return: a (sample, target) tuple where the target is a
            torchvision-style annotation for object detection
            https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """
        img_id = self.img_ids[index]
        img_dict: EgoObjectsImage = self.ego_api.load_imgs(ids=[img_id])[0]
        annotation_dicts = self.targets[index]

        # Transform from EgoObjects dictionary to torchvision-style target
        num_objs = len(annotation_dicts)

        boxes = []
        labels = []
        areas = []
        for i in range(num_objs):
            xmin = annotation_dicts[i]['bbox'][0]
            ymin = annotation_dicts[i]['bbox'][1]
            if self.bbox_format == 'ltrb':
                # Left, Top, Right, Bottom
                xmax = annotation_dicts[i]['bbox'][2]
                ymax = annotation_dicts[i]['bbox'][3]

                boxw = xmax - xmin
                boxh = ymax - ymin
            else:
                # Left, Top, Width, Height
                boxw = annotation_dicts[i]['bbox'][2]
                boxh = annotation_dicts[i]['bbox'][3]

                xmax = boxw + xmin
                ymax = boxh + ymin

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(annotation_dicts[i]['category_id'])
            areas.append(boxw * boxh)

        
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
        else:
            boxes = np.empty((0, 4),dtype=np.float32)

        labels = np.array(labels, dtype=np.int64)
        image_id = np.array([img_id])
        areas = np.array(areas,dtype=np.float32)

        img = self._load_img(img_dict)

        target = dict()
        target['img_prefix'] = ''
        target['filename'] = img
        target['img_info'] = {'filename':img}
        target["image_id"] = image_id
        target['bbox_fields'] = []
        target['ann_info'] = {}
        target['ann_info']['bboxes'] = boxes
        target['ann_info']['labels'] = labels
        target['ann_info']["area"] = areas
        target['ann_info']['labels_ignore'] = np.empty((0, 4))

        if self.transform is not None:
            target = self.transform(target)
        return target


    def __len__(self):
        return len(self.img_ids)

    def _load_img(self, img_dict):
        img_url = img_dict['url']
        splitted_url = img_url.split('/')
        img_path = 'images/' + splitted_url[-1]
        img_path_alt = 'cltest/' + splitted_url[-1]

        final_path = self.root / img_path  # <root>/images/<img_id>.jpg
        if not final_path.exists():
            final_path = self.root / img_path_alt
        #return Image.open(str(final_path)).convert("RGB")
        return str(final_path)

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
    
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        json_results = self.detection_exporter(results)

        with open('./test.json', 'w') as f:
            json.dump(json_results, f)
            #json.dump(json_results, f, cls=TensorEncoder)

        return {},None

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
            json_results[img_id]={"boxes": bboxes[:,:-1].tolist(),
                                  "scores":bboxes[:,-1].tolist(),
                                  "labels":labels.tolist()}
        return json_results



class EgoObjectsDetectionTargets(Sequence[List[EgoObjectsAnnotation]]):
    def __init__(
            self,
            ego_api: EgoObjects,
            img_ids: List[int] = None,
            categories_id_mapping: List[int] = None):
        super(EgoObjectsDetectionTargets, self).__init__()
        self.ego_api = ego_api

        if categories_id_mapping is not None:
            self.reversed_mapping = dict()
            for mapped_id, real_id in enumerate(categories_id_mapping):
                self.reversed_mapping[real_id] = mapped_id
        else:
            self.reversed_mapping = None

        if img_ids is None:
            img_ids = list(sorted(ego_api.get_img_ids()))

        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        annotation_ids = self.ego_api.get_ann_ids(img_ids=[img_id])
        annotation_dicts: List[EgoObjectsAnnotation] = \
            self.ego_api.load_anns(annotation_ids)

        if self.reversed_mapping is None:
            return annotation_dicts

        mapped_anns: List[EgoObjectsAnnotation] = []
        for ann_dict in annotation_dicts:
            ann_dict: EgoObjectsAnnotation = dict(ann_dict)
            ann_dict['category_id'] = \
                self.reversed_mapping[ann_dict['category_id']]
            mapped_anns.append(ann_dict)

        return mapped_anns


__all__ = [
    'ChallengeDetectionDataset'
]
