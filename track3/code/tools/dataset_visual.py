"""
A simple script to show how to load and visualize the dataset data.

The starting templates already define the correct data loading process.
You can modify this script to analyze/explore the dataset.
"""

from pathlib import Path

from ego_objects import EgoObjectsVis

from devkit_tools import ChallengeDetectionDataset

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image


def main():
    import matplotlib
    matplotlib.use('Agg')
    sample_root: Path = '/youtu/fuxi-team2-2/CLVision/submit_datasets'
    vis_Path = '/youtu/fuxi-team2-2/CLVision/submit_datasets/vis_train/'

    sample_dataset = ChallengeDetectionDataset(root=sample_root)
    ego_api = sample_dataset.ego_api

    print('Categories:', len(ego_api.get_cat_ids()))
    print('Images:', len(ego_api.get_img_ids()))
    print('Annotations:', len(ego_api.get_ann_ids()))

    ''' 
    # Example 1: dataset-agnostic simple plot from dataset output
    n_to_show = 5
    for img_idx in range(n_to_show):
        image, target = sample_dataset[img_idx]
        plot_sample(image, target, vis_Path + f'img_{img_idx}.jpg')
        plt.show()
        plt.clf()
    '''

    # Example 2: plot using EgoObjectsVis
    #ego_vis = EgoObjectsVis(ego_api, img_dir=str(sample_root / 'cltest'))
    #ego_vis = EgoObjectsVis(ego_api, img_dir=str(sample_root)+ '/' + 'cltest')
    ego_vis = EgoObjectsVis('/youtu/fuxi-team2-2/CLVision/submit_datasets/ego_objects_challenge_train.json', img_dir='/youtu/fuxi-team2-2/CLVision/submit_datasets/images')
    for img_id in ego_api.get_img_ids()[:5]:
        fig, _ = ego_vis.vis_img(img_id=img_id, show_boxes=True,
                                 show_classes=True)
        img = ego_api.load_imgs([img_id])[0]
        img_name = img["url"].split("/")[-1].replace('.jpg', '_vis.jpg')
        fig.savefig(vis_Path + img_name)
        #fig.savefig(vis_Path + f'img_{img_id}_vis.jpg')
        plt.close(fig)


def plot_sample(img: Image.Image, target, save_as=None):
    img_id = int(target['image_id'])
    plt.title(f'Image ID: {img_id}')

    plt.gca().imshow(img)
    for box in target['boxes']:
        box = box.tolist()

        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        plt.gca().add_patch(rect)

    if save_as is not None:
        plt.savefig(str(save_as))


if __name__ == '__main__':
    main()
