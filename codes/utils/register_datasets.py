import os
from itertools import product

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_datasets(dataset_names, json_dir, img_root):
    """Register the datasets of the same method. e.g. blurblend

    Args:
        dataset_names (iterable of str): desired name for dataset
        json_dir (str): path to dir of json files
        img_root (str): dir to images
    """

    # img_root = os.path.join(json_dir, 'frames')

    for d in dataset_names:
        # if 'data_aug' in json_dir:
        #     img_root.replace('frames', d)

        register_coco_instances(f"mbg_{d.lower()}", {},
                                os.path.join(json_dir, f"coco_format_{d}.json"), img_root)

        MetadataCatalog.get(f"mbg_{d.lower()}").set(thing_classes=[
            'tire',
            # 'pool',
            # 'bucket',
            # 'water_tank',
            # 'puddle',
            # 'bottle',
        ])

        cdc_metadata = MetadataCatalog.get(f"mbg_{d.lower()}")


def register_mosquitoes():
    this_filepath = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(this_filepath, '..', '..', 'data')

    #####

    subsets = ["train", "test", "set_1", "set_2", "set_3", "set_4"]
    original_json_dir = os.path.join(data_root, '_under_construction')
    original_imgs_dir = os.path.join(data_root, '_under_construction', 'frames')
    register_datasets(subsets, original_json_dir, original_imgs_dir)

    #####
    # register kfold datasets
    folds = [f'fold{k}' for k in range(3)]
    subsets = ["test", "set_1", "set_2", "set_3", "set_4"]
    dataset_names = list(product(folds, subsets))

    # dataset_names = ['_'.join(tups) for tups in dataset_names]
    # do the same as above: more readable
    dataset_names = list(map("_".join, dataset_names))

    original_json_dir = os.path.join(data_root, '_under_construction')
    original_imgs_dir = os.path.join(data_root, '_under_construction', 'frames')
    register_datasets(dataset_names, original_json_dir, original_imgs_dir)

    #####

    # data_aug_names = ["paste", "blend", "blurblend", "lum_blurblend"]
    folds = [f'fold{k}' for k in range(3)]
    subsets = ["set_1", "set_2", "set_3", "set_4"]

    data_aug_methods = [
        "lum_blurblend",
    ]
    for method in data_aug_methods:
        dataset_names = list(product([method], folds, subsets))
        dataset_names = list(map("_".join, dataset_names))

        aug_method_dir = os.path.join(data_root, 'data_aug', method)
        register_datasets(dataset_names, aug_method_dir, aug_method_dir)

    #####

    # register GAN dataset
    # TODO: kinda hardcoded. Could do any better?
    gan_path = os.path.join(data_root, 'data_aug', 'GAN', 'harmonized_frames')
    register_datasets(('gan_v0', ), gan_path, gan_path)


if __name__ == "__main__":
    register_mosquitoes()
