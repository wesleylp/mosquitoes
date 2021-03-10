import argparse
import fnmatch
import json
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from annotation import AnnotationImage
from boxes import box_area, clip_box_to_image, xyxy_to_xywh


class Convert2Coco:
    category_dict = {
        # '__background__': 0,
        'tire': 0,
        # 'pool': 1,
        # 'bucket': 2,
        # 'water_tank': 3,
        # 'puddle': 4,
        # 'bottle': 5,
    }

    def __init__(self):
        self._reset()

    def _reset(self):
        self.ann_dict = {}
        self.images = []
        self.annotations = []

        self.img_id = 0
        self.ann_id = 0

        categories = [{"id": self.category_dict[name], "name": name} for name in self.category_dict]
        self.ann_dict['categories'] = categories

    def update(self, filename, width, height, gt):
        image = {}

        image['file_name'] = filename
        image['width'] = width
        image['height'] = height
        image['id'] = self.img_id
        self.img_id += 1

        for label, box in gt.items():
            obj_name = label.split('-')[0]
            if obj_name not in list(self.category_dict.keys()):
                continue

            ann = {}
            box = clip_box_to_image(box, height, width)

            ann['image_id'] = image['id']
            ann['id'] = self.ann_id
            self.ann_id += 1

            ann['category_id'] = self.category_dict[obj_name]
            ann['iscrowd'] = 0
            ann['area'] = box_area(box)
            ann['bbox'] = xyxy_to_xywh(box)

            self.annotations.append(ann)

        self.images.append(image)

        self.ann_dict['images'] = self.images
        self.ann_dict['annotations'] = self.annotations

    def export(self, output_path):
        with open(output_path, 'w') as outfile:
            outfile.write(json.dumps(self.ann_dict))


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    # parser.add_argument('--dataset', help="cocostuff, cityscapes", default=None, type=str)
    parser.add_argument(
        '--outdir', help="output dir for png files", default='data/_under_construction', type=str)

    parser.add_argument(
        '--anndir',
        help="annotation dir for txt files",
        default='data/_under_construction',
        type=str)

    parser.add_argument(
        '--datadir',
        help="data dir for annotations to be converted",
        default='data/_under_construction',
        type=str)

    parser.add_argument(
        '--file_set',
        help="file path for train sets separation",
        default='train_sets_kfold.xls',
        type=str)

    parser.add_argument('--folds', help="number of folds", default=3, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def convert_mosquitoes_instance_only(data_dir, ann_dir, out_dir, file_set, folds=1):

    subsets = ['set_1', 'set_2', 'set_3', 'set_4', 'test']

    sets = product(np.arange(folds, dtype=np.int8), subsets)

    json_name = 'coco_format_fold%s_%s.json'
    if 'kfold' in file_set:
        df_set = {
            k: pd.read_excel(file_set, sheet_name=f'train_sets_k{k}')
            for k in np.arange(folds)
        }

    else:
        df_set = {0: pd.read_csv(file_set)}

    category_dict = {
        #'__background__': 0,
        'tire': 0,
        # 'pool': 1,
        # 'bucket': 2,
        # 'water_tank': 3,
        # 'puddle': 4,
        # 'bottle': 5,
    }

    annot_stats = {}

    #TODO: Convert the code below using the class above!
    for fold, data_set in tqdm(sets):
        print(f'Starting {data_set}')

        videos_set = df_set[fold][df_set[fold][data_set] == True]['Video'].tolist()

        ann_dict = {}
        images = []
        annotations = []

        # ann_dir = 'zframer-marcacoes'

        img_id = 0
        ann_id = 0
        # cat_id = 1

        annot_stats.update({data_set: {}})

        for (dirpath, dirnames, filenames) in os.walk(os.path.join(data_dir, 'frames')):

            if len(filenames) == 0:
                continue

            video_name = dirpath.split('/')[-1]
            if video_name not in videos_set:
                continue

            annot_stats[data_set].update({dirpath: {}})
            annot_stats[data_set][dirpath].update({obj: 0 for obj in category_dict.keys()})

            print(dirpath)
            for filename in filenames:
                if filename.lower().endswith(('.png')):
                    image = {}

                    img_name_short = os.path.join(os.path.split(dirpath)[-1], filename)
                    img_name = os.path.join(dirpath, filename)

                    width = get_img_size(img_name)[0]
                    height = get_img_size(img_name)[1]

                    image['file_name'] = img_name_short
                    # image['seg_file_name'] = img_name
                    image['width'] = width
                    image['height'] = height
                    image['id'] = img_id
                    img_id += 1

                    for box, label in zip(*get_groundtruth(img_name, ann_dir)):
                        # trim bbox for img limits

                        box = clip_box_to_image(box, height, width)
                        obj_name = label.split('-')[0]

                        if obj_name != 'tire':
                            continue

                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']

                        ann['category_id'] = category_dict[obj_name]
                        ann['iscrowd'] = 0
                        ann['area'] = box_area(box)
                        ann['bbox'] = xyxy_to_xywh(box)
                        # just to save segmentation
                        # ann['segmentation'] = ann['bbox']

                        annot_stats[data_set][dirpath][obj_name] += 1

                        annotations.append(ann)
                    images.append(image)

        df = pd.DataFrame(annot_stats[data_set]).T
        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        print(df)

        with open(os.path.join(out_dir, json_name % (fold, data_set)), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))
    return


def get_img_size(datapath):
    img = Image.open(datapath)
    width, height = img.size
    return width, height


def get_groundtruth(img_path, annotation_folder):
    annot_path = None
    if annotation_folder is not None:
        # look for annotation file
        annot_path = _find_annot_file(img_path, annotation_folder)

    if annot_path is not None:
        frame_number = _get_frame_number(img_path)
        annotation = AnnotationImage(frame_number, annot_path)
        boxes, labels = annotation.get_bboxes_labels()

        return boxes, labels

    return [], []


def _find_annot_file(frame_path, annot_folder):
    """Find annotation file based on video name.

    Arguments:
        frame_path {str} -- video path
        annot_folder {str} -- folder where to look in order to find the annotation file

    Returns:
        str -- [The annotation file path]
    """

    vid_filename = frame_path.split('/')[-2]
    # vid_filename, vid_ext = os.path.splitext(vid_filename)
    found = False

    for (dirpath, dirnames, filenames) in os.walk(annot_folder):

        if len(filenames) == 0:
            continue

        for file_name in filenames:
            if fnmatch.fnmatch(file_name, vid_filename + '.txt'):
                annot_path = os.path.join(dirpath, file_name)
                found = True
                break
        if found:
            return annot_path

    return


def _get_frame_number(frame_path):
    frame_filename = os.path.split(frame_path)[-1]
    frame_filename, vid_ext = os.path.splitext(frame_filename)

    frame_number = frame_filename.split('_')[-1]

    return int(frame_number)


if __name__ == '__main__':
    args = parse_args()
    # if args.dataset == "cityscapes_instance_only":
    convert_mosquitoes_instance_only(args.datadir, args.anndir, args.outdir, args.file_set,
                                     args.folds)
    # else:
    #     print("Dataset not supported: %s" % args.dataset)
