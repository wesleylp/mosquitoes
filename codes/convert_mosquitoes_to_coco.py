import argparse
import fnmatch
import json
import os
import sys
from itertools import product
from detectron2 import data
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from annotation import AnnotationImage
from cvat_annotation import CVATAnnotationImage
from boxes import box_area, clip_box_to_image, xyxy_to_xywh


class Convert2Coco:
    def __init__(
        self,
        category_dict={
            # '__background__': 0,
            'tire': 0,
            # 'pool': 1,
            # 'bucket': 2,
            # 'watertank': 3,
            # 'puddle': 4,
            # 'bottle': 5
        },
    ):

        self.category_dict = category_dict

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

        # gambiarra (tranform tuple into dict)
        if isinstance(gt, tuple):
            gt = {label: box for label, box in zip(gt[1], gt[0])}

        for label, box in gt.items():  #zip(gt[1], gt[0]):
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
    this_filedir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Convert dataset')
    # parser.add_argument('--dataset', help="cocostuff, cityscapes", default=None, type=str)
    parser.add_argument('--outdir', help="output dir for png files", default='data/v1', type=str)

    parser.add_argument('--anndir',
                        help="annotation dir for txt files",
                        default=os.path.join(this_filedir, '../data/v1/annotation'),
                        type=str)

    parser.add_argument('--annotator', help="annotator name used", default='cvat', type=str)

    parser.add_argument('--datadir',
                        help="data dir for annotations to be converted",
                        default=os.path.join(this_filedir, '../data/v1/frames'),
                        type=str)

    parser.add_argument('--file_set',
                        help="file path for train sets separation",
                        default=os.path.join(this_filedir, '../data/v1/train_sets_kfold_v1.0.xls'),
                        type=str)

    parser.add_argument('--folds', help="number of folds", default=1, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def convert_mosquitoes_instance_only(data_dir, ann_dir, annotator, out_dir, file_set, folds=1):

    category_dict = {
        #'__background__': 0,
        'tire': 0,
        # 'pool': 1,
        # 'bucket': 2,
        # 'watertank': 0,
        # 'puddle': 4,
        # 'bottle': 4,
    }

    json_name = f'coco_format_%s_{"_".join(category_dict.keys())}.json'

    # subsets = ['train', 'val', 'test']

    # sets = product(np.arange(folds, dtype=np.int8), subsets)

    if 'kfold' in file_set:
        df_set = {
            k: pd.read_excel(file_set, sheet_name=f'train_sets_k{k}')
            for k in np.arange(folds)
        }

    else:
        df_set = {0: pd.read_csv(file_set)}

    mask = df_set[0]['test'] == True
    videos_train_val = df_set[0][~mask]['seq'].tolist()
    videos_test = df_set[0][mask]['seq'].tolist()

    sets = [f'train{n}' for n in np.arange(len(videos_train_val))]
    sets += [f'val{n}' for n in np.arange(len(videos_train_val))]
    sets += ['train+val', 'test']

    annot_stats = {}

    # for fold, data_set in tqdm(sets):
    r = re.compile(r'\d{8}_rectified_DJI_\d{4}')
    for data_set in tqdm(sets):
        print(f'Starting {data_set}')

        if 'train+val' in data_set:
            videos_set = videos_train_val

        elif 'train' in data_set:
            idx_val = int(data_set[-1])
            videos_set = [x for i, x in enumerate(videos_train_val) if i != idx_val]

        elif 'val' in data_set:
            idx_val = int(data_set[-1])
            videos_set = [videos_train_val[idx_val]]

        elif data_set == 'test':
            videos_set = videos_test

        else:
            raise ValueError("Invalid data_set")

        # videos_set = df_set[fold][df_set[fold][data_set] == True]['Video'].tolist()

        ann_dict = Convert2Coco(category_dict)
        for (dirpath, dirnames, filenames) in os.walk(data_dir):
            if len(filenames) == 0:
                continue

            video_name = dirpath.split('/')[-1]
            if video_name not in videos_set:
                continue

            print(dirpath)
            for filename in filenames:
                if filename.lower().endswith(('.png')):

                    # this piece of code changes the video name in the format 'YYYYMMDD_DJI_XXXX'
                    # to 'videoNN' as used in paper
                    if r.match(video_name) is not None:
                        video_name = df_set[0][df_set[0]["Video"] == video_name]['seq'].tolist()[0]

                    img_name_short = os.path.join(video_name, filename)
                    img_name = os.path.join(dirpath, filename)

                    width, height = get_img_size(img_name)
                    bboxes = get_groundtruth(img_name, ann_dir, annotator)

                    ann_dict.update(img_name_short, width, height, bboxes)
        ann_dict.export(os.path.join(out_dir, json_name % data_set))

    return


def get_img_size(datapath):
    img = Image.open(datapath)
    width, height = img.size
    return width, height


def get_groundtruth(img_path, annotation_folder, annotator):
    annot_path = None
    if annotation_folder is not None:
        # look for annotation file
        annot_path = _find_annot_file(img_path, annotation_folder, annotator)

    if annot_path is not None:
        frame_number = _get_frame_number(img_path)

        if annotator.lower() == 'zframer':
            annotation = AnnotationImage(frame_number, annot_path)
        elif annotator.lower() == 'cvat':
            annotation = CVATAnnotationImage(frame_number, annot_path)
        else:
            raise ValueError(f"annotator must be zframer or cvat: {annotator}")

        boxes, labels = annotation.get_bboxes_labels()

        return boxes, labels

    return [], []


def _find_annot_file(frame_path, annot_folder, annotator):
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

    if annotator.lower() == 'zframer':
        fileformat = '.txt'
    elif annotator.lower() == 'cvat':
        fileformat = '.xml'
    else:
        raise ValueError(f"annotator must be zframer or cvat: {annotator}")

    for (dirpath, dirnames, filenames) in os.walk(annot_folder):

        if len(filenames) == 0:
            continue

        for file_name in filenames:
            if fnmatch.fnmatch(file_name, vid_filename + fileformat):
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
    convert_mosquitoes_instance_only(args.datadir, args.anndir, args.annotator, args.outdir,
                                     args.file_set, args.folds)
    # else:
    #     print("Dataset not supported: %s" % args.dataset)
