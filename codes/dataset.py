import fnmatch
import os

import torch
from PIL import Image

import torchvision
from annotation import AnnotationImage
from maskrcnn_benchmark.structures.bounding_box import BoxList
from utils.files_utils import Directory
from utils.vid_utils import find_annot_file
from video_handling import videoObj


class VideoLoader:
    def __init__(self, root_dir, annotation_folder=None, transform=None):
        ext = ('.mp4', '.mov')
        self.root_dir = root_dir
        self.video_list = Directory.get_files(self.root_dir, ext, recursive=True)
        self.annotation_folder = annotation_folder
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        annot_path = None
        if self.annotation_folder is not None:
            # look for annotation file
            annot_path = find_annot_file(self.video_list[idx], self.annotation_folder)

        vid = videoObj(self.video_list[idx], annot_path)

        frames = vid.get_batch_frames()
        # annot = vid.get_annotations()
        bboxes = vid.get_batch_annotations()

        if self.transform:
            frames = self.transform(frames)

        # bboxes = [bbox for bbox in annot.annotation_dict.values()]
        # bboxes = annot.annotation_dict

        sample = {'clip': vid, 'frames': frames, 'bboxes': bboxes}

        return sample


class MosquitoDataset(object):
    def __init__(self, root_dir, annotation_folder=None, transforms=None):
        ext = ('.png')
        self.root_dir = root_dir
        self.frames_list = Directory.get_files(self.root_dir, ext, recursive=True)
        self.annotation_folder = annotation_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        img_path = self.frames_list[idx]
        self.img = Image.open(open(img_path, 'rb'))

        annot_path = None
        if self.annotation_folder is not None:
            # look for annotation file
            annot_path = _find_annot_file(self.frames_list[idx], self.annotation_folder)

        if annot_path is not None:
            frame_number = _get_frame_number(self.frames_list[idx])
            annotation = AnnotationImage(frame_number, annot_path)
            boxes, labels = annotation.get_bboxes_labels()

        # TODO: Make a better decision about this
        else:
            boxes, labels = [], []
            print('Annotation not found: ', self.frames_list[idx])

        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # create a BoxList from the boxes
        target = BoxList(boxes, self.img.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", labels)

        if self.transforms:
            self.img, target = self.transforms(self.img, target)

        return self.img, target, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_height = self.img.height
        img_width = self.img.width

        return {"height": img_height, "width": img_width}


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
