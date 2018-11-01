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
