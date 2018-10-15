import os

from utils.vid_utils import find_annot_file
from video_handling import videoObj


class VideoDataset:
    def __init__(self, root_dir, annotation_folder):
        self.root_dir = root_dir
        self.annotation_folder = annotation_folder

    def __len__(self):

        videos = []

        for (dirpath, dirnames, filenames) in os.walk(self.root_dir):

            if len(filenames) == 0:
                continue

            # getting only the video files
            [videos.append(s) for s in filenames if s.lower().endswith(('.mov', '.mp4'))]

        return len(videos)

    def __getitem__(self, video_path):

        annot_path = find_annot_file(video_path, self.annotation_folder)

        vid = videoObj(video_path, annot_path)
        videoname = vid.videopath

        frames = vid.get_all_frames()
        bboxes = [bbox for bbox in vid._annotation.annotation_dict.values()]

        sample = {'{:s}'.format(videoname): {'frames': frames, 'bboxes': bboxes}}

        return sample
