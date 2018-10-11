import sys

from utils.vid_utils import find_annot_file
from video_handling import videoObj

sys.path.append('../codes')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Play video.')

    parser.add_argument(
        '--videopath',
        type=str,
        default='../data/CEFET/VideoDataSet/5m/DJI00804_05m_04aTomada.mp4',
        help='Path of video to play.')

    parser.add_argument(
        '--annotfolder',
        type=str,
        default='../data/CEFET/zframer-marcacoes',
        help='Folder where the annotation files are.')

    parser.add_argument(
        '--showbb', type=bool, default=False, help='Whether display bounding box or not.')

    parser.add_argument(
        '--framebyframe', type=bool, default=False, help='If true, display video frame by frame.')

    args = parser.parse_args()

    video_path = args.videopath

    show_bb = args.showbb

    frame_by_frame = args.framebyframe

    if show_bb:
        annotation_folder = args.annotfolder
        annot_path = find_annot_file(video_path, annotation_folder)
    else:
        annot_path = None

    # create video object
    video = videoObj(videopath=video_path, annotation_path=annot_path)

    video.videoInfo.printAllInformation()

    if frame_by_frame:
        video.play_frame_by_frame(show_bb=show_bb)
    else:
        video.play_video(show_bb=show_bb)
