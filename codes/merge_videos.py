import imageio
import argparse
from tqdm import tqdm


def merge_videos(list_video_path, output_path, quality=5):
    # TODO: for sure there is a better way to do this.
    """Downsample a video.

    Args:
        video_path (list): List of videos path to be merged.
        output_path (str): Path to the output video.
        quality (float): The quality level of new videos generated. From 0 to 10. the higher, the better.
    """

    if len(list_video_path) == 1:
        print('Only one video, no need to merge.')
        return

    reader = imageio.get_reader(list_video_path[0])
    fps = reader.get_meta_data()['fps']
    reader.close()

    if output_path is None:
        output_path = list_video_path[0].replace('.avi', '_merged.avi')
    writer = imageio.get_writer(output_path, fps=fps, quality=quality)

    for video_path in tqdm(list_video_path):
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        nb_frames = int(reader.get_meta_data()['duration'] * fps)

        for frame_nb, frame in enumerate(tqdm(reader, total=nb_frames)):
            writer.append_data(frame)

        reader.close()

    writer.close()
    print('Done!')


if __name__ == '__main__':

    list_videos = [
        "~/Downloads/20210317_rectified_DJI_0075.avi",
        "~/Downloads/20210317_rectified_DJI_0076.avi",
    ]

    output_path = "~/Downloads/20210317_rectified_DJI_0075_0076.avi"

    merge_videos(list_videos, output_path, quality=5)
