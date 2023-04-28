import imageio
import argparse
from tqdm import tqdm


def donwsample_video(video_path, output_path, sample_ratio=30, quality=5):
    """Downsample a video.

    Args:
        video_path (str): Path to the video to be downsampled.
        output_path (str): Path to the output video.
        quality (float): The quality level of new videos generated. From 0 to 10. the higher, the better.
    """

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    nb_frames = int(reader.get_meta_data()['duration'] * fps)

    if output_path is None:
        output_path = video_path.replace('.avi', '_sampled.avi')

    writer = imageio.get_writer(output_path, fps=fps / sample_ratio, quality=quality)

    for frame_nb, frame in enumerate(tqdm(reader, total=nb_frames)):
        if frame_nb % sample_ratio == 0:
            writer.append_data(frame)

    reader.close()

    writer.close()
    print('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Downsample videos.')

    parser.add_argument('--video_path',
                        type=str,
                        help='Path to the video to be downsampled.',
                        default='~/Downloads/20210317_rectified_DJI_0075_0076.avi')
    parser.add_argument('--output_path', type=str, help='Path to the output video.', default=None)
    parser.add_argument('--sample_ratio', type=int, default=30, help='The sample ratio.')
    parser.add_argument(
        '--quality',
        type=int,
        default=5,
        help='The quality level of new videos generated. From 0 to 10. the higher, the better.')

    args = parser.parse_args()

    video_path = args.video_path
    output_path = args.output_path
    sample_ratio = args.sample_ratio
    quality = args.quality

    donwsample_video(video_path, output_path, sample_ratio, quality)
