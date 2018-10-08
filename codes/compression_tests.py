import os
import warnings

from generate_videos import generate_video_imageio, generate_video_opencv

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description=
        'Testing compression at different levels; We extract the frames and generated new videos with different levels of compression.'
    )

    parser.add_argument(
        '--datapath',
        type=str,
        default='../data/DJI4_cam/2018-09-05/seq001',
        help='Data path of videos to be tested.')

    args = parser.parse_args()

    data_path = args.datapath

    for (dirpath, dirnames, filenames) in os.walk(data_path):

        if len(filenames) == 0:
            continue

        for filename in filenames:

            # if the file is video, then process!
            if filename.lower().endswith(('.mov', '.mp4')):

                input_path = os.path.join(dirpath, filename)
                name, ext = os.path.splitext(filename)

                # generate the videos:

                # using opencv
                output_file = '{}_opencv{}'.format(name, ext)
                output_path = os.path.join(dirpath, output_file)
                generate_video_opencv(input_path, output_path)

                # using imageio
                for quality in range(0, 11):
                    output_file = '{}_imageio{:02d}{}'.format(name, quality, ext)
                    output_path = os.path.join(dirpath, output_file)
                    generate_video_imageio(input_path, output_path, quality)
