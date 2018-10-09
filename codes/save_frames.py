import os

from utils.vid_utils import save_frames

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Rectify videos.')

    parser.add_argument(
        '--datapath',
        type=str,
        default='../data/DJI4_cam/2018-09-05/seq001',
        help='Data path of videos to extract frames from.')

    parser.add_argument('--save_every', type=int, default=50, help='Number of frames to skip.')

    args = parser.parse_args()

    data_path = args.datapath
    save_every = args.save_every

    data_path = '../data/DJI4_cam/2018-09-05/seq001'

    for (dirpath, dirnames, filenames) in os.walk(data_path):

        if len(filenames) == 0:
            continue

        for filename in filenames:

            if filename.lower().endswith(('.mov', '.mp4')):

                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(dirpath, os.path.splitext(filename)[0])

                print(input_path)
                print(output_path)
                save_frames(input_path, output_path, save_every=save_every)
