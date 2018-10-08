import glob
import os

from utils.vid_utils import compute_cam_params, rectify_video

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Rectify videos.')

    parser.add_argument(
        '--datapath',
        type=str,
        default='../data/dataset/',
        help='Data path of calibration and videos to be rectified.')

    parser.add_argument(
        '--quality',
        type=float,
        default=5,
        help='The quality level of new videos generated. From 0 to 10. the higher, the better.')

    args = parser.parse_args()

    data_path = args.datapath
    quality = args.quality

    for (dirpath, dirnames, filenames) in os.walk(data_path):

        if len(filenames) == 0 or len(dirnames) == 0:
            continue

        for dirname in dirnames:

            if 'seq' in dirname:

                calibration_path = os.path.join(dirpath, dirname, 'calibration')
                video_calib_path = glob.glob(os.path.join(calibration_path, '*.MOV'))[0]
                print(video_calib_path)
                cam_params = compute_cam_params(video_path=video_calib_path)

                missions_path = os.path.join(dirpath, dirname, 'missions')
                videos_missions = glob.glob(os.path.join(missions_path, '*.MOV'))

                for video_mission in videos_missions:

                    print('Rectifying video {} ...'.format(video_mission))

                    video_name = os.path.split(video_mission)[-1]
                    video_rectified_name = 'rectfied_' + video_name
                    output_path = os.path.join(missions_path, video_rectified_name)

                    rectify_video(video_mission, output_path, cam_params, quality)