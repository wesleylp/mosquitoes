import glob
import os

from utils.calibration import (compute_cam_params, compute_undistortion_map,
                               find_chessboard_kpts_video, rectify_video)

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

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.0,
        help=
        'Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image'
    )

    args = parser.parse_args()

    data_path = args.datapath
    alpha = args.alpha
    quality = args.quality

    for (dirpath, dirnames, filenames) in os.walk(data_path):

        if len(filenames) == 0 or len(dirnames) == 0:
            continue

        for dirname in dirnames:

            if 'seq' in dirname:

                # setting calibration video path
                calibration_path = os.path.join(dirpath, dirname, 'calibration')
                video_calib_path = glob.glob(os.path.join(calibration_path, '*.MOV'))[0]

                # computing cam params and maps
                objpoints, imgpoints, img_size = find_chessboard_kpts_video(
                    video_path=video_calib_path, every=10)
                cam_params = compute_cam_params(objpoints, imgpoints, img_size, alpha=alpha)
                maps = compute_undistortion_map(cam_params, img_size)

                # setting missions videos path
                missions_path = os.path.join(dirpath, dirname, 'missions')
                videos_missions = glob.glob(os.path.join(missions_path, '*.MOV'))

                # applying calibration to each video
                for video_mission in videos_missions:

                    print('Rectifying video {} ...'.format(video_mission))

                    video_name = os.path.split(video_mission)[-1]
                    video_rectified_name = 'rectified_' + video_name
                    output_path = os.path.join(missions_path, video_rectified_name)

                    rectify_video(video_mission, output_path, cam_params, maps, quality)
