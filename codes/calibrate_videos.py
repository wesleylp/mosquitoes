import glob
import os
from time import sleep

import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

import cv2


def chessboard_keypoints(video_path,
                         first_frame=0,
                         last_frame=None,
                         every=20,
                         pattern_size=(9, 6),
                         square_size=1.0,
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                         debug=False,
                         verbose=False):

    # TODO: Save a file with the cam params
    # so that, we don't need to recalcute these all the time

    print('Detecting keypoint in video: {}...'.format(video_path))

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    num_pat_found = 0  # number of images where the pattern has been found

    video = cv2.VideoCapture(video_path)

    # Frames to scan in order to detect keypoints
    if last_frame is None:
        last_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # loop over frames
    for i in tqdm(range(first_frame, last_frame)):

        sleep(0.01)

        video.grab()

        if (i % every) == 0:

            # Get the ith frame
            img = video.retrieve()[1]
            # retval, img, _ = self.get_fra?Zme(i)

            # if not retval:
            #     tqdm.write('video capture failed!')
            #     break

            if verbose:
                tqdm.write(' Searching for chessboard in frame ' + str(i) + '...')

            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (w, h) = gray.shape

            # cv2.FindChessboardCorners cannot detect chessboard on very large images
            # The likely correct way to proceed is to start at a lower resolution
            # (i.e. downsizing), then scale up the positions of the corners thus found,
            # and use them as the initial estimates for a run of cvFindCornersSubpix at
            # full resolution.
            # (https://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca/15074774)

            # blur before resize
            # blur = cv2.GaussianBlur(gray, (7, 7), 1)

            # resize image
            # TODO: Find a way to compute the best way to compute scale_factor
            # maybe put the image in a standard size before finding keypoints
            scale_factor = .3
            gray_small = cv2.resize(
                gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # Find the chess board corners
            ret, corners_small = cv2.findChessboardCorners(
                gray_small,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret is True:

                if verbose:
                    tqdm.write('pattern found')

                # scale up the positions
                corners = corners_small / scale_factor

                # update number of images where the pattern has been found
                num_pat_found += 1

                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints.append(corners)

                if debug is True:

                    # print('Searching for chessboard in frame ' + str(i) + '...')
                    # Draw and display the corners
                    # img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # plt.axis('off')
                    # plt.show()

                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.scatter(x=corners[:, :, 0], y=corners[:, :, 1], c='r', s=20)
                    plt.show()
                    tqdm.write(str(img.shape))
            else:
                if verbose:
                    tqdm.write('pattern NOT found')

                if debug is True:
                    # Draw and display the corners
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
                    tqdm.write(str(img.shape))

    print('Number of pattern found: ', num_pat_found)

    return objpoints, imgpoints, w, h


def compute_cam_params(video_path):

    objpoints, imgpoints, w, h = chessboard_keypoints(video_path=video_path)

    # Camera calibration
    print('computing cam params...')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # optimize camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    print('Done!')

    cam_params = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': rvecs,
        'newcameramtx': newcameramtx,
        'roi': roi
    }

    return cam_params


def rectify_video(input_path, output_path, cam_params, quality=5):
    '''
    Produz um vídeo de saída (fileDestino) com os frames do vídeo de origem (fileOrigem) utilizando
    o ImageIO, wrapper do ffmpeg.
    Argumentos:
        * fileOrigem: Caminho do vídeo de origem.
        * fileDestino Caminho do vídeo de saída.
        * quality: Qualidade utilizada pelo codec, onde:
                   0: maior compressão possível (pior qualidade)
                   10: menor compressão possível (melhor qualidade)
    '''
    # Carrego o vídeo de origem e seu fps
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']
    # nb_frames = reader.get_meta_data()['nframes']

    mtx = cam_params['mtx']
    dist = cam_params['dist']
    newcameramtx = cam_params['newcameramtx']

    # Crio o writer para gerar um vídeo de saída com qualidade 10 (menor compressão possível)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality)
    frameCountDest = 0

    print('generating video: {}'.format(output_path))
    for img in tqdm(reader):
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        writer.append_data(dst)
        frameCountDest += 1
    writer.close()
    print('Done!')


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
