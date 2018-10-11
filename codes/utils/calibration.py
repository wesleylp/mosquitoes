from time import sleep

import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

import cv2


def find_chessboard_kpts_img(img,
                             pattern_size=(9, 6),
                             square_size=1.0,
                             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                             debug=False,
                             verbose=False):
    """Find the corners on the chessboard pattern. It first resize the image for a smaller size, try to find the keypoint, reescale the keypoints in order to be compatible to original image size and the refine the locations of keypoints.

    Arguments:
        img {np.array} -- image containing calibration pattern

    Keyword Arguments:
        pattern_size {tuple} -- Pattern size (default: {(9, 6)})
        square_size {float} -- size of squares in the calibration pattern (default: {1.0})
        criteria {tuple} -- opencv flags (default: {(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)})
        debug {bool} -- flag to help to debug the function (default: {False})
        verbose {bool} -- flag to output info (default: {False})

    Returns:
        tuple -- ret: inform if the pattern was found or not
                 corners: list with corners positions
                 img_size: image size (heigh, width)


    """

    # get image size
    # h, w = img.shape[:2]
    img_size = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.FindChessboardCorners cannot detect chessboard on very large images
    # The likely correct way to proceed is to start at a lower resolution
    # (i.e. downsizing), then scale up the positions of the corners thus found,
    # and use them as the initial estimates for a run of cvFindCornersSubpix at
    # full resolution.
    # (https://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca/15074774)

    # resize image
    # TODO: Find a way to compute the best way to compute scale_factor
    # maybe put the image in a standard size before finding keypoints
    scale_factor = .3
    gray_small = cv2.resize(
        gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    corners = None

    # Find the chess board corners
    ret, corners_small = cv2.findChessboardCorners(
        gray_small,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if ret is True:

        # scale up the positions
        corners = corners_small / scale_factor

        corners = cv2.cornerSubPix(gray, corners, (23, 23), (-1, -1), criteria)

    return ret, corners, img_size


def find_chessboard_kpts_video(video_path,
                               first_frame=0,
                               last_frame=None,
                               every=20,
                               pattern_size=(9, 6),
                               square_size=1.0,
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30,
                                         0.001),
                               debug=False,
                               verbose=False):
    """Run 'find_chessboard_kpts_img' on a video sequence.

    Arguments:
        video_path {str} -- path of the calibration video

    Keyword Arguments:
        first_frame {int} -- first frame to find keypoints (default: {0})
        last_frame {int} -- last frame to find keypoints. If None, use the the last video frame (default: {None})
        every {int} -- try to find keypoints at 'every' frame of the video. Number of frames to skip. (default: {20})
        pattern_size {tuple} -- pattern size to find (default: {(9, 6)})
        square_size {float} -- square size in patter (default: {1.0})
        criteria {tuple} -- opencv flags (default: {(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30,0.001)})
        debug {bool} -- flag to help to debug function (default: {False})
        verbose {bool} -- flag to output info (default: {False})

    Returns:
        tuple -- objpoints: list of correspondent points in the real world
                 imgpoints: list of correspondent points on 2D image
                 img_size: video resolution (height, width)
    """

    # TODO: Save a file with the cam params
    # so that, we don't need to recalcute these all the time

    print('Detecting keypoints...')

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

            if verbose:
                tqdm.write(' Searching for chessboard in frame ' + str(i) + '...')

            ret, corners, img_size = find_chessboard_kpts_img(
                img=img,
                pattern_size=pattern_size,
                square_size=square_size,
                criteria=criteria,
                debug=debug,
                verbose=verbose)

            if ret is True:

                if verbose or debug:
                    tqdm.write('pattern found')

                # update number of images where the pattern has been found
                num_pat_found += 1

                objpoints.append(objp)
                imgpoints.append(corners)

                if debug is True:

                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.scatter(x=corners[:, :, 0], y=corners[:, :, 1], c='r', s=20)
                    plt.show()
                    tqdm.write(str(img.shape))
            else:
                if verbose or debug:
                    tqdm.write('pattern NOT found')

                if debug is True:
                    # Draw and display the corners
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
                    tqdm.write(str(img.shape))

    print('Number of pattern found: ', num_pat_found)

    return objpoints, imgpoints, img_size


def compute_cam_params(objpoints, imgpoints, img_size, alpha=0):
    """Compute the camera parameters

    Arguments:
        objpoints {list} -- [points in 3D real world]
        imgpoints {list} -- [points in 2D image]
        img_size {tuple} -- [image size (heigh, width)]


    Keyword Arguments:
        alpha {float} -- [Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image).
        alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Obviously, any intermediate value yields an intermediate result between those two extreme cases] (default: {0})

    Returns:
        [dict] -- [camera parameters]
    """

    h, w = img_size[:2]

    # Camera calibration
    print('computing cam params...')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # optimize camera matrix
    # we use only k1,k2, p1, and p2 dist coeff because using more coefs can lead to numerical instability
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist[0][:4], (w, h), alpha, (w, h))
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


def compute_undistortion_map(cam_params, image_size):
    """Computes the undistortion and rectification transformation map.

    Arguments:
        cam_params {dict} -- dict containing cam params
        image_size {tuple} -- tuple with image size (height, width)

    Returns:
        tuple -- output maps
    """

    h, w = image_size[:2]

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(cam_params['mtx'], cam_params['dist'][0][:4], None,
                                             cam_params['newcameramtx'], (w, h), 5)
    return (mapx, mapy)


def rectify_img(img, maps, cam_params):
    """rectify image by using remap.

    Arguments:
        img {np.array} -- image to be rectified
        maps {tuple} -- output maps. maps = (mapx, mapy)
        cam_params {dict} -- dict containing camera params.

    Returns:
        np.array -- rectified image
    """

    # get maps
    mapx, mapy = maps

    # INTER_LINEAR - a bilinear interpolation (used by default)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # # undistorting image
    # # we use only k1,k2, p1, and p2 dist coeff because using more coefs can lead to numerical instability
    # dst = cv2.undistort(img, cam_params['mtx'], cam_params['dist'][0][:4], None,
    #                     cam_params['newcameramtx'])

    # crop and save the image
    # Output rectangle inside the rectified images where all the pixels are valid. If alpha=0 (see vid_utils.compute_cam_params), the ROI cover the whole images. Otherwise, they are likely to be smaller.
    x, y, w, h = cam_params['roi']
    dst = dst[y:y + h, x:x + w]

    return dst


def rectify_video(input_path, output_path, cam_params, maps, quality=5):
    """Rectify the input video by generating a new video using imageio,
    a ffmpeg wrapper, with quality 'quality'

    Arguments:
        input_path {str} -- [input video path]
        output_path {str} -- [output video path]
        cam_params {dict} -- dict containing cam params
        maps {tuple} -- tuple containing the undistortion and rectification transformation maps. maps = (mapx, mapy)

    Keyword Arguments:
        quality {int} -- [video quality from 0 to 10. The higher, the better] (default: {5})
    """

    # load original video and its fps
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']

    # Crio o writer para gerar um vídeo de saída com qualidade 10 (menor compressão possível)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality)

    print('generating video: {}'.format(output_path))
    for img in tqdm(reader):
        dst = rectify_img(img, maps, cam_params)
        writer.append_data(dst)

    writer.close()
    print('Done!')
