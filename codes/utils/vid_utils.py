import fnmatch
import os
from time import sleep

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.autonotebook import tqdm

import cv2
from utils.img_utils import (compute_mse, compute_psnr, compute_ssim, find_chessboard_keypoints_img,
                             rectify_img)

sns.set(
    'paper',
    'white',
    'colorblind',
    font_scale=2.2,
    rc={
        'lines.linewidth': 2,
        'figure.figsize': (10.0, 6.0),
        'image.interpolation': 'nearest',
        'image.cmap': 'gray'
    })


def generate_video_opencv(input_path,
                          output_path,
                          first_frame=0,
                          last_frame=None,
                          skip_frames=0,
                          codec='X264',
                          fps=None,
                          verbose=False):

    print('This may take a while...')

    #     cam_matrix, profile = create_matrix_profile(FC, CC, KC)

    # Load video
    # video_input = videoObj(input_path)
    video_input = cv2.VideoCapture(input_path)

    # if codec is None:
    #     a, b, c, d = video_input.get(cv2.CAP_PROP_FOURCC)
    # codec = video_input.videoInfo.getCodecType()

    fourcc = cv2.VideoWriter.fourcc(*list(codec))

    # number of frames of video
    # nb_frames = video_input.videoInfo.getNumberOfFrames()
    nb_frames = video_input.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps is None:
        # fps = video_input.videoInfo.getFrameRateFloat()
        fps = video_input.get(cv2.CAP_PROP_FPS)

    size = (int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # size = (video_input.videoInfo.getWidth(), video_input.videoInfo.getHeight())

    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    # if last_frame is not specified, use the total number of frames
    if last_frame is None:
        last_frame = int(nb_frames)

    print('generating video: {}'.format(output_path))
    for idx in tqdm(range(0, last_frame)):
        if verbose:
            tqdm.write('On frame {} of {}.'.format(idx, last_frame))

        # _, frame, _ = video_input.get_frame(idx)
        video_input.grab()
        frame = video_input.retrieve()[1]
        writer.write(frame)

    video_input.release()
    writer.release()


def generate_video_imageio(input_path, output_path, quality):
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

    # Crio o writer para gerar um vídeo de saída com qualidade 10 (menor compressão possível)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality)

    print('generating video: {}'.format(output_path))
    for im in tqdm(reader):
        writer.append_data(im)

    writer.close()


def compute_video_size(filepath):
    """Returns the file size in MB

    Arguments:
        filepath {str} -- path to file

    Returns:
        float -- the file size in MB
    """

    return os.path.getsize(filepath) / (1024**2)


def compare_videos(filepath1,
                   filepath2,
                   first_frame=0,
                   last_frame=None,
                   compute_every=1,
                   verbose=False,
                   save_frames=False,
                   save_path=None,
                   debug=False):

    print('Comparing videos... This may take a while... \n')

    vid1 = cv2.VideoCapture(filepath1)
    vid2 = cv2.VideoCapture(filepath2)

    # # print all videos information
    print('Video 1')
    print_video_info(filepath1)
    print('Video 2')
    print_video_info(filepath2)

    # number of frames of both videos
    nb_frames_vid1 = vid1.get(cv2.CAP_PROP_FRAME_COUNT)
    nb_frames_vid2 = vid2.get(cv2.CAP_PROP_FRAME_COUNT)

    if nb_frames_vid1 != nb_frames_vid2:
        raise IOError('Videos have different number of frames!')

    # Initializing MSE_sum and SSIM_sum to compute mean
    mse = []
    psnr = []
    ssim = []

    # if last_frame is not specified, use the total number of frames
    if last_frame is None:
        last_frame = int(nb_frames_vid1)

    # count = 0

    for idx in tqdm(range(first_frame, last_frame)):

        # count += 1

        if verbose:
            tqdm.write('On frame {} of {}'.format(idx + 1, last_frame))

        vid1.grab()
        vid2.grab()

        if (idx % compute_every) == 0:

            frame_vid1 = vid1.retrieve()[1]
            frame_vid2 = vid2.retrieve()[1]

            if save_frames:

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                compression_level = 3
                ext_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

                cv2.imwrite(
                    os.path.join(save_path, 'frame_{:04d}_original.png'.format(idx)), frame_vid1,
                    ext_params)
                cv2.imwrite(
                    os.path.join(save_path, 'frame_{:04d}_compressed.png'.format(idx)), frame_vid2,
                    ext_params)

            # _, frame_vid1, _ = vid1.get_frame(idx)
            # _, frame_vid2, _ = vid2.get_frame(idx)

            if debug:
                # Draw and display the corners
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(frame_vid1, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(frame_vid2, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

            mse.append(compute_mse(frame_vid1, frame_vid2))
            psnr.append(compute_psnr(frame_vid1, frame_vid2))
            ssim.append(compute_ssim(frame_vid1, frame_vid2))

    vid1.release()
    vid2.release()

    return mse, psnr, ssim


def compute_cam_params(objpoints, imgpoints, w, h, alpha=0):
    """[summary]

    Arguments:
        objpoints {list} -- [points in 3D real world]
        imgpoints {list} -- [points in 2D image]
        w {int} -- [image width]
        h {int} -- [image height]

    Keyword Arguments:
        alpha {float} -- [Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1 (when all the source image pixels are retained in the undistorted image).
        alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Obviously, any intermediate value yields an intermediate result between those two extreme cases] (default: {0})

    Returns:
        [dict] -- [camera parameters]
    """

    # Camera calibration
    print('computing cam params...')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    # optimize camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx,
        dist[0][:4],
        (w, h),
        alpha=alpha,
    )
    print('Done!')

    cam_params = {
        'ret': ret,
        'mtx': mtx,
        'dist': dist[0][:4],
        'rvecs': rvecs,
        'tvecs': rvecs,
        'newcameramtx': newcameramtx,
        'roi': roi
    }

    return cam_params


def print_video_info(filepath):

    filesize = compute_video_size(filepath)

    video = cv2.VideoCapture(filepath)

    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print('{}'.format(filepath))
    print('video size: {:.2f} MB'.format(filesize))
    print('fps: {:.2f}'.format(fps))
    print('frames: {:d}'.format(total_frames))
    print('resolution: {}'.format(resolution))

    print('\n')


def chessboard_keypoints_video(video_path,
                               first_frame=0,
                               last_frame=None,
                               every=20,
                               pattern_size=(9, 6),
                               square_size=1.0,
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30,
                                         0.001),
                               debug=False,
                               verbose=False):

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

            ret, corners, w, h = find_chessboard_keypoints_img(
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

    return objpoints, imgpoints, w, h


def rectify_video(input_path, output_path, cam_params, quality=5):
    """Rectify the input video by generating a new video using imageio,
    a ffmpeg wrapper, with quality 'quality'

    Arguments:
        input_path {str} -- [input video path]
        output_path {str} -- [output video path]
        cam_params {dict} -- [dict containing camera parameters]

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
        dst = rectify_img(img, cam_params)
        writer.append_data(dst)

    writer.close()
    print('Done!')


def save_frames(input_path,
                output_path,
                first_frame=0,
                last_frame=None,
                save_every=1,
                verbose=False):

    # Load video
    vid = cv2.VideoCapture(input_path)

    nb_frames_vid = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    if last_frame is None:
        last_frame = int(nb_frames_vid)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx in tqdm(range(first_frame, last_frame)):

        if verbose:
            tqdm.write('On frame {} of {}'.format(idx + 1, last_frame))

        vid.grab()

        if (idx % save_every) == 0:

            frame_vid = vid.retrieve()[1]

            compression_level = 3
            ext_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

            cv2.imwrite(
                os.path.join(output_path, 'frame_{:04d}.png'.format(idx)), frame_vid, ext_params)

    vid.release()


def find_annot_file(video_path, annot_folder):
    """Find annotation file based on video name.

    Arguments:
        video_path {str} -- video path
        annot_folder {str} -- folder where to look in order to find the annotation file

    Returns:
        str -- [The annotation file path]
    """

    vid_filename = os.path.split(video_path)[-1]
    vid_filename, vid_ext = os.path.splitext(vid_filename)

    found = False

    for (dirpath, dirnames, filenames) in os.walk(annot_folder):

        if len(filenames) == 0:
            continue

        for file_name in filenames:

            if fnmatch.fnmatch(file_name, vid_filename + '.txt'):
                annot_path = os.path.join(dirpath, file_name)
                found = True
                break
        if found:
            break
    return annot_path
