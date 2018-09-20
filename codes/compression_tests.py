import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_ssim as ssim
from tqdm.autonotebook import tqdm

import cv2

# from video_utils import videoObj


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
    frameCountDest = 0

    for im in reader:
        writer.append_data(im)
        frameCountDest += 1
    writer.close()


def compute_mse(A, B):
    return ((A - B)**2).sum() / np.prod([*A.shape])


def compute_ssim(frame1, frame2, multichannel=True):
    return ssim(frame1, frame2, multichannel=multichannel)


def compute_filesize(filepath):
    """Returns the file size in MB

    Arguments:
        filepath {str} -- path to file

    Returns:
        float -- the file size in MB
    """

    return os.path.getsize(filepath) / (1024**2)


def compare_videos(filepath1,
                   filepath2,
                   first_frame=1,
                   last_frame=None,
                   compute_every=10,
                   verbose=False,
                   debug=False):

    print('this may take a while... \n')

    # Load video
    # vid1 = videoObj(filepath1)
    # vid2 = videoObj(filepath2)

    vid1 = cv2.VideoCapture(filepath1)
    vid2 = cv2.VideoCapture(filepath2)

    # # print all videos information
    # vid1.videoInfo.printAllInformation()
    # vid2.videoInfo.printAllInformation()
    print('Video 1')
    print_video_info(vid1)
    print('Video 2')
    print_video_info(vid2)

    # number of frames of both videos
    # nb_frames_vid1 = vid1.videoInfo.getNumberOfFrames()
    # nb_frames_vid2 = vid2.videoInfo.getNumberOfFrames()
    nb_frames_vid1 = vid1.get(cv2.CAP_PROP_FRAME_COUNT)
    nb_frames_vid2 = vid2.get(cv2.CAP_PROP_FRAME_COUNT)

    if nb_frames_vid1 != nb_frames_vid2:
        raise IOError('Videos have different number of frames!')

    # Initializing MSE_sum and SSIM_sum to compute mean
    mse = []
    ssim = []

    # if last_frame is not specified, use the total number of frames
    if last_frame is None:
        last_frame = int(nb_frames_vid1)

    count = 0

    for idx in tqdm(range(first_frame - 1, last_frame)):

        count += 1

        if verbose:
            tqdm.write('On frame {} of {}'.format(idx + 1, last_frame))

        vid1.grab()
        vid2.grab()

        if (count % compute_every) == 0:

            frame_vid1 = vid1.retrieve()[1]
            frame_vid2 = vid2.retrieve()[1]

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
            ssim.append(compute_ssim(frame_vid1, frame_vid2))

    vid1.release()
    vid2.release()

    return mse, ssim


def print_video_info(video):

    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print('fps: {}'.format(fps))
    print('frames: {}'.format(total_frames))
    print('resolution: {}'.format(resolution))
    print('\n')
