import os

from tqdm.autonotebook import tqdm

import cv2


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
            save_frames(input_path, output_path, save_every=50)
