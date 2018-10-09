import os
import pickle
import warnings

from generate_videos import compare_videos

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Rectify videos.')

    parser.add_argument(
        '--datapath',
        type=str,
        default='../data/DJI4_cam/2018-09-05/seq001',
        help='Data path of  videos to be compared.')

    args = parser.parse_args()

    data_path = args.datapath

    for (dirpath, dirnames, filenames) in os.walk(data_path):

        if len(filenames) == 0:
            continue

        # getting only the video files
        videos = [s for s in filenames if s.lower().endswith(('.mov', '.mp4'))]

        if len(videos) == 0:
            continue

        # segregating the original the generated videos
        original_video = [s for s in videos if ('opencv' not in s) and ('imageio' not in s)]
        generated_videos = [s for s in videos if ('opencv' in s or 'imageio' in s)]

        if len(generated_videos) == 0:
            continue

        # original video complete path
        input_path = os.path.join(dirpath, original_video[0])

        for generated_video in generated_videos:

            # generated video complete path
            output_path = os.path.join(dirpath, generated_video)
            mse, psnr, _ = compare_videos(
                input_path, output_path, compute_every=10, save_frames=False)

            # saving values computed
            txt_file = os.path.join(dirpath, '{}.data'.format(os.path.splitext(generated_video)[0]))
            print('saving results in: {}'.format(txt_file))
            results = {'mse': mse, 'psnr': psnr}
            with open(txt_file, 'wb') as f:
                pickle.dump(results, f)
            f.close()
