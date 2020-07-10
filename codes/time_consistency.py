import os
import pickle
from multiprocessing import Pool

import cv2
from tqdm.autonotebook import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from utils.time_consist import apply_time_consistency, video_phaseCorrelation
from video_handling import videoObj


def run_on_video(video_path, predictor, output_path=None):

    video = videoObj(video_path)
    frame_gen = video.frame_from_video()

    detections = dict()

    if output_path is not None:

        output_file = cv2.VideoWriter(
            filename=output_path,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=video.fps,
            frameSize=(video.width, video.height),
            isColor=True,
        )

    for idx, frame in enumerate(tqdm(frame_gen, total=video.videoInfo.getNumberOfFrames())):

        #         im = cv2.resize(frame, (1333, 800)) # resize as model was trained
        res = predictor(frame)
        detections[f'frame_{idx:04d}'] = res

        # resize to original size
        #         res = cv2.resize(res, (video.width, video.height))

        if output_path is not None:
            output_file.write(res[:, :, ::-1])

    if output_path is not None:
        output_file.release()

    return detections


if __name__ == "__main__":

    this_filedir = os.path.dirname(os.path.realpath(__file__))

    data_path = "../data/_under_construction"
    video_name = "20181022_rectfied_DJI_0031.avi"
    video_path = os.path.join(this_filedir, data_path, video_name)

    model_name = "faster_rcnn_R_50_C4_1x"
    config_file = os.path.join(this_filedir, "configs", "mosquitoes", model_name + ".yaml")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = os.path.join(this_filedir, "..", "output", model_name, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 600
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    predictor = DefaultPredictor(cfg)
    # preds = run_on_video(video_path, predictor)
    pkl_file = open('preds.pkl', 'rb')
    # pickle.dump(preds, pkl_file)
    preds = pickle.load(pkl_file)
    pkl_file.close()

    vid = videoObj(video_path, video_path.replace(".avi", ".txt"))
    # phase_corr = video_phaseCorrelation(vid, scale=1)
    pkl_file = open('phasecorr.pkl', 'rb')
    # pickle.dump(phase_corr, pkl_file)
    phase_corr = pickle.load(pkl_file)
    pkl_file.close()

    win_size = 5
    center = True
    threshold = 0.5
    preds_time = apply_time_consistency(
        preds, phase_corr, win_size=win_size, center=True, thr=threshold)

    print("end")
