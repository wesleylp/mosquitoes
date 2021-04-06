import argparse
import copy
import json
import os
import pickle
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from convert_mosquitoes_to_coco import Convert2Coco
from detectron2.config import get_cfg
from detectron2.data import (build_detection_test_loader, get_detection_dataset_dicts)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluators, inference_context,
                                   inference_on_dataset)
from utils.evaluation import CfnMat
from utils.eval_pipes import PipesEval
from utils.time_consist import (apply_time_consistency, filter_pipes, pipe_to_frame_instances,
                                tube_space_time, video_phaseCorrelation)
from video_handling import videoObj


def run_on_video(video_path, predictor, every=1, output_path=None):

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

        if (idx % every) != 0:
            continue

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


def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not filepath.endswith('.pkl'):
        filepath = filepath + '.pkl'

    with open(f"{filepath}", 'wb') as file:
        pickle.dump(results, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Time Consist")
    parser.add_argument("--model_name",
                        type=str,
                        default="faster_rcnn_R_101_FPN_3x",
                        metavar="FILE",
                        help="model_name")

    parser.add_argument("--fold", type=int, default=0, help="k number of kfold")

    parser.add_argument("--win_size", type=int, default=5, help="temporal window size")

    parser.add_argument("--center", default=1, help="whether the window is centered")

    parser.add_argument("--threshold",
                        default=0.5,
                        help="IoU threshold to consider boxes in/out image")

    parser.add_argument("--every",
                        default=1,
                        help="interval of frames to predict and use time consistency")

    parser.add_argument("--min_votes",
                        type=float,
                        default=0.5,
                        help="percentage of boxes in temporal window to consider FP or TP")

    parser.add_argument("--eval_every",
                        type=int,
                        default=30,
                        help="interval of frames to evaluate ")

    args = parser.parse_args()

    model_name = args.model_name
    fold = args.fold
    win_size = args.win_size
    center = bool(args.center)
    threshold = args.threshold
    every = args.every  # predict and apply time consitency in every `every` frames
    min_votes = args.min_votes  # % of detection in the window
    eval_every = args.eval_every

    this_filedir = os.path.dirname(os.path.realpath(__file__))

    config_file = os.path.join(this_filedir, "configs", "mosquitoes", model_name + ".yaml")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = os.path.join(this_filedir, "..", "output", model_name,
                                     f"mbg_fold{fold}_set_1", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 600
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    predictor = DefaultPredictor(cfg)

    data_path = "../data/_under_construction"

    file_set = os.path.join(this_filedir, "..", "train_sets_kfold.xls")
    df_set = pd.read_excel(file_set, sheet_name=f'train_sets_k{fold}')

    videos_names = df_set[df_set['test'] == True]['Video'].tolist()
    videos_names = [s + ".avi" for s in videos_names]

    video_coco = Convert2Coco()
    out_idx = 0
    out = {}
    out_time = {}

    pipes_eval = PipesEval()
    pipes_eval.reset()

    for video_name in videos_names:
        video_path = os.path.join(this_filedir, data_path, video_name)

        print('Predicting...')

        pred_file = video_path.replace(".avi", f"_{model_name}_fold{fold}_preds.pkl")
        if os.path.isfile(pred_file):
            print('loading predictions...')
            pkl_file = open(pred_file, 'rb')
            preds_loaded = pickle.load(pkl_file)

            preds = {}
            for n, (k, v), in enumerate(preds_loaded.items()):
                if (n % every) != 0:
                    continue
                preds[k] = v

        else:
            print('computing predictions...')
            pkl_file = open(pred_file, 'wb')
            print(video_path)
            preds = run_on_video(video_path, predictor, every=1)
            pickle.dump(preds, pkl_file)
        pkl_file.close()
        # preds = run_on_video(video_path, predictor, every=every)

        vid = videoObj(video_path, video_path.replace(".avi", ".txt"))

        print('Phase correlation...')
        phase_corr_file = video_path.replace(".avi", "_phaseCorr.pkl")

        if os.path.isfile(phase_corr_file):
            print('loading phase corr...')
            pkl_file = open(phase_corr_file, 'rb')
            phase_corr = pickle.load(pkl_file)
        else:
            print('computing phase corr...')
            pkl_file = open(phase_corr_file, 'wb')
            phase_corr = video_phaseCorrelation(vid, scale=0.4)
            pickle.dump(phase_corr, pkl_file)
        pkl_file.close()

        width = vid.videoInfo.getWidth()
        height = vid.videoInfo.getHeight()

        preds_copy = copy.deepcopy(preds)

        pipes = tube_space_time(preds_copy, phase_corr)
        filtered_pipes = filter_pipes(pipes, cut_exts=0, thr=min_votes)
        preds_time = pipe_to_frame_instances(filtered_pipes, vid)

        pipes_eval.process(vid.get_annotations().filter_objects(objects=['tire']), filtered_pipes)

        print('Applying time consistency...')
        # preds_time = apply_time_consistency(preds,
        #                                     phase_corr,
        #                                     vid_height_width=(height, width),
        #                                     win_size=win_size,
        #                                     center=center,
        #                                     thr=threshold,
        #                                     min_votes=min_votes)

        for frame_idx in tqdm(range(vid.videoInfo.getNumberOfFrames())):
            if (frame_idx % eval_every) != 0:
                continue

            gt = vid.get_frame_annotations(frame_idx)
            video_coco.update(
                os.path.join(video_name.split('.avi')[0], f'frame_{frame_idx:04d}.png'), width,
                height, gt)

            out[out_idx] = preds[f'frame_{frame_idx:04d}']
            out_time[out_idx] = preds_time[f'frame_{frame_idx:04d}']
            out_idx += 1

    with open(os.path.join(this_filedir, "../output/time_consist/", f"preds_time_{model_name}.pkl"),
              'wb') as outfile:
        pickle.dump(out_time, outfile)

    valid_data = f"preds_{model_name}"
    video_coco.export(f'{valid_data}.json')
    register_coco_instances(f'{valid_data}', {}, f'{valid_data}.json',
                            os.path.join(this_filedir, data_path, 'frames'))
    # val_loader = build_detection_test_loader(cfg, "gt")
    val_loader = get_detection_dataset_dicts([
        f'{valid_data}',
    ], filter_empty=False)

    coco_eval = COCOEvaluator(f'{valid_data}', cfg, False, output_dir=None)
    cfn_mat = CfnMat(f'{valid_data}', output_dir=None)
    evaluators = DatasetEvaluators([cfn_mat, coco_eval])

    coco_eval_time = COCOEvaluator(f'{valid_data}', cfg, False, output_dir=None)
    cfn_mat_time = CfnMat(f'{valid_data}', output_dir=None)
    evaluators_time = DatasetEvaluators([cfn_mat_time, coco_eval_time])

    evaluators.reset()
    evaluators_time.reset()
    print('Computing results...')
    for idx, inputs in enumerate(val_loader):
        outputs = out[idx]
        outputs_time = out_time[idx]

        evaluators.process([inputs], [outputs])
        evaluators_time.process([inputs], [outputs_time])

    results = evaluators.evaluate()
    results_time = evaluators_time.evaluate()
    results_tube = pipes_eval.evaluate()

    print('Prediction results...')
    print(results)

    print('Prediction with TIME consistency results...')
    print(results_time)

    print('Prediction with TIME consistency results...')
    print(results_tube)

    print("saving results...")
    res_dir = os.path.join(this_filedir, "../output/time_consist")
    res_path = os.path.join(res_dir, f"{model_name}_size{win_size}_vot{int(min_votes*100)}.pkl")
    save_results(results_time, res_path)

    res_dir = os.path.join(this_filedir, "../output/tube")
    res_path = os.path.join(res_dir, f"{model_name}_vot{int(min_votes*100)}.pkl")
    save_results(results_tube, res_path)

    print("end")
