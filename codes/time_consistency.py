import argparse
import copy
import json
import os
import pickle
from multiprocessing import Pool

from collections import OrderedDict
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
from utils.evaluation import CfnMat, filter_preds_score_video, filter_preds_margin_video, filter_annot_margin_video, filter_gt_margin_image
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

    this_filedir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="Time Consist")
    parser.add_argument("--model_name",
                        type=str,
                        default="faster_rcnn_R_101_FPN_3x",
                        metavar="FILE",
                        help="model_name")

    parser.add_argument("--object", type=str, default="watertank", help="object to detect")

    parser.add_argument("--fold", type=int, default=0, help="k number of kfold")

    parser.add_argument("--set", type=str, default='val', help="train, val or test set")

    parser.add_argument("--phasecorrfold",
                        type=str,
                        default=os.path.join(this_filedir, "..", "data", "phase_correlation"),
                        help="folder to look or save the phase correlation files")

    parser.add_argument("--win_size", type=int, default=5, help="temporal window size")

    parser.add_argument("--center", default=1, help="whether the window is centered")

    parser.add_argument("--score_thr", default=0.9, help="score thresholding filtering")

    parser.add_argument("--threshold",
                        default=0.5,
                        help="IoU threshold to consider boxes in/out image")

    parser.add_argument("--every",
                        default=1,
                        help="interval of frames to predict and use time consistency")

    parser.add_argument("--min_votes",
                        type=float,
                        default=0.7,
                        help="percentage of boxes in temporal window to consider FP or TP")

    parser.add_argument("--eval_every",
                        type=int,
                        default=24,
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

    config_file = os.path.join(this_filedir, "configs", "mosquitoes", model_name + ".yaml")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = os.path.join(this_filedir, "..", "output/v1_new", model_name,
                                     f"mbg_fold{fold}_train_{args.object}", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 600
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    data_path = "../data/v1"

    file_set = os.path.join(this_filedir, "../data/v1", "train_sets_kfold_v1.0.xls")
    df_set = pd.read_excel(file_set, sheet_name=f'train_sets_k{fold}')

    videos_names = df_set[df_set[f'{args.set}'] == True]['Video'].tolist()
    videos_names = [s + ".avi" for s in videos_names]

    scores = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    votes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    margin = (0, 0)

    res = {
        'score': [],
        'voting': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'Pr': [],
        'Rc': [],
        'F1': [],
        'AP50': [],
    }

    res_time = {
        'score': [],
        'voting': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'Pr': [],
        'Rc': [],
        'F1': [],
        'AP50': [],
    }

    res_tube = {
        'score': [],
        'voting': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'Pr': [],
        'Rc': [],
        'F1': [],
        'AP50': [],
    }

    for score_thr in scores:
        for min_votes in votes:

            res['score'].append(score_thr)
            res['voting'].append(min_votes)
            res_time['score'].append(score_thr)
            res_time['voting'].append(min_votes)
            res_tube['score'].append(score_thr)
            res_tube['voting'].append(min_votes)

            video_coco = Convert2Coco(category_dict={f'{args.object}': 0})
            out_idx = 0
            out = {}
            out_time = {}

            pipes_eval = PipesEval()
            pipes_eval.reset()

            for video_name in videos_names:
                video_path = os.path.join(this_filedir, data_path, 'videos', video_name)

                print('Predicting...')

                pred_file = cfg.MODEL.WEIGHTS.replace("model_final.pth",
                                                      f"{video_name.split('.')[0]}_preds.pkl")

                # PREDICTIONS
                # in case it was forwarded before load
                if os.path.isfile(pred_file):
                    print('loading predictions...')
                    pkl_file = open(pred_file, 'rb')
                    preds_loaded = pickle.load(pkl_file)
                # otherwise, run over the whole video
                else:
                    print('computing predictions...')
                    predictor = DefaultPredictor(cfg)
                    pkl_file = open(pred_file, 'wb')
                    print(video_path)
                    preds_loaded = run_on_video(video_path, predictor, every=1)
                    pickle.dump(preds_loaded, pkl_file)
                pkl_file.close()

                preds = {}
                for n, (k, v), in enumerate(preds_loaded.items()):
                    if (n % every) != 0:
                        continue
                    preds[k] = v

                # filter pred
                del preds_loaded  # free memory

                preds = filter_preds_score_video(preds, score_thr)
                preds = filter_preds_margin_video(preds, margin)

                # PHASE CORRELATION
                vid = videoObj(video_path,
                               video_path.replace(".avi", ".xml").replace('videos', 'annotation'))

                print('Phase correlation...')
                phase_corr_file = os.path.join(args.phasecorrfold,
                                               video_name.replace(".avi", "_phaseCorr.pkl"))

                # in case it was previously computed, load it
                if os.path.isfile(phase_corr_file):
                    print('loading phase corr...')
                    pkl_file = open(phase_corr_file, 'rb')
                    phase_corr = pickle.load(pkl_file)
                # otherwise, compute and save for future usage
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

                video_annotations = vid.get_annotations().filter_objects(objects=[f'{args.object}'])
                video_annotations = filter_annot_margin_video(video_annotations, (height, width),
                                                              margin)
                pipes_eval.process(video_annotations, filtered_pipes)

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
                    if len(gt) > 0:
                        gt = filter_gt_margin_image(gt, (height, width), margin)
                    video_coco.update(
                        os.path.join(video_name.split('.avi')[0], f'frame_{frame_idx:04d}.png'),
                        width, height, gt)

                    out[out_idx] = preds[f'frame_{frame_idx:04d}']
                    out_time[out_idx] = preds_time[f'frame_{frame_idx:04d}']
                    out_idx += 1

            with open(
                    os.path.join(this_filedir, "../output/time_consist/",
                                 f"preds_time_{model_name}.pkl"), 'wb') as outfile:
                pickle.dump(out_time, outfile)

            valid_data = f"preds_{model_name}_score{score_thr}_voting{min_votes}"
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
            os.remove(valid_data)

            print('Computing results...')
            for idx, inputs in enumerate(val_loader):
                outputs = out[idx]
                outputs_time = out_time[idx]

                evaluators.process([inputs], [outputs])
                evaluators_time.process([inputs], [outputs_time])

            results = evaluators.evaluate()
            results_time = evaluators_time.evaluate()
            results_tube = pipes_eval.evaluate()

            print('Prediction results without time consistency...')
            print(results)

            print('Prediction with TIME consistency results...')
            print(results_time)

            print('Prediction TUBES...')
            print(results_tube)

            print("saving results...")
            res_dir = os.path.join(this_filedir, "../output/time_consist")
            res_path = os.path.join(res_dir,
                                    f"{model_name}_size{win_size}_vot{int(min_votes*100)}.pkl")
            save_results(results_time, res_path)

            res_dir = os.path.join(this_filedir, "../output/tube")
            res_path = os.path.join(res_dir, f"{model_name}_vot{int(min_votes*100)}.pkl")
            save_results(results_tube, res_path)

            #### frame

            res['TP'].append(results['tp'])
            res['FP'].append(results['fp'])
            res['FN'].append(results['fn'])
            res['AP50'].append(results['bbox']['AP50'])

            pr = results['tp'] / (results['tp'] + results['fp'])
            rc = results['tp'] / (results['tp'] + results['fn'])
            f1 = (2 * pr * rc) / (pr + rc)

            res['Pr'].append(pr)
            res['Rc'].append(rc)
            res['F1'].append(f1)

            #### frame with time consist

            res_time['TP'].append(results_time['tp'])
            res_time['FP'].append(results_time['fp'])
            res_time['FN'].append(results_time['fn'])
            res_time['AP50'].append(results_time['bbox']['AP50'])

            pr = results_time['tp'] / (results_time['tp'] + results_time['fp'] + 1e-16)
            rc = results_time['tp'] / (results_time['tp'] + results_time['fn'] + 1e-16)
            f1 = (2 * pr * rc) / (pr + rc + 1e-16)

            res_time['Pr'].append(pr)
            res_time['Rc'].append(rc)
            res_time['F1'].append(f1)

            #### tube

            res_tube['TP'].append(results_tube['tp'])
            res_tube['FP'].append(results_tube['fp'])
            res_tube['FN'].append(results_tube['fn'])
            res_tube['AP50'].append(100 * results_tube['AP'])

            res_tube['Pr'].append(results_tube['precision'])
            res_tube['Rc'].append(results_tube['recall'])
            f1 = (2 * results_tube['precision'] * results_tube['recall']) / (
                results_tube['precision'] + results_tube['recall'] + 1e-16)
            res_tube['F1'].append(f1)

    df = pd.DataFrame(res)
    df_time = pd.DataFrame(res_time)
    df_tube = pd.DataFrame(res_tube)

    # df.to_csv(f'{args.object}_{model_name}_margin{margin[0]}.csv')
    # df_time.to_csv(f'{args.object}_{model_name}_margin{margin[0]}_time.csv')
    df_tube.to_csv(f'{args.object}_{model_name}_margin{margin[0]}_tube.csv')

    print("end")
