import argparse
# import json
import os
import pandas as pd
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog, build_detection_test_loader)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluators, inference_on_dataset)
from detectron2.utils.logger import setup_logger
from utils.evaluation import CfnMat
from utils.register_datasets import register_mosquitoes

# import random

# import cv2
# import numpy as np

# from detectron2.utils.visualizer import Visualizer

if __name__ == "__main__":

    this_filepath = os.path.dirname(os.path.abspath(__file__))
    config_default = os.path.join(this_filepath, "configs", "mosquitoes",
                                  "faster_rcnn_R_50_FPN_1x.yaml")

    data_dir_default = os.path.join(this_filepath, '..', 'data', 'v1')

    parser = argparse.ArgumentParser(description="MBG Eval")
    parser.add_argument("--config-file",
                        default=config_default,
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--data-dir", default=data_dir_default, metavar="FILE", help="path to data")
    parser.add_argument("--data-train",
                        default="mbg_mosaic_train1_tire",
                        metavar="FILE",
                        help="path to data")

    parser.add_argument("--model_iter",
                        default=1146,
                        help="model iteration to be evaluated. May be an int or `final`.")

    parser.add_argument("--data-test", default="mbg_val1_tire", metavar="FILE", help="path to data")

    args = parser.parse_args()

    cfg = get_cfg()

    config_file = args.config_file

    cfg.merge_from_file(config_file)

    # register mosquitoes datasets
    register_mosquitoes()

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.data_train)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    setup_logger(cfg.OUTPUT_DIR)

    cfg.DATASETS.TRAIN = (args.data_train, )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 600
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (6000, 8000)
    cfg.SOLVER.IMS_PER_BATCH = 4

    # cfg.MODEL.DEVICE = "cpu"
    try:
        model_iter = f"{int(args.model_iter):07d}"
    except ValueError:
        model_iter = args.model_iter

    obj = args.data_test.split('_')[-1]
    model_name = os.path.splitext(os.path.basename(args.config_file))[0]
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"model_{model_iter}.pth")

    def init_res_dict():
        return {
            'score': [],
            'TP': [],
            'FP': [],
            'FN': [],
            'Pr': [],
            'Rc': [],
            'F1': [],
            'AP50': [],
        }

    res = init_res_dict()

    scores = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    # scores = [0.9]

    for score in scores:

        res['score'].append(score)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score  # set the testing

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)

        val_loader = build_detection_test_loader(cfg, args.data_test)
        evaluator = COCOEvaluator(args.data_test,
                                  cfg,
                                  False,
                                  output_dir=os.path.join(cfg.OUTPUT_DIR, args.data_test))
        cfn_mat = CfnMat(args.data_test, output_dir=cfg.OUTPUT_DIR)

        # inference_on_dataset(trainer.model, val_loader, DatasetEvaluators([evaluator, cfn_mat]))

        results = inference_on_dataset(trainer.model, val_loader,
                                       DatasetEvaluators([evaluator, cfn_mat]))

        res['TP'].append(results['tp'])
        res['FP'].append(results['fp'])
        res['FN'].append(results['fn'])
        res['AP50'].append(results['bbox']['AP50'])

        pr = results['tp'] / (results['tp'] + results['fp'] + 1e-16)
        rc = results['tp'] / (results['tp'] + results['fn'] + 1e-16)
        f1 = (2 * pr * rc) / (pr + rc + 1e-16)

        res['Pr'].append(pr)
        res['Rc'].append(rc)
        res['F1'].append(f1)

    df = pd.DataFrame(res)

    save_results_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
    name_base = f'{obj}_{model_name}_model_{model_iter}_{args.data_test.split("_")[-2]}'
    print(df)

    df.to_csv(os.path.join(save_results_dir, name_base + '.csv'))
