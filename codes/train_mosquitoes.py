import argparse
from mlflow import log_metrics, log_param, log_artifact

import json
import os
import random

import cv2
from mlflow.tracking.fluent import log_metrics
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog, build_detection_test_loader)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluators, inference_on_dataset)
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from lossEvalHooker import MyTrainer2 as MyTrainer
from utils.evaluation import CfnMat
from utils.register_datasets import register_mosquitoes


def train(cfg, resume=False):

    trainer = MyTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    return trainer


if __name__ == "__main__":

    this_filepath = os.path.dirname(os.path.abspath(__file__))
    config_default = os.path.join(this_filepath, "configs", "mosquitoes",
                                  "faster_rcnn_R_50_FPN_1x.yaml")
    data_dir_default = os.path.join(this_filepath, '..', 'data', '_under_construction')

    parser = argparse.ArgumentParser(description="MBG Training")
    parser.add_argument("--config-file",
                        default=config_default,
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--data-dir", default=data_dir_default, metavar="FILE", help="path to data")
    parser.add_argument("--data-train",
                        default="mbg_train+val_tire",
                        metavar="FILE",
                        help="path to data")
    parser.add_argument("--data-val", default=None, metavar="FILE", help="path to data")

    parser.add_argument("--data-test", default="mbg_test_tire", metavar="FILE", help="path to data")
    parser.add_argument("--weights", default=None, help="path to model weights")
    parser.add_argument("--iters", default=10000, help="Max number of iterations")

    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    cfg = get_cfg()
    config_file = args.config_file
    cfg.merge_from_file(config_file)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # setup_logger(cfg.OUTPUT_DIR)

    # register datasets
    register_mosquitoes()

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, args.data_train)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(cfg.OUTPUT_DIR)

    cfg.DATASETS.TRAIN = (args.data_train, )
    cfg.DATASETS.VAL = (args.data_val, )
    cfg.DATASETS.TEST = (args.data_test,
                         )  # ("mbg_test", )  # no metrics implemented for this dataset
    cfg.TEST.EVAL_PERIOD = 0  # 100
    cfg.VAL_PERIOD = 1  # 100

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 600
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MAX_ITER = int(args.iters)  #18000
    # cfg.iters_to_save = [None]
    cfg.iters_to_save = [1431, 7642, 8202, 13504]

    cfg.SOLVER.STEPS = (6000, 8000)  #(12000, 16000)
    cfg.SOLVER.IMS_PER_BATCH = 4

    if args.weights is not None:
        cfg.MODEL.WEIGHTS = args.weights

    # save training configuration
    with open(os.path.join(cfg.OUTPUT_DIR, 'training_config.yaml'), 'w') as tc:
        cfg.dump(stream=tc)
    log_artifact(os.path.join(cfg.OUTPUT_DIR, 'training_config.yaml'))

    trainer = train(cfg, resume=args.resume)

    # eval model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    log_artifact(cfg.MODEL.WEIGHTS)

    trainer = MyTrainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator(args.data_train, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, args.data_train)
    cfn_mat = CfnMat(args.data_train, output_dir=cfg.OUTPUT_DIR)
    res = inference_on_dataset(trainer.model, val_loader, DatasetEvaluators([evaluator, cfn_mat]))
    ap = res.pop('bbox', None)
    res.update(ap)
    res = {'train/' + k: v for k, v in res.items()}
    log_metrics(res)

    if args.data_val is not None:
        evaluator = COCOEvaluator(args.data_val, cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, args.data_val)
        cfn_mat = CfnMat(args.data_val, output_dir=cfg.OUTPUT_DIR)
        res = inference_on_dataset(trainer.model, val_loader,
                                   DatasetEvaluators([evaluator, cfn_mat]))
        res = {'val/' + k: v for k, v in res.items()}
        ap = res.pop('bbox', None)
        res.update(ap)
        log_metrics(res)

    evaluator = COCOEvaluator(args.data_test, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, args.data_test)
    cfn_mat = CfnMat(args.data_test, output_dir=cfg.OUTPUT_DIR)
    res = inference_on_dataset(trainer.model, val_loader, DatasetEvaluators([evaluator, cfn_mat]))
    ap = res.pop('bbox', None)
    res.update(ap)
    res = {'test/' + k: v for k, v in res.items()}
    log_metrics(res)
