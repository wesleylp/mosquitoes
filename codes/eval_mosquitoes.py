import argparse
# import json
import os

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
                                  "faster_rcnn_R_50_C4_1x.yaml")

    data_dir_default = os.path.join(this_filepath, '..', 'data', '_under_construction')

    parser = argparse.ArgumentParser(description="MBG Eval")
    parser.add_argument("--config-file",
                        default=config_default,
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--data-dir", default=data_dir_default, metavar="FILE", help="path to data")
    parser.add_argument("--data-train",
                        default="mbg_fold0_set_2",
                        metavar="FILE",
                        help="path to data")

    parser.add_argument("--data-test", default="mbg_test", metavar="FILE", help="path to data")

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

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    val_loader = build_detection_test_loader(cfg, args.data_test)
    evaluator = COCOEvaluator(args.data_test, cfg, False, output_dir=cfg.OUTPUT_DIR)
    cfn_mat = CfnMat(args.data_test, output_dir=cfg.OUTPUT_DIR)

    inference_on_dataset(trainer.model, val_loader, DatasetEvaluators([evaluator, cfn_mat]))
