import argparse
import json
import os
import random

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


def train(cfg):

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return trainer


if __name__ == "__main__":

    this_filepath = os.path.dirname(os.path.abspath(__file__))

    config_default = os.path.join(this_filepath, "configs", "mosquitoes",
                                  "faster_rcnn_R_50_C4_1x.yaml")

    data_dir_default = os.path.join(this_filepath, '..', 'data', '_under_construction')

    parser = argparse.ArgumentParser(description="MBG Training")
    parser.add_argument("--config-file",
                        default=config_default,
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--data-dir", default=data_dir_default, metavar="FILE", help="path to data")

    args = parser.parse_args()

    cfg = get_cfg()

    config_file = args.config_file

    cfg.merge_from_file(config_file)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    setup_logger(cfg.OUTPUT_DIR)

    for d in ["Train", "Test"]:
        register_coco_instances(f"mbg_{d.lower()}", {},
                                os.path.join(args.data_dir, f"coco_format_{d}.json"),
                                os.path.join(args.data_dir, d))

        MetadataCatalog.get(f"mbg_{d.lower()}").set(thing_classes=[
            'tire',
            # 'pool',
            # 'bucket',
            # 'water_tank',
            # 'puddle',
            # 'bottle',
        ])

        cdc_metadata = MetadataCatalog.get(f"mbg_{d.lower()}")

    cfg.DATASETS.TRAIN = ("mbg_train", )
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 600
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 300
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MAX_ITER = 10000  #18000
    cfg.SOLVER.STEPS = (6000, 8000)  #(12000, 16000)
    cfg.SOLVER.IMS_PER_BATCH = 4

    trainer = train(cfg)

    evaluator = COCOEvaluator("mbg_train", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "mbg_train")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    # evaluator = COCOEvaluator("mbg_valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # val_loader = build_detection_test_loader(cfg, "mbg_valid")
    # inference_on_dataset(trainer.model, val_loader, evaluator)

    evaluator = COCOEvaluator("mbg_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "mbg_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)