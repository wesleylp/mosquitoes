import contextlib
import io
import logging
import os

import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures.boxes import Boxes, BoxMode, pairwise_iou


class CfnMat(DatasetEvaluator):
    def __init__(self, dataset_name, thr=0.5, output_dir=None):
        self.thr = thr

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = -1

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            prediction["instances"] = output["instances"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):

        self._logger.info("Evaluating confusion matrix ...")
        imgs_errors = []

        for pred in self._predictions:

            pred_boxes = pred["instances"].pred_boxes

            ann_ids = self._coco_api.getAnnIds(imgIds=pred["image_id"])
            anno = self._coco_api.loadAnns(ann_ids)

            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in anno
                if obj["iscrowd"] == 0
            ]

            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)

            tp, fp, fn = self._cnf_mat(pred_boxes, gt_boxes, thr=self.thr)

            assert (tp + fn) == len(gt_boxes)

            if fp or fn:
                imgs_errors.append({
                    "img_id": pred["image_id"],
                    "gt": len(gt_boxes),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

            self.tp += tp
            self.fp += fp
            self.fn += fn

        for img_errors in imgs_errors:
            print(img_errors)

        print("True positive:{} ".format(self.tp))
        print("False positive: {} ".format(self.fp))
        print("False negative: {} ".format(self.fn))

        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}

    def _cnf_mat(self, pred, gt, thr=0.5):
        gt_overlaps = torch.zeros(len(gt))
        overlaps = pairwise_iou(pred, gt)

        for j in range(min(len(pred), len(gt))):

            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0

            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]

            # record the iou coverage of this gt box
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr

            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        tp = (gt_overlaps >= thr).int().sum().item()
        assert tp >= 0

        fp = len(pred) - tp
        assert fp >= 0

        fn = len(gt) - tp
        assert fn >= 0

        return tp, fp, fn
