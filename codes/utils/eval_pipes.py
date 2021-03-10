import contextlib
import io
import logging
import os
from itertools import chain
import numpy as np

import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures.boxes import Boxes, BoxMode, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.utils.logger import create_small_table


class PipesEval(DatasetEvaluator):
    def __init__(self, thr=0.5, method='COCO'):
        self.thr = thr
        self._method = method
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._gt = []
        self._predictions = []
        self.nb_tp = 0
        self.nb_fp = 0
        self.nb_fn = 0
        self.nb_tn = -1
        self._video_id = 0

    def process(self, inputs, outputs):
        # List with all gts (Ex: [[video_id, {boxes, frames}], ...]
        self._gt.extend(
            list(chain.from_iterable(([self._video_id, ann], ) for ann in inputs.values())))

        # List with all preds (Ex: [[video_id, instances, confidence], ...]
        self._predictions.extend(
            list(
                chain.from_iterable(
                    ([self._video_id, inst, self._comp_tube_conf(inst)], )
                    for inst in outputs.values())))
        self._video_id += 1

    def evaluate(self):

        # append 0 in gts so that all detections are undected initialy
        [g.append(0) for g in self._gt]
        # append 0 in preds so that all detections are FPs initialy
        [p.append(0) for p in self._predictions]

        for vid_id in range(self._video_id):
            gts = [gt for gt in self._gt if gt[0] == vid_id]
            preds = [pred for pred in self._predictions if pred[0] == vid_id]

            # a bool is appended to gt to find out whether it is detected (1) or not (0)
            # a bool is appended to preds to find out whether it is TP (1) of FP (0)
            nb_tp, nb_fp, nb_fn = self._cnf_mat(preds, gts, thr=self.thr)

            assert (nb_tp + nb_fn) == len(gts)

            self.nb_tp += nb_tp
            self.nb_fp += nb_fp
            self.nb_fn += nb_fn

        # sort detections by decreasing confidence
        dects = sorted(self._predictions, key=lambda conf: conf[2], reverse=True)
        TP = np.array([d[3] for d in dects])
        FP = np.logical_not(TP).astype(int)

        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / len(self._gt)
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        if self._method.lower() == 'coco':
            [ap, mpre, mrec, ii] = PipesEval.npts_interp_AP(rec, prec, self._method)

        # res = {
        #     # "tp": self.nb_tp,
        #     # "fp": self.nb_fp,
        #     # "fn": self.nb_fn,
        #     # "tn": self.nb_tn,
        #     # "P": self.precision,
        #     # "R": self.recall,
        #     # "F1": self.f1,
        # }

        res = {
            # 'class': c,
            'AP': ap,
            # 'interpolated precision': mpre,
            # 'interpolated recall': mrec,
            # 'total positives': npos,
            # 'total TP': np.sum(TP),
            # 'total FP': np.sum(FP)
            "tp": self.nb_tp,
            "fp": self.nb_fp,
            "fn": self.nb_fn,
            'precision': self.nb_tp / (self.nb_tp + self.nb_fp),
            'recall': self.nb_tp / (self.nb_tp + self.nb_fn)
        }

        self._logger.info("Pipe Confusion matrix metrics: \n" + create_small_table(res))
        print("Confusion matrix metrics: \n" + create_small_table(res))

        return res

    def _cnf_mat(self, preds, gts, thr=0.5):

        gts_ = [gt[1] for gt in gts]
        preds_ = [pred[1] for pred in preds]

        gt_overlaps = torch.zeros(len(gts_))
        overlaps = self._pipe_pairwise_iou(preds_, gts_)

        for j in range(min(len(preds_), len(gts_))):

            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0

            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]

            if gt_ovr >= thr:
                gts[gt_ind][2] = 1
                preds[box_ind][3] = 1

            # record the iou coverage of this gt box
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr

            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        tp = (gt_overlaps >= thr).int().sum().item()
        assert tp >= 0

        fp = len(preds_) - tp
        assert fp >= 0

        fn = len(gts_) - tp
        assert fn >= 0

        return tp, fp, fn

    def _pipe_pairwise_iou(self, predictions, annots):
        inter = torch.zeros(len(predictions), len(annots))
        union = torch.zeros(len(predictions), len(annots))

        det_idx = 0
        for det in predictions:
            gt_idx = 0
            for gt in annots:
                inter[det_idx, gt_idx] = self._get_inter(det, gt)
                union[det_idx, gt_idx] = self._get_union(det, gt)

                gt_idx += 1

            det_idx += 1

        iou = inter / (union - inter)

        return iou

    def _get_intersection_frames(self, tube1, tube2):
        def get_frames(tube):
            if isinstance(tube, Instances):
                frames = tube.get('frames')
            else:
                frames = tube['frames']
            return frames

        frames_tube1 = set(get_frames(tube1))
        frames_tube2 = set(get_frames(tube2))

        frames_intersection = frames_tube1.intersection(frames_tube2)

        if len(frames_intersection) > 0:
            return frames_intersection

        return None

    def _get_inter(self, tube1, tube2):
        inter_frames = self._get_intersection_frames(tube1, tube2)

        inter = 0

        if inter_frames is not None:

            def get_tube_box_frame(tube, frame_idx):
                if isinstance(tube, Instances):
                    frames = tube.get('frames')
                    inst_idx = frames.index(frame_idx)
                    box = tube[inst_idx].get('pred_boxes')

                else:
                    frames = tube['frames']
                    inst_idx = frames.index(frame_idx)
                    box = tube['boxes'][inst_idx]
                    box = Boxes(torch.tensor([box], device='cuda'))

                return box

            for inter_frame in inter_frames:
                boxes1 = get_tube_box_frame(tube1, inter_frame)
                boxes2 = get_tube_box_frame(tube2, inter_frame)

                boxes1, boxes2 = boxes1.tensor, boxes2.tensor

                width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
                    boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

                width_height.clamp_(min=0)  # [N,M,2]
                inter += width_height.prod(dim=2)  # [N,M]
                del width_height

        return inter

    def _get_union(self, tube1, tube2):
        def get_boxes_areas(tube):
            if isinstance(tube, Instances):
                boxes = tube.get('pred_boxes')

            elif isinstance(tube, dict):
                boxes = Boxes(torch.tensor(tube['boxes'], device='cuda'))

            else:
                raise ValueError

            return boxes.area()

        area1 = get_boxes_areas(tube1).sum()
        area2 = get_boxes_areas(tube2).sum()

        union = area1 + area2

        return union

    # TODO: move this to time_consist.py?
    def _comp_tube_conf(self, tube):
        frames_detected = np.asarray(tube.get('detected'))
        return frames_detected.sum() / len(frames_detected)

    # n-point interpolated average precision
    @staticmethod
    def npts_interp_AP(rec, prec, method="coco"):

        n_pts = 101 if method.lower() == "coco" else 11

        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)

        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)

        recallValues = np.linspace(0, 1, n_pts)
        recallValues = list(recallValues[::-1])

        rhoInterp = []
        recallValid = []

        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)

            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])

            recallValid.append(r)
            rhoInterp.append(pmax)

        # By definition AP = sum(max(precision whose recall is above r))/n_pts
        ap = sum(rhoInterp) / n_pts

        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)

        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)

        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])

            if p not in cc:
                cc.append(p)

            p = (rvals[i], pvals[i])

            if p not in cc:
                cc.append(p)

        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]

        return [ap, rhoInterp, recallValues, None]

    @staticmethod
    def all_pts_AP(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)

        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)

        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
