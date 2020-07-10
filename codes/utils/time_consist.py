import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm

from detectron2.structures.boxes import Boxes, pairwise_iou
from utils.img_utils import phase_correlation


def video_phaseCorrelation(video_obj, scale=0.5):

    frame_generator = video_obj.frame_from_video()

    prev_frame = next(frame_generator)

    corr = dict()
    (x, y), c = phase_correlation(prev_frame, prev_frame, scale=scale)
    corr[0] = [x, y, c]

    for idx, frame in enumerate(
            tqdm(frame_generator, total=video_obj.videoInfo.getNumberOfFrames())):

        (x, y), c = phase_correlation(prev_frame, frame, scale=scale)
        corr[idx + 1] = [x, y, c]
        prev_frame = frame

    df = pd.DataFrame(corr).T
    df.columns = ['x', 'y', 'conf']
    df.head()

    return df


def shift_boxes(boxes, shift):
    boxes_shifted = boxes.clone()

    boxes_shifted.tensor[:, 0::2] += shift[0]
    boxes_shifted.tensor[:, 1::2] += shift[1]

    return boxes_shifted


def get_shift(df_shift, frame_from, n_frames):
    desloc = df_shift.loc[min(frame_from, frame_from + n_frames) +
                          1:max(frame_from, frame_from + n_frames)]

    total_shift_x = desloc.sum()['x']
    total_shift_y = desloc.sum()['y']

    if n_frames < 0:
        total_shift_x *= -1
        total_shift_y *= -1

    return total_shift_x, total_shift_y


def concat_boxes(boxes1, boxes2):
    list_boxes = [boxes1, boxes2]
    return Boxes(torch.cat([box.tensor for box in list_boxes], dim=0))


def get_missing_boxes(boxes_current, boxes_offset, shift, thr=0.5):

    # reprojeto as caixas do frame offset no frame de ref
    boxes_offset_ = shift_boxes(boxes_offset, shift)

    # verifico se todas as boxes do frame deslocado estão no frame atual
    overlaps = pairwise_iou(boxes_offset_, boxes_current)
    pred_ovr, pred_ind = overlaps.max(dim=1)

    boxes_missing = boxes_offset_[pred_ovr < thr]

    return boxes_missing


def get_frames_to_see(frame_ref, window_size, center=False):
    if center:
        return np.arange(frame_ref - window_size // 2, frame_ref + window_size // 2 + 1)
    # causal window
    return np.arange(frame_ref - window_size + 1, frame_ref + 1)


def compare_boxes(bbs_current, bbs_to_compare, thr=0.5):
    overlaps = pairwise_iou(bbs_current, bbs_to_compare)
    gt_overlaps = torch.zeros(len(bbs_to_compare))

    for j in range(min(len(bbs_to_compare), len(bbs_current))):
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # find which gt box is 'best' covered (i.e. 'best' = most iou)
        gt_ovr, gt_ind = max_overlaps.max(dim=0)
        assert gt_ovr >= 0

        # find the proposal box that covers the best covered gt box
        box_ind = argmax_overlaps[gt_ind]

        # record the iou coverage of this gt box
        gt_overlaps[gt_ind] = overlaps[box_ind, gt_ind]
        assert gt_overlaps[gt_ind] == gt_ovr

        # mark the proposal box and the gt box as used
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1

    return (gt_overlaps >= thr).int()


def get_majority(votes, axis=0):
    sum_votes = votes.sum(axis=axis)
    final_votes = sum_votes > votes.shape[axis] / 2

    return final_votes.int()


def get_pred_boxes_frame(frame_idx, predictions):
    instances = predictions[f'frame_{frame_idx:04d}']['instances']
    return instances.get('pred_boxes').to(device='cpu')


def get_boxes_to_compare(frame_ref, predictions, df_shift, window_size=5, thr=0.5, center=False):
    assert isinstance(window_size, int)

    frames_to_see = get_frames_to_see(frame_ref, window_size, center=center)

    bbs_pred = get_pred_boxes_frame(frame_ref, predictions)
    boxes_to_compare = bbs_pred.clone()

    for frame_idx in frames_to_see:
        offset = frame_idx - frame_ref
        if offset == 0:
            continue

        bbs_pred_offset = get_pred_boxes_frame(frame_ref + offset, predictions)
        if len(bbs_pred_offset) == 0:
            continue

        shift = get_shift(df_shift, frame_ref + offset, -offset)
        missing_boxes = get_missing_boxes(boxes_to_compare, bbs_pred_offset, shift, thr)
        boxes_to_compare = concat_boxes(boxes_to_compare, missing_boxes)

    return boxes_to_compare


def shift_all_frame_bboxes(predictions, df_shift, frame_from, n_frames, max_box_size=(2160, 3840)):

    bbs_pred = get_pred_boxes_frame(frame_from, predictions)
    total_shift = get_shift(df_shift, frame_from, n_frames)

    # offset the bboxes
    bbs_pred_offset = shift_boxes(bbs_pred, total_shift)

    # clamp the boxes with images limits
    bbs_pred_offset.clip(max_box_size)

    return bbs_pred_offset


def apply_time_consistency(predictions, df_shift, win_size=5, center=True, thr=0.5):

    if center:
        first_frame = win_size // 2
        last_frame = len(predictions) - first_frame
    else:
        first_frame = win_size - 1
        last_frame = len(predictions)

    new_preds = dict()
    for frame_ref in tqdm(range(first_frame, last_frame), total=last_frame):

        bbs_pred = get_pred_boxes_frame(frame_ref, predictions)
        if len(bbs_pred) == 0:
            new_preds[f'frame_{frame_ref:04d}'] = bbs_pred
            continue

        all_candidates = get_boxes_to_compare(
            frame_ref, predictions, df_shift, window_size=win_size, thr=thr, center=center)

        votes = torch.zeros((win_size, len(all_candidates)))

        frames_to_see = get_frames_to_see(frame_ref, win_size, center=center)

        for idx, frame_idx in enumerate(frames_to_see):
            current_boxes = get_pred_boxes_frame(frame_idx, predictions)

            offset = frame_idx - frame_ref
            shift = get_shift(df_shift, frame_ref, offset)

            candidates_offset = shift_boxes(all_candidates, shift)

            frame_votes = compare_boxes(current_boxes, candidates_offset, thr=0.5)

            votes[idx, :] = frame_votes

        final_votes = get_majority(votes)
        res = all_candidates[final_votes.nonzero().flatten(), :]

        new_preds[f'frame_{frame_ref:04d}'] = res

    return new_preds
