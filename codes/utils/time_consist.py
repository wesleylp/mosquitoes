# import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm
import copy
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances

from .img_utils import phase_correlation


def clone_instances(instances):
    return Instances(instances.image_size, **instances.get_fields())


def pop_instances(instances, idx):
    return instances[torch.arange(len(instances)) != idx]


def video_phaseCorrelation(video_obj, scale=0.5):

    frame_generator = video_obj.frame_from_video()

    prev_frame = next(frame_generator)

    corr = dict()
    (x, y), c = phase_correlation(prev_frame, prev_frame, scale=scale)
    corr[0] = [x, y, c]

    for idx, frame in enumerate(tqdm(frame_generator,
                                     total=video_obj.videoInfo.getNumberOfFrames())):

        (x, y), c = phase_correlation(prev_frame, frame, scale=scale)
        corr[idx + 1] = [x, y, c]
        prev_frame = frame

    df = pd.DataFrame(corr).T
    df.columns = ['x', 'y', 'conf']
    df.head()

    return df


def get_instances_boxes(instances):
    return instances.get('pred_boxes')


def shift_instances(instances, shift, max_box_size=(2160, 3840)):
    instances_shifted = clone_instances(instances)

    boxes = get_instances_boxes(instances)
    boxes_shifted = shift_boxes(boxes, shift, max_box_size)

    instances_shifted.set('pred_boxes', boxes_shifted)

    return instances_shifted


def shift_boxes(boxes, shift, max_box_size=(2160, 3840)):
    boxes_shifted = boxes.clone()

    boxes_shifted.tensor[:, 0::2] += shift[0]
    boxes_shifted.tensor[:, 1::2] += shift[1]

    if max_box_size is not None:
        boxes_shifted.clip(max_box_size)

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


def get_missing_boxes(boxes_current, boxes_offset, shift, max_box_size=(2160, 3840), thr=0.5):

    # reprojeto as caixas do frame offset no frame de ref
    boxes_offset_ = shift_boxes(boxes_offset, shift, max_box_size)

    # verifico se todas as boxes do frame deslocado estão no frame atual
    overlaps = pairwise_iou(boxes_offset_, boxes_current)
    pred_ovr, pred_ind = overlaps.max(dim=1)

    boxes_missing = boxes_offset_[pred_ovr < thr]

    return boxes_missing


def get_missing_instances(instances_current,
                          instances_offset,
                          shift,
                          max_box_size=(2160, 3840),
                          thr=0.5):

    # reprojeto as caixas do frame offset no frame de ref
    instances_offset_ = shift_instances(instances_offset, shift, max_box_size)

    # verifico se todas as boxes do frame deslocado estão no frame atual
    boxes_offset_ = get_instances_boxes(instances_offset_)
    boxes_current = get_instances_boxes(instances_current)

    overlaps = pairwise_iou(boxes_offset_, boxes_current)
    pred_ovr, pred_ind = overlaps.max(dim=1)

    instances_missing = instances_offset_[pred_ovr < thr]

    return instances_missing


def frame_from_key(key):
    return int(key.split('_')[-1])


def get_frames_to_see(predictions, frame_ref, window_size, center=False):
    list_of_frames = [frame_from_key(k) for k in predictions.keys()]
    frame_ref_idx = list_of_frames.index(frame_ref)

    if center:
        return list_of_frames[frame_ref_idx - window_size // 2:frame_ref_idx + window_size // 2 + 1]

    return list_of_frames[frame_ref_idx - window_size + 1:frame_ref_idx + 1]

    # if center:
    #     return np.arange(frame_ref - window_size // 2, frame_ref + window_size // 2 + 1)
    # # causal window
    # return np.arange(frame_ref - window_size + 1, frame_ref + 1)


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


def voting(votes, min_votes=0.5, axis=0):
    sum_votes = votes.sum(axis=axis)
    final_votes = sum_votes > np.ceil(votes.shape[axis] * min_votes)

    return final_votes.int()


def get_frame_instances(frame_idx, predictions):
    instances = predictions[f'frame_{frame_idx:04d}']['instances']
    return instances


def get_pred_boxes_frame(frame_idx, predictions):
    instances = get_frame_instances(frame_idx, predictions)
    return get_instances_boxes(instances)


def get_instances_to_compare(frame_ref,
                             predictions,
                             df_shift,
                             max_box_size=(2160, 3840),
                             window_size=5,
                             thr=0.5,
                             center=False):
    assert isinstance(window_size, int)

    frames_to_see = get_frames_to_see(predictions, frame_ref, window_size, center=center)

    instances = get_frame_instances(frame_ref, predictions)

    # bbs_pred = get_pred_boxes_frame(frame_ref, predictions)
    # boxes_to_compare = bbs_pred.clone()

    instaces_to_compare = instances

    for frame_idx in frames_to_see:
        offset = frame_idx - frame_ref
        if offset == 0:
            continue

        # bbs_pred_offset = get_pred_boxes_frame(frame_ref + offset, predictions)
        instances_offset = get_frame_instances(frame_ref + offset, predictions)

        if len(instances_offset) == 0:
            continue

        shift = get_shift(df_shift, frame_ref + offset, -offset)

        # missing_boxes = get_missing_boxes(boxes_to_compare, bbs_pred_offset, shift, thr)
        missing_instaces = get_missing_instances(instaces_to_compare, instances_offset, shift,
                                                 max_box_size, thr)

        # boxes_to_compare = concat_boxes(boxes_to_compare, missing_boxes)
        instaces_to_compare = Instances.cat([instaces_to_compare, missing_instaces])

    return instaces_to_compare


def shift_all_frame_bboxes(predictions, df_shift, frame_from, n_frames, max_box_size=(2160, 3840)):

    bbs_pred = get_pred_boxes_frame(frame_from, predictions)
    total_shift = get_shift(df_shift, frame_from, n_frames)

    # offset the bboxes
    bbs_pred_offset = shift_boxes(bbs_pred, total_shift, max_box_size)

    # clamp the boxes with images limits
    # bbs_pred_offset.clip(max_box_size)

    return bbs_pred_offset


def apply_time_consistency(predictions,
                           df_shift,
                           vid_height_width,
                           win_size=5,
                           center=True,
                           thr=0.5,
                           min_votes='majority'):
    new_preds = dict()

    if center:
        first_frame = win_size // 2
        last_frame = len(predictions) - first_frame
    else:
        first_frame = win_size - 1
        last_frame = len(predictions)

    ### Using sampling frames in predictions ###
    for cnt, k in enumerate(predictions.keys()):

        if cnt < first_frame:
            # just copy the instances when I don't have a complete window
            new_preds[k] = dict()
            new_preds[k]['instances'] = get_frame_instances(frame_from_key(k), predictions)
            continue

        if cnt <= last_frame:

            bbs_pred = get_pred_boxes_frame(frame_from_key(k), predictions)
            if len(bbs_pred) == 0:
                new_preds[k] = dict()
                new_preds[k]['instances'] = get_frame_instances(frame_from_key(k), predictions)
                continue

            frame_ref = frame_from_key(k)
            all_candidates = get_instances_to_compare(frame_ref,
                                                      predictions,
                                                      df_shift,
                                                      vid_height_width,
                                                      window_size=win_size,
                                                      thr=thr,
                                                      center=center)

            votes = torch.zeros((win_size, len(all_candidates)))

            frames_to_see = get_frames_to_see(predictions, frame_ref, win_size, center=center)

            for idx, frame_idx in enumerate(frames_to_see):
                current_boxes = get_pred_boxes_frame(frame_idx, predictions)

                offset = frame_idx - frame_ref
                shift = get_shift(df_shift, frame_ref, offset)

                candidates_offset = shift_boxes(all_candidates.get('pred_boxes'),
                                                shift,
                                                max_box_size=vid_height_width)

                frame_votes = compare_boxes(current_boxes, candidates_offset, thr=0.5)

                votes[idx, :] = frame_votes

            final_votes = voting(votes, min_votes)

            res = all_candidates[final_votes.nonzero().flatten()]

            new_preds[k] = dict()
            new_preds[k]['instances'] = res
            continue

        else:
            new_preds[k] = dict()
            new_preds[k]['instances'] = get_frame_instances(frame_from_key(k), predictions)
            continue

    ###########################

    # # just copy the instances when I don't have a complete window
    # for frame_ref in range(0, first_frame):
    #     new_preds[f'frame_{frame_ref:04d}'] = dict()
    #     new_preds[f'frame_{frame_ref:04d}']['instances'] = get_frame_instances(
    #         frame_ref, predictions)

    # for frame_ref in tqdm(range(first_frame, last_frame), total=last_frame):

    #     bbs_pred = get_pred_boxes_frame(frame_ref, predictions)
    #     if len(bbs_pred) == 0:
    #         new_preds[f'frame_{frame_ref:04d}'] = dict()
    #         new_preds[f'frame_{frame_ref:04d}']['instances'] = get_frame_instances(
    #             frame_ref, predictions)
    #         continue

    #     all_candidates = get_instances_to_compare(
    #         frame_ref, predictions, df_shift, window_size=win_size, thr=thr, center=center)

    #     votes = torch.zeros((win_size, len(all_candidates)))

    #     frames_to_see = get_frames_to_see(frame_ref, win_size, center=center)

    #     for idx, frame_idx in enumerate(frames_to_see):
    #         current_boxes = get_pred_boxes_frame(frame_idx, predictions)

    #         offset = frame_idx - frame_ref
    #         shift = get_shift(df_shift, frame_ref, offset)

    #         candidates_offset = shift_boxes(all_candidates.get('pred_boxes'), shift)

    #         frame_votes = compare_boxes(current_boxes, candidates_offset, thr=0.5)

    #         votes[idx, :] = frame_votes

    #     final_votes = get_majority(votes)
    #     res = all_candidates[final_votes.nonzero().flatten()]

    #     new_preds[f'frame_{frame_ref:04d}'] = dict()
    #     new_preds[f'frame_{frame_ref:04d}']['instances'] = res

    # for frame_ref in range(last_frame, len(predictions)):
    #     new_preds[f'frame_{frame_ref:04d}'] = dict()
    #     new_preds[f'frame_{frame_ref:04d}']['instances'] = get_frame_instances(
    #         frame_ref, predictions)

    return new_preds


def tube_space_time(all_preds, df, thr=0.5):
    pipe = dict()

    h = 2160
    w = 3840

    def _scan(direction=1):

        pipe_parts = []
        detected = []
        frame_idx_det = []
        offset = 0
        frame_nb = frame_idx
        instance_to_compare = clone_instances(instance)

        while True:

            offset += direction

            try:
                instances_offset_frame = get_frame_instances(frame_nb + direction, all_preds)
            except KeyError:
                break

            # get shift between frames
            shift = get_shift(df, frame_nb, direction)

            instance_shifted = shift_instances(instance_to_compare, shift)

            if len(instances_offset_frame) == 0:
                iou = torch.tensor([0], dtype=torch.float32)
            else:
                iou = pairwise_iou(get_instances_boxes(instances_offset_frame),
                                   get_instances_boxes(instance_shifted))

            if iou.max() > thr:
                detected.append(1)
                arg_max_ovr = iou.argmax().item()

                # get instance to make part of tube
                pipe_parts.append(instances_offset_frame[arg_max_ovr])

                instance_to_compare = clone_instances(instances_offset_frame[arg_max_ovr])

                # remove from list of boxes to see
                all_preds[f'frame_{(frame_nb + direction):04d}']['instances'] =\
                    pop_instances(all_preds[f'frame_{(frame_nb + direction):04d}']['instances'], arg_max_ovr)

            else:

                boxes_shifted_ = get_instances_boxes(instance_shifted).tensor
                # check if box is in image

                if (boxes_shifted_[:, 0::2] >= w).all() or (boxes_shifted_[:, 0::2] <= 0).all() or\
                    (boxes_shifted_[:, 1::2] >= h).all() or (boxes_shifted_[:, 1::2] <= 0).all():
                    break

                else:
                    detected.append(0)
                    pipe_parts.append(instance_shifted)
                    instance_to_compare = clone_instances(instance_shifted)

            frame_idx_det.append(frame_nb + direction)
            frame_nb += direction

        if len(pipe_parts) == 0:
            pipe_conn = Instances(instance_to_compare[0].image_size)
            pipe_conn.set('pred_boxes', Boxes(torch.empty(0, 4, dtype=torch.float32,
                                                          device='cuda')))
            pipe_conn.set('scores', torch.empty(0, dtype=torch.float32, device='cuda'))
            pipe_conn.set('pred_classes', torch.empty(0, dtype=torch.int64, device='cuda'))
            pipe_conn.set('detected', detected)
            pipe_conn.set('frames', frame_idx_det)

            # pipe_conn = clone_instances(instances_offset_frame)
            # pipe_conn.set('detected', detected)
            # pipe_conn.set('frames', frame_idx_det)
            return pipe_conn

        if direction == -1:
            pipe_parts = [p for p in reversed(pipe_parts)]
            frame_idx_det = [f for f in reversed(frame_idx_det)]
            detected = [d for d in reversed(detected)]

        pipe_conn = Instances.cat(pipe_parts)
        pipe_conn.set('detected', detected)
        pipe_conn.set('frames', frame_idx_det)

        return pipe_conn

    def get_pipe(instance):
        pipe_init = clone_instances(instance)
        pipe_init.set('detected', [1])
        pipe_init.set('frames', [frame_idx])

        # before = len(all_preds[frame]['instances'])
        # remove from list of boxes to see
        all_preds[frame]['instances'] = pop_instances(all_preds[frame]['instances'], 0)
        # print(frame, before, len(all_preds[frame]['instances']))

        pipe_forward = _scan(direction=1)
        pipe_backward = _scan(direction=-1)

        return Instances.cat([pipe_backward, pipe_init, pipe_forward])

    pipe_idx = 0
    for frame in all_preds.keys():

        frame_idx = frame_from_key(frame)
        instances = all_preds[frame]['instances']

        if len(instances) == 0:
            continue

        for inst_idx in range(len(instances)):
            instance = instances[inst_idx]
            pipe[pipe_idx] = get_pipe(instance)
            pipe_idx += 1

    return pipe


def cut_pipe(p, qt=0.1):
    to_cut = int(len(p) * qt)
    return p[to_cut:-to_cut]


def filter_pipes(pipes, cut_exts=0, thr=0.5):
    filt_pipe = dict()
    idx = 0

    for k, p in pipes.items():
        if cut_exts > 0:
            p = cut_pipe(p, cut_exts)

        appear = sum(p.get('detected')) / (len(p) + 1e-18)
        if appear >= thr:
            filt_pipe[idx] = p
            idx += 1
            # print(k, appear)

    return filt_pipe


def pipe_to_frame_instances(preds, vid):
    pipes = copy.deepcopy(preds)
    pred_frame = dict()
    for frame_idx in range(vid.videoInfo.getNumberOfFrames()):
        # print(frame_idx)
        frame_instances = []

        for conn in pipes.values():
            frame_appear = conn.get('frames')

            if frame_idx in frame_appear:
                frame_instances.append(conn[frame_appear.index(frame_idx)])
                # print(frame_idx, len(frame_instances))

        if len(frame_instances) == 0:
            frame_instances = Instances((2160, 3840))
            frame_instances.set('pred_boxes',
                                Boxes(torch.empty(0, 4, dtype=torch.float32, device='cuda')))
            frame_instances.set('scores', torch.empty(0, dtype=torch.float32, device='cuda'))
            frame_instances.set('pred_classes', torch.empty(0, dtype=torch.int64, device='cuda'))

        else:
            frame_instances = Instances.cat(frame_instances)

        pred_frame[f'frame_{frame_idx:04d}'] = dict()
        pred_frame[f'frame_{frame_idx:04d}']['instances'] = frame_instances

    return pred_frame
