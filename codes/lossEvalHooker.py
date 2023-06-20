import datetime
import logging
import os
import time
import detectron2.data.transforms as T
import numpy as np
import torch
import copy
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader, detection_utils
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase, PeriodicWriter
from detectron2.evaluation import COCOEvaluator, inference_context
from detectron2.utils.logger import log_every_n_seconds
# from codes.utils.evaluation import CfnMat
import albumentations as AB
import detectron2.data.transforms.external as A
from detectron2.structures import BoxMode, Boxes
from custom_augmentations import CutOut, AugmentHSV, RandomPerspective
import nni


class LossEvalHook(HookBase):

    def __init__(self, cfg, model, data_loader, checkpointer):
        self._model = model
        self._period = cfg.VAL_PERIOD
        self._data_loader = data_loader
        self._min_loss = 999
        self._best_iteration = -1
        self._checkpointer = checkpointer

    def _do_loss_eval(self):
        # copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        # evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0

        losses = []

        with torch.no_grad():

            for idx, inputs in enumerate(self._data_loader):
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_compute_time = 0

                start_compute_time = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                seconds_per_img = total_compute_time / iters_after_start
                if idx >= num_warmup * 2 or seconds_per_img > 5:
                    total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))

                    log_every_n_seconds(
                        logging.INFO,
                        "Loss on Validation done {}/{}. {:.4f} s / img. ETA={}".format(
                            idx + 1, total, seconds_per_img, str(eta)),
                        n=5,
                    )

                loss_batch = self._get_loss(inputs)
                losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)

        comm.synchronize()

        return mean_loss

    def _get_loss(self, data):
        metrics_dict = self._model(data)

        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        total_losses_reduced = sum(loss for loss in metrics_dict.values())

        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter

        if is_final or (self._period > 0 and next_iter % self._period == 0):
            mean_loss = self._do_loss_eval()

            # save best model
            if mean_loss < self._min_loss:
                self._min_loss = mean_loss
                self._best_iteration = next_iter
                self._checkpointer.save("model_best", iteration=self._best_iteration)

            if is_final:
                log_every_n_seconds(
                    logging.INFO,
                    "Best model at iteration {}. Loss {:.4f}.".format(self._best_iteration,
                                                                      self._min_loss),
                    n=5,
                )

        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg, self.model,
                build_detection_test_loader(self.cfg, self.cfg.DATASETS.VAL[0],
                                            DatasetMapper(self.cfg, True)), self.checkpointer))
        return hooks


# #### Compute loss in the validation set


class ValidationLoss(HookBase):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)

        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                "val_" + k: v.item()
                for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, **loss_dict_reduced)

            nni.report_intermediate_result(losses_reduced)

    def after_train(self):
        nni.report_final_result(self.trainer.storage.latest()["total_val_loss"][0])


class SaveModel(HookBase):

    def __init__(self, checkpointer, iters_to_save) -> None:
        super().__init__()
        self._checkpointer = checkpointer
        # R50FPN
        self.iters_to_save = iters_to_save

    def after_step(self):
        next_iter = self.trainer.iter + 1

        if next_iter in self.iters_to_save:
            self._checkpointer.save(f"model_{next_iter:07d}", iteration=next_iter)


class KeepPredsPeriod(HookBase):

    def __init__(self, folder_path, period) -> None:
        super().__init__()
        self.period = period
        self.folder_path = folder_path

    def after_step(self):
        next_iter = self.trainer.iter + 1

        if (next_iter % self.period) == 0:
            if os.path.isdir(self.folder_path):
                os.rename(self.folder_path, f"{self.folder_path}_iter{next_iter:05d}")


def build_train_aug(cfg):

    augs = [
        T.Albumentations(AB.Blur(p=0.01)),
        T.Albumentations(AB.MedianBlur(p=0.01)),
        T.Albumentations(AB.ToGray(p=0.01)),
        T.Albumentations(AB.CLAHE(p=0.01)),
        T.Albumentations(AB.RandomBrightnessContrast(p=0.0)),
        T.Albumentations(AB.RandomGamma(p=0.0)),
        T.Albumentations(AB.ImageCompression(quality_lower=75, p=0.0)),
    ]

    # augs = T.Albumentations(
    #     AB.MedianBlur(p=0.01),
    # AB.ToGray(p=0.01),
    # AB.CLAHE(p=0.01),
    # AB.RandomBrightnessContrast(p=0.0),
    # AB.RandomGamma(p=0.0),
    # AB.ImageCompression(quality_lower=75, p=0.0),
    # )
    #    T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
    #                          cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)

    # if cfg.INPUT.CROP.ENABLED:
    #     augs.append(
    #         T.RandomCrop_CategoryAreaConstraint(
    #             cfg.INPUT.CROP.TYPE,
    #             cfg.INPUT.CROP.SIZE,
    #             cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
    #             cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
    #         ))

    augs.append(AugmentHSV(hgain=0.015, sgain=0.7, vgain=0.4))
    augs.append(RandomPerspective(degrees=0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0))

    augs.append(T.RandomFlip())

    augs.append(
        T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                             cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING))
    return augs


class MyTrainer2(DefaultTrainer):

    def build_hooks(self):
        hooks = super().build_hooks()

        if self.cfg.DATASETS.VAL[0] is not None:
            hooks.insert(-1, ValidationLoss(self.cfg))
            hooks[-1] = PeriodicWriter(self.build_writers(), period=self.cfg.VAL_PERIOD)

        if self.cfg.iters_to_save[0] is not None:
            hooks.insert(-1, SaveModel(self.checkpointer, self.cfg.iters_to_save))

        # compute predictions
        # hooks.insert(
        #     -1,
        #     KeepPredsPeriod(os.path.join(self.cfg.OUTPUT_DIR, "inference"),
        #                     self.cfg.TEST.EVAL_PERIOD))

        # multi gpu
        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is remove
        # hooks = hooks[:-2] + hooks[-2:][::-1]

        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        coco_evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        # cfn_matx_evaluator = CfnMat(dataset_name, output_dir=output_folder)

        return [coco_evaluator]

    @classmethod
    def build_train_loader(cls, cfg):
        # TODO: make a control parameter to use augmentation
        if cfg.INPUT.CUSTOM_AUGMENTATIONS:
            print("USING EAAI CUSTOM AUGMENTATIONS!")
            mapper = CustomDatasetMapper(cfg, is_train=True, augmentations=build_train_aug(cfg))
        else:
            print("USING DEFAULT AUGMENTATIONS!")
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


class CustomDatasetMapper(DatasetMapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        h, w = image[:2]
        detection_utils.check_image_size(dataset_dict, image)

        # annos = dataset_dict["annotations"]
        # instances = detection_utils.annotations_to_instances(annos, image.shape[:2])

        # boxes = instances.get("gt_boxes").tensor

        # # Transform XYXY_ABS -> XYXY_REL
        # boxes = np.array(boxes) / np.array(
        #     [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        # augmentation in fact
        aug_input = T.AugInput(image)  #, boxes=boxes)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        # boxes = aug_input.boxes
        h, w = image[:2]

        # Transform XYXY_REL -> XYXY_ABS
        # boxes = np.array(boxes) * np.array(
        #     [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        # instances.gt_boxes = Boxes(boxes)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                detection_utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            instances = detection_utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format)

            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict
