import datetime
import logging
import os
import time

import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase, PeriodicWriter
from detectron2.evaluation import COCOEvaluator, inference_context
from detectron2.utils.logger import log_every_n_seconds


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


class MyTrainer2(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, ValidationLoss(self.cfg))

        hooks[-1] = PeriodicWriter(self.build_writers(), period=self.cfg.VAL_PERIOD)

        # multi gpu
        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        # hooks = hooks[:-2] + hooks[-2:][::-1]

        return hooks
