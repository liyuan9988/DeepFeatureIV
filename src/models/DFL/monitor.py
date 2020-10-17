from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
import torch
import logging

from src.data.data_class import TrainDataSetTorch, TestDataSetTorch
from src.models.DFIV.model import DFIVModel
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred

logger = logging.getLogger()
if TYPE_CHECKING:
    from src.models.DFL.trainer import DFLTrainer


class DFLMonitor:
    train_data_t: TrainDataSetTorch
    test_data_t: TestDataSetTorch
    validation_data_t: TrainDataSetTorch

    def __init__(self, dump_folder: Path, trainer: DFLTrainer):

        self.metrics = {"stage1_insample_loss": [],
                        "stage1_outsample_loss": [],
                        "stage2_insample_loss": [],
                        "stage2_outsample_loss": [],
                        "test_loss": []}

        self.dump_folder: Path = dump_folder
        self.trainer: DFLTrainer = trainer

    def configure_data(self, train_data_t: TrainDataSetTorch,
                       test_data_t: TestDataSetTorch,
                       validation_data_t: TrainDataSetTorch):

        self.train_data_t = train_data_t
        self.test_data_t = test_data_t
        self.validation_data_t = validation_data_t

    def record(self, verbose: int):
        self.trainer.treatment_net.train(False)
        if self.trainer.covariate_net is not None:
            self.trainer.covariate_net.train(False)

        n_train_data = self.train_data_t.treatment.size()[0]
        n_val_data = self.validation_data_t.treatment.size()[0]
        n_test_data = self.test_data_t.treatment.size()[0]
        with torch.no_grad():
            treatment_train_feature = self.trainer.treatment_net(self.train_data_t.treatment)
            treatment_val_feature = self.trainer.treatment_net(self.validation_data_t.treatment)
            treatment_test_feature = self.trainer.treatment_net(self.test_data_t.treatment)

            covariate_train_feature = None
            covariate_val_feature = None
            covariate_test_feature = None
            if self.trainer.covariate_net is not None:
                covariate_train_feature = self.trainer.covariate_net(self.train_data_t.covariate)
                covariate_val_feature = self.trainer.covariate_net(self.validation_data_t.covariate)
                covariate_test_feature = self.trainer.covariate_net(self.test_data_t.covariate)

            # stage2
            feature = DFIVModel.augment_stage2_feature(treatment_train_feature,
                                                       covariate_train_feature,
                                                       self.trainer.add_intercept)

            weight = fit_linear(self.train_data_t.outcome, feature, self.trainer.lam)
            insample_pred = linear_reg_pred(feature, weight)
            insample_loss = torch.norm(self.train_data_t.outcome - insample_pred) ** 2 / n_train_data

            val_feature = DFIVModel.augment_stage2_feature(treatment_val_feature,
                                                           covariate_val_feature,
                                                           self.trainer.add_intercept)
            outsample_pred = linear_reg_pred(val_feature, weight)
            outsample_loss = torch.norm(self.validation_data_t.outcome - outsample_pred) ** 2 / n_val_data

            # eval for test
            test_feature = DFIVModel.augment_stage2_feature(treatment_test_feature,
                                                            covariate_test_feature,
                                                            self.trainer.add_intercept)
            test_pred = linear_reg_pred(test_feature, weight)
            test_loss = torch.norm(self.test_data_t.structural - test_pred) ** 2 / n_test_data

            if verbose >= 1:
                logger.info(f"insample_loss:{insample_loss.item()}")
                logger.info(f"outsample_loss:{outsample_loss.item()}")
                logger.info(f"test_loss:{test_loss.item()}")
