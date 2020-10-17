from __future__ import annotations
from typing import List, Optional,TYPE_CHECKING
from pathlib import Path
import torch
import logging

from src.data.data_class import TrainDataSetTorch, TestDataSetTorch
from src.models.DFIV.model import DFIVModel
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred


logger = logging.getLogger()
if TYPE_CHECKING:
    from src.models.DFIV.trainer import DFIVTrainer


class DFIVMonitor:

    def __init__(self, dump_folder: Path, trainer: DFIVTrainer):

        self.metrics = {"stage1_insample_loss": [],
                        "stage1_outsample_loss": [],
                        "stage2_insample_loss": [],
                        "stage2_outsample_loss": [],
                        "test_loss": []}

        self.dump_folder: Path = dump_folder
        self.trainer: DFIVTrainer = trainer

        self.train_1st_data_t: Optional[TrainDataSetTorch] = None
        self.train_2nd_data_t: Optional[TrainDataSetTorch] = None
        self.test_data_t: Optional[TestDataSetTorch] = None
        self.validation_data_t: Optional[TrainDataSetTorch] = None

    def configure_data(self, train_1st_data_t: TrainDataSetTorch,
                       train_2nd_data_t: TrainDataSetTorch,
                       test_data_t: TestDataSetTorch,
                       validation_data_t: TrainDataSetTorch):

        self.train_1st_data_t = train_1st_data_t
        self.train_2nd_data_t = train_2nd_data_t
        self.test_data_t = test_data_t
        self.validation_data_t = validation_data_t

    def record(self, verbose: int):
        self.trainer.treatment_net.train(False)
        self.trainer.instrumental_net.train(False)
        if self.trainer.covariate_net is not None:
            self.trainer.covariate_net.train(False)

        n_1st_data = self.train_1st_data_t.treatment.size()[0]
        n_2nd_data = self.train_2nd_data_t.treatment.size()[0]
        n_val_data = self.validation_data_t.treatment.size()[0]
        n_test_data = self.test_data_t.treatment.size()[0]
        with torch.no_grad():
            treatment_1st_feature = self.trainer.treatment_net(self.train_1st_data_t.treatment)
            treatment_val_feature = self.trainer.treatment_net(self.validation_data_t.treatment)
            treatment_test_feature = self.trainer.treatment_net(self.test_data_t.treatment)

            instrumental_1st_feature = self.trainer.instrumental_net(self.train_1st_data_t.instrumental)
            instrumental_2nd_feature = self.trainer.instrumental_net(self.train_2nd_data_t.instrumental)
            instrumental_val_feature = self.trainer.instrumental_net(self.validation_data_t.instrumental)

            covariate_2nd_feature = None
            covariate_val_feature = None
            covariate_test_feature = None
            if self.trainer.covariate_net is not None:
                covariate_2nd_feature = self.trainer.covariate_net(self.train_2nd_data_t.covariate)
                covariate_val_feature = self.trainer.covariate_net(self.validation_data_t.covariate)
                covariate_test_feature = self.trainer.covariate_net(self.test_data_t.covariate)

            # stage1
            feature = DFIVModel.augment_stage1_feature(instrumental_1st_feature,
                                                       self.trainer.add_stage1_intercept)
            stage1_weight = fit_linear(treatment_1st_feature, feature, self.trainer.lam1)
            insample_1st_pred = linear_reg_pred(feature, stage1_weight)
            stage1_insample = torch.norm(treatment_1st_feature - insample_1st_pred) ** 2 / n_1st_data

            val_feature = DFIVModel.augment_stage1_feature(instrumental_val_feature,
                                                           self.trainer.add_stage1_intercept)
            outsample_1st_pred = linear_reg_pred(val_feature, stage1_weight)
            stage1_outsample = torch.norm(treatment_val_feature - outsample_1st_pred) ** 2 / n_val_data

            # predicting for stage 2
            feature = DFIVModel.augment_stage1_feature(instrumental_2nd_feature,
                                                       self.trainer.add_stage1_intercept)
            predicted_treatment_feature = linear_reg_pred(feature, stage1_weight)

            # stage2
            feature = DFIVModel.augment_stage2_feature(predicted_treatment_feature,
                                                       covariate_2nd_feature,
                                                       self.trainer.add_stage2_intercept)

            stage2_weight = fit_linear(self.train_2nd_data_t.outcome, feature,
                                       self.trainer.lam2)
            insample_2nd_pred = linear_reg_pred(feature, stage2_weight)
            stage2_insample = torch.norm(self.train_2nd_data_t.outcome - insample_2nd_pred) ** 2 / n_2nd_data

            val_feature = DFIVModel.augment_stage2_feature(outsample_1st_pred,
                                                           covariate_val_feature,
                                                           self.trainer.add_stage2_intercept)
            outsample_2nd_pred = linear_reg_pred(val_feature, stage2_weight)
            stage2_outsample = torch.norm(self.validation_data_t.outcome - outsample_2nd_pred) ** 2 / n_val_data

            # eval for test
            test_feature = DFIVModel.augment_stage2_feature(treatment_test_feature,
                                                            covariate_test_feature,
                                                            self.trainer.add_stage2_intercept)
            test_pred = linear_reg_pred(test_feature, stage2_weight)
            test_loss = torch.norm(self.test_data_t.structural - test_pred) ** 2 / n_test_data

            if verbose >= 1:
                logger.info(f"stage1_insample:{stage1_insample.item()}")
                logger.info(f"stage1_outsample:{stage1_outsample.item()}")
                logger.info(f"stage2_insample:{stage2_insample.item()}")
                logger.info(f"stage2_outsample:{stage2_outsample.item()}")
                logger.info(f"test_loss:{test_loss.item()}")

