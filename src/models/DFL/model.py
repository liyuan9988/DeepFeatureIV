from typing import List, Optional
import torch
from torch import nn
import numpy as np
import logging

from src.utils.pytorch_linear_reg_utils import fit_linear, linear_reg_pred, outer_prod, add_const_col
from src.data.data_class import TrainDataSet, TestDataSet, TrainDataSetTorch, TestDataSetTorch

logger = logging.getLogger()


class DFLModel:
    weight_mat: torch.Tensor

    def __init__(self,
                 treatment_net: nn.Module,
                 covariate_net: Optional[nn.Module],
                 add_intercept: bool
                 ):
        self.treatment_net = treatment_net
        self.covariate_net = covariate_net
        self.add_intercept = add_intercept

    @staticmethod
    def augment_feature(treatment_feature: torch.Tensor,
                        covariate_feature: Optional[torch.Tensor],
                        add_intercept: bool):
        feature = treatment_feature
        if add_intercept:
            feature = add_const_col(feature)

        if covariate_feature is not None:
            feature_tmp = covariate_feature
            if add_intercept:
                feature_tmp = add_const_col(feature_tmp)
            feature = outer_prod(feature, feature_tmp)
            feature = torch.flatten(feature, start_dim=1)

        return feature

    @staticmethod
    def fit_dfl(treatment_feature: torch.Tensor,
                covariate_feature: Optional[torch.Tensor],
                outcome_t: torch.Tensor,
                lam: float, add_intercept: bool
                ):

        # stage1
        feature = DFLModel.augment_feature(treatment_feature,
                                           covariate_feature,
                                           add_intercept)

        weight = fit_linear(outcome_t, feature, lam)
        pred = linear_reg_pred(feature, weight)
        loss = torch.norm((outcome_t - pred)) ** 2 + lam * torch.norm(weight) ** 2

        return dict(weight=weight, loss=loss)

    def fit_t(self, train_data_t: TrainDataSetTorch, lam: float):
        treatment_feature = self.treatment_net(train_data_t.treatment)
        outcome_t = train_data_t.outcome
        covariate_feature = None
        if self.covariate_net is not None:
            covariate_feature = self.covariate_net(train_data_t.covariate)

        res = DFLModel.fit_dfl(treatment_feature, covariate_feature, outcome_t, lam, self.add_intercept)
        self.weight_mat = res["weight"]

    def fit(self, train_data: TrainDataSet, lam: float):
        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        self.fit_t(train_data_t, lam)

    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor]):
        treatment_feature = self.treatment_net(treatment)
        covariate_feature = None
        if self.covariate_net:
            covariate_feature = self.covariate_net(covariate)

        feature = DFLModel.augment_feature(treatment_feature, covariate_feature, self.add_intercept)
        return linear_reg_pred(feature, self.weight_mat)

    def predict(self, treatment: np.ndarray, covariate: Optional[np.ndarray]):
        treatment_t = torch.tensor(treatment, dtype=torch.float32)
        covariate_t = None
        if covariate is not None:
            covariate_t = torch.tensor(covariate, dtype=torch.float32)
        return self.predict_t(treatment_t, covariate_t).data.numpy()

    def evaluate_t(self, test_data: TestDataSetTorch):
        target = test_data.structural
        with torch.no_grad():
            pred = self.predict_t(test_data.treatment, test_data.covariate)
        return (torch.norm((target - pred)) ** 2) / target.size()[0]

    def evaluate(self, test_data: TestDataSet):
        return self.evaluate_t(TestDataSetTorch.from_numpy(test_data)).data.item()
