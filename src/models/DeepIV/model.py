from typing import Optional

import torch
from torch import nn
from torch.distributions import Normal, OneHotCategorical
import numpy as np

from src.data.data_class import TestDataSetTorch, TestDataSet
from src.data import preprocess


class DeepIVModel(object):

    def __init__(self, response_net: nn.Module, instrumental_net: nn.Module, data_name: str):
        self.instrumental_net = instrumental_net
        self.response_net = response_net
        self.data_name = data_name

    @classmethod
    def sample_from_density(cls, n_sample: int, response_net: nn.Module, norm: Normal,
                            cat: OneHotCategorical, covariate: Optional[torch.Tensor]):
        pred_list = []
        for i in range(n_sample):
            cat_sample = cat.sample().unsqueeze(1)  # size = [n_data, 1, n_component]
            norm_sample = norm.sample()  # size = [n_data, output_dim, n_component]
            sample = torch.sum(cat_sample * norm_sample, dim=2)  # size = [n_data, output_dim]
            if covariate is not None:
                pred = response_net(sample, covariate)
            else:
                pred = response_net(sample)
            pred_list.append(pred)
        return torch.cat(pred_list, dim=0)

    def predict_t(self, treatment: torch.Tensor, covariate: Optional[torch.Tensor]):
        treatment = preprocess.rescale_treatment(treatment, self.data_name)
        if covariate is None:
            return self.response_net(treatment)
        else:
            return self.response_net(treatment, covariate)

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
        pred = preprocess.inv_rescale_outcome(pred, self.data_name)
        return (torch.norm((target - pred)) ** 2) / target.size()[0]

    def evaluate(self, test_data: TestDataSet):
        return self.evaluate_t(TestDataSetTorch.from_numpy(test_data)).data.item()
