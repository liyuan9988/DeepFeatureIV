from typing import Dict, Any, Optional
from pathlib import Path
import logging

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal, OneHotCategorical
from torch.autograd import detect_anomaly
import numpy as np


from src.models.DeepIV.nn_structure import build_extractor
from src.data import generate_train_data, generate_test_data
from src.data import preprocess
from src.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch
from src.models.DeepIV.model import DeepIVModel

logger = logging.getLogger()


class DeepIVTrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.n_epochs = train_params["n_epoch"]
        self.batch_size = train_params["batch_size"]
        self.n_sample = train_params["n_sample"]

        dropout_rate = min(1000. / (1000. + data_configs["data_size"]), 0.5)
        args = dict(dropout_rate=dropout_rate)
        networks = build_extractor(data_configs["data_name"], **args)
        self.instrumental_net = networks[0]
        self.response_net = networks[1]
        if self.gpu_flg:
            self.instrumental_net.to("cuda:0")
            self.response_net.to("cuda:0")

        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(),
                                                 weight_decay=0.0001)
        self.response_opt = torch.optim.Adam(self.response_net.parameters(),
                                             weight_decay=0.001)

    def train(self, rand_seed: int = 42, verbose: int = 0) -> float:
        train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        test_data = generate_test_data(**self.data_config)
        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        try:
            self.update_stage1(train_data_t, verbose)
            self.update_stage2(train_data_t, verbose)
        except RuntimeError:
            return np.nan

        self.response_net.train(False)
        self.instrumental_net.train(False)
        mdl = DeepIVModel(self.response_net, self.instrumental_net, self.data_config["data_name"])
        if self.gpu_flg:
            torch.cuda.empty_cache()
        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        return oos_loss

    @staticmethod
    def density_est_loss(cat: OneHotCategorical, norm: Normal, treatment: torch.Tensor) -> torch.Tensor:
        assert treatment.size() == norm.mean.size()[:2]
        loglik = norm.log_prob(treatment.unsqueeze(2).expand_as(norm.mean))
        loglik = torch.sum(loglik, dim=1)
        loglik = torch.clamp(loglik, min=-40)
        loss = -torch.logsumexp(cat.logits + loglik, dim=1)
        return torch.sum(loss)

    def update_stage1(self, train_data: TrainDataSetTorch, verbose: int):
        data_set = TensorDataset(train_data.instrumental, train_data.treatment)
        loss_val = None
        for t in range(self.n_epochs):
            data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)

            for instrumental, treatment in data_loader:
                self.instrumental_opt.zero_grad()
                norm, cat = self.instrumental_net(instrumental)
                treatment = preprocess.rescale_treatment(treatment, self.data_config["data_name"])
                loss_val = self.density_est_loss(cat, norm, treatment)
                try:
                    with detect_anomaly():
                        loss_val.backward()
                        self.instrumental_opt.step()
                except RuntimeError:
                    logger.info("NaN detected, skipping batch")
            if verbose >= 2 and loss_val is not None:
                logger.info(f"stage1 learning: {loss_val.item()}")

    def update_stage2(self, train_data: TrainDataSetTorch, verbose: int):
        data_set = TensorDataset(train_data.instrumental, train_data.outcome)
        loss = nn.MSELoss()
        if train_data.covariate is not None:
            data_set = TensorDataset(train_data.instrumental,
                                     train_data.covariate,
                                     train_data.outcome)
        loss_val = None
        self.instrumental_net.train(False)
        for t in range(self.n_epochs):
            data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
            for data in data_loader:
                self.response_opt.zero_grad()
                instrumental = data[0]
                outcome = data[-1]
                covariate = None
                if train_data.covariate is not None:
                    covariate = data[1]
                outcome = preprocess.rescale_outcome(outcome, self.data_config["data_name"])
                outcome = outcome.repeat((self.n_sample, 1))
                norm, cat = self.instrumental_net(instrumental)
                if torch.sum(torch.isnan(cat.probs)):
                    raise RuntimeError("NaN prob detected")
                pred = DeepIVModel.sample_from_density(self.n_sample, self.response_net,
                                                       norm, cat, covariate)
                loss_val = loss(pred, outcome)
                loss_val.backward()
                self.response_opt.step()

            if verbose >= 2 and loss_val is not None:
                logger.info(f"stage2 learning: {loss_val.item()}")
