from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from torch import nn
import logging
from pathlib import Path
import copy

from sklearn.model_selection import train_test_split
import numpy as np

from src.models.DFIV.nn_structure import build_extractor
from src.data import generate_train_data, generate_test_data
from src.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred
from src.models.DFL.model import DFLModel
from src.models.DFL.monitor import DFLMonitor

logger = logging.getLogger()


class DFLTrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")

        # configure training params
        self.n_epoch: int = train_params["n_epoch"]
        self.weight_decay: float = train_params["weight_decay"]
        self.lam: float = train_params["lam"]
        self.n_iter_treatment = train_params["n_iter_treatment"]
        self.n_iter_covariate = train_params["n_iter_covariate"]
        self.add_intercept: bool = train_params["add_intercept"]

        # build networks
        networks = build_extractor(data_configs["data_name"])
        self.treatment_net: nn.Module = networks[0]
        self.covariate_net: Optional[nn.Module] = networks[2]

        if self.gpu_flg:
            self.treatment_net.to("cuda:0")
            if self.covariate_net is not None:
                self.covariate_net.to("cuda:0")

        self.treatment_opt = torch.optim.Adam(self.treatment_net.parameters(),
                                              weight_decay=self.weight_decay)
        if self.covariate_net:
            self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),
                                                  weight_decay=self.weight_decay)
        self.monitor = None
        if dump_folder is not None:
            self.monitor = DFLMonitor(dump_folder, self)

    def train(self, rand_seed: int = 42, verbose: int = 0) -> float:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        test_data = generate_test_data(**self.data_config)
        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        if self.monitor is not None:
            new_rand_seed = np.random.randint(1e5)
            new_data_config = copy.copy(self.data_config)
            new_data_config["data_size"] = 5000
            validation_data = generate_train_data(rand_seed=new_rand_seed, **self.data_config)
            validation_data_t = TrainDataSetTorch.from_numpy(validation_data)
            if self.gpu_flg:
                validation_data_t = validation_data_t.to_gpu()
            self.monitor.configure_data(train_data_t, test_data_t, validation_data_t)

        self.lam *= train_data_t[0].size()[0]

        for t in range(self.n_epoch):
            self.update_treatment(train_data_t, verbose)
            if self.covariate_net:
                self.update_covariate_net(train_data_t, verbose)

            if verbose >= 1:
                logger.info(f"Epoch {t} ended")
            if self.monitor is not None:
                self.monitor.record(verbose)

        mdl = DFLModel(self.treatment_net, self.covariate_net, self.add_intercept)
        mdl.fit_t(train_data_t, self.lam)
        if self.gpu_flg:
            torch.cuda.empty_cache()

        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        if verbose >= 1:
            logger.info(f"oos_loss:{oos_loss}")
        return oos_loss

    def update_treatment(self, train_data_t, verbose):
        self.treatment_net.train(True)
        if self.covariate_net:
            self.covariate_net.train(False)

        # have covariate features
        covariate_feature = None
        if self.covariate_net:
            covariate_feature = self.covariate_net(train_data_t.covariate).detach()

        for i in range(self.n_iter_treatment):
            self.treatment_opt.zero_grad()
            treatment_feature = self.treatment_net(train_data_t.treatment)
            res = DFLModel.fit_dfl(treatment_feature, covariate_feature, train_data_t.outcome,
                                   self.lam, self.add_intercept)
            loss = res["loss"]
            loss.backward()
            if verbose >= 2:
                logger.info(f"treatment learning: {loss.item()}")
            self.treatment_opt.step()

    def update_covariate_net(self, train_data_t: TrainDataSetTorch, verbose: int):

        self.treatment_net.train(False)
        treatment_feature = self.treatment_net(train_data_t.treatment).detach()
        self.covariate_net.train(True)
        for i in range(self.n_iter_covariate):
            self.covariate_opt.zero_grad()
            covariate_feature = self.covariate_net(train_data_t.covariate)
            res = DFLModel.fit_dfl(treatment_feature, covariate_feature, train_data_t.outcome,
                                   self.lam, self.add_intercept)
            loss = res["loss"]
            loss.backward()
            if verbose >= 2:
                logger.info(f"update covariate: {loss.item()}")
            self.covariate_opt.step()
