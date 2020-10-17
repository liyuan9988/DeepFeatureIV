from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from torch import nn
import logging
from pathlib import Path
import copy

from sklearn.model_selection import train_test_split
import numpy as np

from src.models.DeepGMM.nn_structure import build_extractor
from src.models.DeepGMM.model import DeepGMMModel
from src.data import generate_train_data, generate_test_data
from src.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch, TestDataSet

logger = logging.getLogger()


class DeepGMMTrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.dual_iter: int = train_params["dual_iter"]
        self.primal_iter: int = train_params["primal_iter"]
        self.n_epoch: int = train_params["n_epoch"]

        # build networks
        networks = build_extractor(data_configs["data_name"])
        self.primal_net: nn.Module = networks[0]
        self.dual_net: nn.Module = networks[1]
        self.primal_weight_decay = train_params["primal_weight_decay"]
        self.dual_weight_decay = train_params["dual_weight_decay"]

        if self.gpu_flg:
            self.primal_net.to("cuda:0")
            self.dual_net.to("cuda:0")

        self.primal_opt = torch.optim.Adam(self.primal_net.parameters(),
                                           weight_decay=self.primal_weight_decay,
                                           lr=0.0005, betas=(0.5, 0.9))
        self.dual_opt = torch.optim.Adam(self.dual_net.parameters(),
                                         weight_decay=self.dual_weight_decay,
                                         lr=0.0025, betas=(0.5, 0.9))

        # build monitor
        self.monitor = None

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
        if train_data.covariate is not None:
            train_data = TrainDataSet(treatment=np.concatenate([train_data.treatment, train_data.covariate], axis=1),
                                      structural=train_data.structural,
                                      covariate=None,
                                      instrumental=train_data.instrumental,
                                      outcome=train_data.outcome)
            test_data = TestDataSet(treatment=np.concatenate([test_data.treatment, test_data.covariate], axis=1),
                                     covariate=None,
                                     structural=test_data.structural)

        train_data_t = TrainDataSetTorch.from_numpy(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_data_t = train_data_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        for t in range(self.n_epoch):
            self.dual_update(train_data_t, verbose)
            self.primal_update(train_data_t, verbose)
            if verbose >= 1:
                logger.info(f"Epoch {t} ended")
                mdl = DeepGMMModel(self.primal_net, self.dual_net)
                logger.info(f"test error {mdl.evaluate_t(test_data_t).data.item()}")

        mdl = DeepGMMModel(self.primal_net, self.dual_net)
        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        logger.info(f"target variance: {np.var(train_data.outcome)}")
        return oos_loss

    def dual_update(self, train_data_t: TrainDataSetTorch, verbose: int):
        self.dual_net.train(True)
        self.primal_net.train(False)
        with torch.no_grad():
            epsilon = train_data_t.outcome - self.primal_net(train_data_t.treatment)
        for t in range(self.dual_iter):
            self.dual_opt.zero_grad()
            moment = torch.mean(self.dual_net(train_data_t.instrumental) * epsilon)
            reg = 0.25 * torch.mean((self.dual_net(train_data_t.instrumental) * epsilon) ** 2)
            loss = -moment + reg
            if verbose >= 2:
                logger.info(f"dual loss:{loss.data.item()}")
            loss.backward()
            self.dual_opt.step()

    def primal_update(self, train_data_t: TrainDataSetTorch, verbose: int):
        self.dual_net.train(False)
        self.primal_net.train(True)
        with torch.no_grad():
            dual = self.dual_net(train_data_t.instrumental)
        for t in range(self.primal_iter):
            self.primal_opt.zero_grad()
            epsilon = train_data_t.outcome - self.primal_net(train_data_t.treatment)
            loss = torch.mean(dual * epsilon)
            if verbose >= 2:
                logger.info(f"primal loss:{loss.data.item()}")
            loss.backward()
            self.primal_opt.step()
