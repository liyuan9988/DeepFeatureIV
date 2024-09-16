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
from src.models.DFIV.monitor import DFIVMonitor
from src.models.DFIV.model import DFIVModel
from src.models.DFIV.data_loader import get_minibatch_loader

from src.data import generate_train_data, generate_test_data
from src.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSetTorch
from src.utils.pytorch_linear_reg_utils import linear_reg_loss, fit_linear, linear_reg_pred

logger = logging.getLogger()


class DFIVTrainer(object):

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs
        self.gpu_flg = gpu_flg and torch.cuda.is_available()
        if self.gpu_flg:
            logger.info("gpu mode")
        # configure training params
        self.lam1: float = train_params["lam1"]
        self.lam2: float = train_params["lam2"]
        self.stage1_iter: int = train_params["stage1_iter"]
        self.stage2_iter: int = train_params["stage2_iter"]
        self.covariate_iter: int = train_params["covariate_iter"]
        self.n_epoch: int = train_params["n_epoch"]
        self.split_ratio: float = train_params["split_ratio"]
        self.add_stage1_intercept = True
        self.add_stage2_intercept = True
        self.treatment_weight_decay = train_params["treatment_weight_decay"]
        self.instrumental_weight_decay = train_params["instrumental_weight_decay"]
        self.covariate_weight_decay = train_params["covariate_weight_decay"]
        self.stage1_minibatch = train_params.get("stage1_minibatch", None)
        self.stage2_minibatch = train_params.get("stage2_minibatch", None)

        # build networks
        networks = build_extractor(data_configs["data_name"])
        self.treatment_net: nn.Module = networks[0]
        self.instrumental_net: nn.Module = networks[1]
        self.covariate_net: Optional[nn.Module] = networks[2]
        if self.gpu_flg:
            self.treatment_net.to("cuda:0")
            self.instrumental_net.to("cuda:0")
            if self.covariate_net is not None:
                self.covariate_net.to("cuda:0")
        self.treatment_opt = torch.optim.Adam(self.treatment_net.parameters(),
                                              weight_decay=self.treatment_weight_decay)
        self.instrumental_opt = torch.optim.Adam(self.instrumental_net.parameters(),
                                                 weight_decay=self.instrumental_weight_decay)
        if self.covariate_net:
            self.covariate_opt = torch.optim.Adam(self.covariate_net.parameters(),
                                                  weight_decay=self.covariate_weight_decay)

        # build monitor
        self.monitor = None
        if dump_folder is not None:
            self.monitor = DFIVMonitor(dump_folder, self)

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
        train_1st_t, train_2nd_t = self.split_train_data(train_data)
        test_data_t = TestDataSetTorch.from_numpy(test_data)
        if self.gpu_flg:
            train_1st_t = train_1st_t.to_gpu()
            train_2nd_t = train_2nd_t.to_gpu()
            test_data_t = test_data_t.to_gpu()

        
        if self.monitor is not None:
            new_rand_seed = np.random.randint(1e5)
            new_data_config = copy.copy(self.data_config)
            new_data_config["data_size"] = 5000
            validation_data = generate_train_data(rand_seed=new_rand_seed, **self.data_config)
            validation_data_t = TrainDataSetTorch.from_numpy(validation_data)
            if self.gpu_flg:
                validation_data_t = validation_data_t.to_gpu()
            self.monitor.configure_data(train_1st_t, train_2nd_t, test_data_t, validation_data_t)

        self.lam1 *= train_1st_t[0].size()[0]
        self.lam2 *= train_2nd_t[0].size()[0]

        if self.stage1_minibatch is None:
            self.stage1_minibatch = train_1st_t.treatment.shape[0]

        if self.stage2_minibatch is None:
            self.stage2_minibatch = train_2nd_t.treatment.shape[0]
            
        for t in range(self.n_epoch):
            stage1_loader = get_minibatch_loader(train_1st_t, self.stage1_minibatch)
            stage2_loader = get_minibatch_loader(train_2nd_t, self.stage2_minibatch)
            for train_1st_t_sub, train_2nd_t_sub in zip(stage1_loader, stage2_loader):
                self.stage1_update(train_1st_t_sub, verbose)
                if self.covariate_net:
                    self.update_covariate_net(train_1st_t_sub, train_2nd_t_sub, verbose)
                self.stage2_update(train_1st_t_sub, train_2nd_t_sub, verbose)
            if self.monitor is not None:
                self.monitor.record(verbose)
            if verbose >= 1:
                logger.info(f"Epoch {t} ended")

        mdl = DFIVModel(self.treatment_net, self.instrumental_net, self.covariate_net,
                        self.add_stage1_intercept, self.add_stage2_intercept)
        mdl.fit_t(train_1st_t, train_2nd_t, self.lam1, self.lam2)
        if self.gpu_flg:
            torch.cuda.empty_cache()

        oos_loss: float = mdl.evaluate_t(test_data_t).data.item()
        return oos_loss

    def split_train_data(self, train_data: TrainDataSet):
        n_data = train_data[0].shape[0]
        idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=self.split_ratio)

        def get_data(data, idx):
            return data[idx] if data is not None else None

        train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
        train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
        train_1st_data_t = TrainDataSetTorch.from_numpy(train_1st_data)
        train_2nd_data_t = TrainDataSetTorch.from_numpy(train_2nd_data)

        return train_1st_data_t, train_2nd_data_t

    def stage1_update(self, train_1st_t: TrainDataSetTorch, verbose: int):
        self.treatment_net.train(False)
        self.instrumental_net.train(True)
        if self.covariate_net:
            self.covariate_net.train(False)
        
        treatment_feature = self.treatment_net(train_1st_t.treatment).detach()
        for i in range(self.stage1_iter):
            self.instrumental_opt.zero_grad()
            instrumental_feature = self.instrumental_net(train_1st_t.instrumental)
            feature = DFIVModel.augment_stage1_feature(instrumental_feature,
                                                       self.add_stage1_intercept)
            loss = linear_reg_loss(treatment_feature, feature, self.lam1)
            loss.backward()
            if verbose >= 2:
                logger.info(f"stage1 learning: {loss.item()}")
            self.instrumental_opt.step()

    def stage2_update(self, train_1st_t, train_2nd_t, verbose):
        self.treatment_net.train(True)
        self.instrumental_net.train(False)
        if self.covariate_net:
            self.covariate_net.train(False)

        # have instrumental features
        instrumental_1st_feature = self.instrumental_net(train_1st_t.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_t.instrumental).detach()

        # have covariate features
        covariate_2nd_feature = None
        if self.covariate_net:
            covariate_2nd_feature = self.covariate_net(train_2nd_t.covariate).detach()

        for i in range(self.stage2_iter):
            self.treatment_opt.zero_grad()
            treatment_1st_feature = self.treatment_net(train_1st_t.treatment)
            res = DFIVModel.fit_2sls(treatment_1st_feature,
                                     instrumental_1st_feature,
                                     instrumental_2nd_feature,
                                     covariate_2nd_feature,
                                     train_2nd_t.outcome,
                                     self.lam1, self.lam2,
                                     self.add_stage1_intercept,
                                     self.add_stage2_intercept)
            loss = res["stage2_loss"]
            loss.backward()
            if verbose >= 2:
                logger.info(f"stage2 learning: {loss.item()}")
            self.treatment_opt.step()

    def update_covariate_net(self, train_1st_data: TrainDataSetTorch, train_2nd_data: TrainDataSetTorch,
                             verbose: int):
        # have instrumental features
        self.instrumental_net.train(False)
        instrumental_1st_feature = self.instrumental_net(train_1st_data.instrumental).detach()
        instrumental_2nd_feature = self.instrumental_net(train_2nd_data.instrumental).detach()

        self.treatment_net.train(False)
        treatment_1st_feature = self.treatment_net(train_1st_data.treatment).detach()

        feature = DFIVModel.augment_stage1_feature(instrumental_1st_feature, self.add_stage1_intercept)
        stage1_weight = fit_linear(treatment_1st_feature, feature, self.lam1)

        # predicting for stage 2
        feature = DFIVModel.augment_stage1_feature(instrumental_2nd_feature,
                                                   self.add_stage1_intercept)
        predicted_treatment_feature = linear_reg_pred(feature, stage1_weight).detach()

        self.covariate_net.train(True)
        for i in range(self.covariate_iter):
            self.covariate_opt.zero_grad()
            covariate_feature = self.covariate_net(train_2nd_data.covariate)
            feature = DFIVModel.augment_stage2_feature(predicted_treatment_feature,
                                                       covariate_feature,
                                                       self.add_stage2_intercept)

            loss = linear_reg_loss(train_2nd_data.outcome, feature, self.lam2)
            loss.backward()
            if verbose >= 2:
                logger.info(f"update covariate: {loss.item()}")
            self.covariate_opt.step()
