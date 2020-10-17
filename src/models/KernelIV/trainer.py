from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


from src.data import generate_train_data, generate_test_data
from src.data.data_class import TrainDataSet, TrainDataSetTorch
from src.models.KernelIV.model import KernelIVModel

logger = logging.getLogger()


def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


class KernelIVTrainer:

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs

        self.lambda1 = train_params["lam1"]
        self.lambda2 = train_params["lam2"]
        self.split_ratio = train_params["split_ratio"]

    def split_train_data(self, train_data: TrainDataSet):
        n_data = train_data[0].shape[0]
        idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=self.split_ratio)

        def get_data(data, idx):
            return data[idx] if data is not None else None

        train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
        train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
        return train_1st_data, train_2nd_data

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
        train_1st_data, train_2nd_data = self.split_train_data(train_data)

        # get stage1 data
        X1 = train_1st_data.treatment
        if train_1st_data.covariate is not None:
            X1 = np.concatenate([X1, train_1st_data.covariate], axis=-1)
        Z1 = train_1st_data.instrumental
        Y1 = train_1st_data.outcome
        N = X1.shape[0]

        # get stage2 data
        X2 = train_2nd_data.treatment
        if train_2nd_data.covariate is not None:
            X2 = np.concatenate([X2, train_2nd_data.covariate], axis=-1)
        Z2 = train_2nd_data.instrumental
        Y2 = train_2nd_data.outcome
        M = X2.shape[0]

        if verbose > 0:
            logger.info("start stage1")

        sigmaX = get_median(X1)
        sigmaZ = get_median(Z1)
        KX1X1 = KernelIVModel.cal_gauss(X1, X1, sigmaX)
        KZ1Z1 = KernelIVModel.cal_gauss(Z1, Z1, sigmaZ)
        KZ1Z2 = KernelIVModel.cal_gauss(Z1, Z2, sigmaZ)
        KX1X2 = KernelIVModel.cal_gauss(X1, X2, sigmaX)

        if isinstance(self.lambda1, list):
            self.lambda1 = 10 ** np.linspace(self.lambda1[0], self.lambda1[1], 50)
            gamma = self.stage1_tuning(KX1X1, KX1X2, KZ1Z1, KZ1Z2)
        else:
            gamma = np.linalg.solve(KZ1Z1 + N * self.lambda1 * np.eye(N), KZ1Z2)
        W = KX1X1.dot(gamma)
        if verbose > 0:
            logger.info("end stage1")
            logger.info("start stage2")

        if isinstance(self.lambda2, list):
            self.lambda2 = 10 ** np.linspace(self.lambda2[0], self.lambda2[1], 50)
            alpha = self.stage2_tuning(W, KX1X1, Y1, Y2)
        else:
            alpha = np.linalg.solve(W.dot(W.T) + M * self.lambda2 * KX1X1, W.dot(Y2))

        if verbose > 0:
            logger.info("end stage2")

        mdl = KernelIVModel(X1, alpha, sigmaX)
        return mdl.evaluate(test_data)

    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KZ1Z2):
        N = KX1X1.shape[0]
        gamma_list = [np.linalg.solve(KZ1Z1 + N * lam1 * np.eye(N), KZ1Z2) for lam1 in self.lambda1]
        score = [np.trace(gamma.T.dot(KX1X1.dot(gamma)) - 2 * KX1X2.T.dot(gamma)) for gamma in gamma_list]
        self.lambda1 = self.lambda1[np.argmin(score)]
        return gamma_list[np.argmin(score)]

    def stage2_tuning(self, W, KX1X1, Y1, Y2):
        M = W.shape[1]
        b = W.dot(Y2)
        A = W.dot(W.T)
        alpha_list = [np.linalg.solve(A + M * lam2 * KX1X1, b) for lam2 in self.lambda2]
        score = [np.linalg.norm(Y1 - KX1X1.dot(alpha)) for alpha in alpha_list]
        self.lambda2 = self.lambda2[np.argmin(score)]
        return alpha_list[np.argmin(score)]
