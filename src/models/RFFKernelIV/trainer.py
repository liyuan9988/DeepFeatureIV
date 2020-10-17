from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler, Nystroem

from src.data import generate_train_data, generate_test_data
from src.data.data_class import TrainDataSet, TrainDataSetTorch
from src.models.RFFKernelIV.model import RFFKernelIVModel

logger = logging.getLogger()


def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


class RFFKernelIVTrainer:

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs

        self.lambda1 = train_params["lam1"]
        self.lambda2 = train_params["lam2"]
        self.n_feature = train_params["n_feature"]
        self.kernel_approx_name = train_params["kernel_approx"]
        self.split_ratio = train_params["split_ratio"]

    def split_train_data(self, train_data: TrainDataSet):
        n_data = train_data[0].shape[0]
        idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=self.split_ratio)

        def get_data(data, idx):
            return data[idx] if data is not None else None

        train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
        train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
        return train_1st_data, train_2nd_data

    @staticmethod
    def get_kernel_approx(approx_name: str, sigma: float, n_feature: int, data: np.ndarray):
        if approx_name == "RFF":
            mdl = RBFSampler(gamma=sigma, n_components=n_feature).fit(data)
        elif approx_name == "Nystrom":
            mdl = Nystroem(gamma=sigma, n_components=n_feature).fit(data)
        else:
            raise ValueError(f"{approx_name} does not exists")
        return mdl

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

        treatment_sigma = get_median(train_1st_data.treatment)
        treatment_feature_ext = RFFKernelIVTrainer.get_kernel_approx(self.kernel_approx_name,
                                                                     1.0 / treatment_sigma,
                                                                     self.n_feature, train_1st_data.treatment)

        whole_instrumental = train_data.instrumental
        instrument_sigma = get_median(whole_instrumental)
        instrument_feature_ext = RFFKernelIVTrainer.get_kernel_approx(self.kernel_approx_name,
                                                                      1.0 / instrument_sigma,
                                                                      self.n_feature, whole_instrumental)
        covariate_feature_ext = None
        if train_data.covariate is not None:
            covariate_sigma = get_median(train_2nd_data.covariate)
            covariate_feature_ext = RFFKernelIVTrainer.get_kernel_approx(self.kernel_approx_name,
                                                                         1.0 / covariate_sigma,
                                                                         self.n_feature, train_2nd_data.covariate)

        lam1_arr = 10 ** np.linspace(self.lambda1[0], self.lambda1[1], 50)
        lam2_arr = 10 ** np.linspace(self.lambda2[0], self.lambda2[1], 50)
        best_lam1, stage1_weight = self.tune_lambda1(treatment_feature_ext, instrument_feature_ext, train_1st_data,
                                                     train_2nd_data, lam1_arr)
        best_lam2 = self.tune_lambda2(treatment_feature_ext, instrument_feature_ext, covariate_feature_ext,
                                      train_1st_data, train_2nd_data, stage1_weight, lam2_arr)

        mdl = RFFKernelIVModel(treatment_feature_ext, instrument_feature_ext, covariate_feature_ext)
        mdl.fit(train_1st_data, train_2nd_data, best_lam1, best_lam2)
        return mdl.evaluate(test_data)

    def tune_lambda1(self, treatment_feature_ext, instrumental_feature_ext,
                     train_1st_data, train_2nd_data, lam1_array):

        treatment_1st_feature = treatment_feature_ext.transform(train_1st_data.treatment)
        treatment_2nd_feature = treatment_feature_ext.transform(train_2nd_data.treatment)

        instrumental_1st_feature = instrumental_feature_ext.transform(train_1st_data.instrumental)
        instrumental_2nd_feature = instrumental_feature_ext.transform(train_2nd_data.instrumental)
        stage1_weights = [RFFKernelIVModel.stage1_learning(treatment_1st_feature, instrumental_1st_feature,
                                                           lam1) for lam1 in lam1_array]
        scores = [np.linalg.norm(treatment_2nd_feature - instrumental_2nd_feature.dot(weight))
                  for weight in stage1_weights]
        return lam1_array[np.argmin(scores)], stage1_weights[np.argmin(scores)]

    def tune_lambda2(self, treatment_feature_ext, instrumental_feature_ext, covariate_feature_ext,
                     train_1st_data, train_2nd_data, stage1_weight, lam2_array):

        treatment_1st_feature = treatment_feature_ext.transform(train_1st_data.treatment)
        instrumental_2nd_feature = instrumental_feature_ext.transform(train_2nd_data.instrumental)
        covariate_1st_feature = None
        covariate_2nd_feature = None
        if covariate_feature_ext is not None:
            covariate_1st_feature = covariate_feature_ext.transform(train_1st_data.covariate)
            covariate_2nd_feature = covariate_feature_ext.transform(train_2nd_data.covariate)

        stage2_weights = [RFFKernelIVModel.stage2_learning(instrumental_2nd_feature, covariate_2nd_feature,
                                                           train_2nd_data.outcome, stage1_weight, lam2)
                          for lam2 in lam2_array]

        feature = treatment_1st_feature
        if covariate_1st_feature is not None:
            feature = RFFKernelIVModel.outer_prod(feature, covariate_1st_feature)
        scores = [np.linalg.norm(train_1st_data.outcome - feature.dot(weight))
                  for weight in stage2_weights]

        return lam2_array[np.argmin(scores)]
