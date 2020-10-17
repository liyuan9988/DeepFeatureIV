from typing import Optional, NewType, Union
import numpy as np
from sklearn.kernel_approximation import RBFSampler, Nystroem

from src.data.data_class import TrainDataSet, TestDataSet

Approx = NewType('Approx', Union[RBFSampler, Nystroem])


class RFFKernelIVModel:
    stage1_wight: np.ndarray
    stage2_wight: np.ndarray

    def __init__(self, treatment_feature_ext: Approx,
                 instrumental_feature_ext: Approx, covariate_feature_ext: Optional[Approx]):
        """

        Parameters
        ----------
        treatment_feature_ext:  Union[RBFSampler, Nystroem]
            kernel approximation for treatment variable
        instrumental_feature_ext:   Union[RBFSampler, Nystroem]
            kernel approximation for instrumental variable
        covariate_feature_ext:  Optional[Union[RBFSampler, Nystroem]]
            kernel approximation for covariate variable
        """
        self.treatment_feature_ext = treatment_feature_ext
        self.instrumental_feature_ext = instrumental_feature_ext
        self.covariate_feature_ext = covariate_feature_ext

    @staticmethod
    def stage1_learning(treatment_1st_feature: np.ndarray, instrument_1st_feature: np.ndarray, lam1: float):
        n_data, n_dim = instrument_1st_feature.shape
        A = instrument_1st_feature.T.dot(instrument_1st_feature)
        A += lam1 * np.eye(n_dim) * n_data
        b = instrument_1st_feature.T.dot(treatment_1st_feature)
        w = np.linalg.solve(A, b)
        return w

    @staticmethod
    def stage2_learning(instrument_2nd_feature: np.ndarray,
                        covariate_2nd_feature: Optional[np.ndarray],
                        outcome_2nd: np.ndarray,
                        stage1_weight: np.ndarray,
                        lam2: float):

        stage2_feature = instrument_2nd_feature.dot(stage1_weight)
        if covariate_2nd_feature is not None:
            stage2_feature = RFFKernelIVModel.outer_prod(stage2_feature, covariate_2nd_feature)

        n_data, n_dim = stage2_feature.shape
        A = stage2_feature.T.dot(stage2_feature)
        A += lam2 * np.eye(n_dim) * n_data
        b = stage2_feature.T.dot(outcome_2nd)
        w = np.linalg.solve(A, b)
        return w

    @staticmethod
    def outer_prod(mat1, mat2):
        N = mat1.shape[0]
        res = np.array([np.outer(mat1[i], mat2[i]).ravel() for i in range(N)])
        return res

    def fit(self, train_1st_data: TrainDataSet, train_2nd_data: TrainDataSet, lam1: float, lam2: float):
        treatment_1st_feature = self.treatment_feature_ext.transform(train_1st_data.treatment)
        instrumental_1st_feature = self.instrumental_feature_ext.transform(train_1st_data.instrumental)
        self.stage1_wight = self.stage1_learning(treatment_1st_feature, instrumental_1st_feature, lam1)
        instrumental_2nd_feature = self.instrumental_feature_ext.transform(train_2nd_data.instrumental)
        covariate_2nd_feature = None
        if train_2nd_data.covariate is not None:
            covariate_2nd_feature = self.covariate_feature_ext.transform(train_2nd_data.covariate)
        self.stage2_wight = self.stage2_learning(instrumental_2nd_feature, covariate_2nd_feature,
                                                 train_2nd_data.outcome, self.stage1_wight, lam2)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        stage2_feature = self.treatment_feature_ext.transform(treatment)
        if covariate is not None:
            covariate_feature = self.covariate_feature_ext.transform(covariate)
            stage2_feature = self.outer_prod(stage2_feature, covariate_feature)

        return np.dot(stage2_feature, self.stage2_wight)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.treatment, test_data.covariate)
        return np.mean((test_data.structural - pred) ** 2)
