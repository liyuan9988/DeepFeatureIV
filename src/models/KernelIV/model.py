from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from src.data.data_class import TrainDataSet, TestDataSet


class KernelIVModel:

    def __init__(self, X_train: np.ndarray, alpha: np.ndarray, sigma: float):
        """

        Parameters
        ----------
        X_train: np.ndarray[n_stage1, dim_treatment]
            data for treatment
        alpha:  np.ndarray[n_stage1*n_stage2 ,dim_outcome]
            final weight for prediction
        sigma: gauss parameter
        """
        self.X_train = X_train
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def cal_gauss(XA, XB, sigma: float = 1):
        """
        Returns gaussian kernel matrix
        Parameters
        ----------
        XA : np.ndarray[n_data1, n_dim]
        XB : np.ndarray[n_data2, n_dim]
        sigma : float

        Returns
        -------
        mat: np.ndarray[n_data1, n_data2]
        """
        dist_mat = cdist(XA, XB, "sqeuclidean")
        return np.exp(-dist_mat / sigma)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        X = np.array(treatment, copy=True)
        if covariate is not None:
            X = np.concatenate([X, covariate], axis=1)
        Kx = self.cal_gauss(X, self.X_train, self.sigma)
        return np.dot(Kx, self.alpha)

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.treatment, test_data.covariate)
        return np.mean((test_data.structural - pred)**2)

