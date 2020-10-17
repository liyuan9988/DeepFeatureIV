from itertools import product
import numpy as np
from numpy.random import default_rng
import logging
from typing import Tuple, TypeVar

from ..data.data_class import TrainDataSet, TestDataSet

np.random.seed(42)
logger = logging.getLogger()


def generate_test_sin() -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    data_size = 100
    rng = default_rng(seed=100)
    Z = rng.uniform([-3, -3], [3, 3], (data_size, 2))
    e = rng.normal(size=(data_size,))
    X = Z[:, 0] + e + rng.normal(scale=0.1, size=(data_size,))

    Y = np.sin(X) / np.sqrt(1.5)
    test_data = TestDataSet(treatment=X[:, np.newaxis],
                            covariate=None,
                            structural=Y[:, np.newaxis])
    return test_data


def generate_train_sin(data_size: int,
                       rand_seed: int = 42) -> TrainDataSet:
    """

    Parameters
    ----------
    data_size : int
        size of data
    rho : float
        parameter for noise correlation
    rand_seed : int
        random seed


    Returns
    -------
    train_data : TrainDataSet
    """

    rng = default_rng(seed=rand_seed)
    Z = rng.uniform([-3, -3], [3, 3], (data_size, 2))
    e = rng.normal(size=(data_size,))
    X = Z[:, 0] + e + rng.normal(scale=0.1, size=(data_size,))
    Y = np.sin(X) + e + rng.normal(scale=0.1, size=(data_size,))
    Y = Y / np.sqrt(1.5)
    structural = np.sin(X) / np.sqrt(1.5)
    train_data = TrainDataSet(treatment=X[:, np.newaxis],
                              instrumental=Z,
                              covariate=None,
                              outcome=Y[:, np.newaxis],
                              structural=structural[:, np.newaxis])

    return train_data
