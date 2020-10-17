from itertools import product
import numpy as np
from numpy.random import default_rng
import logging
from typing import Tuple, TypeVar

from ..data.data_class import TrainDataSet, TestDataSet

np.random.seed(42)
logger = logging.getLogger()


def psi(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)


def f(p: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 100 + (10 + p) * s * psi(t) - 2 * p


def generate_test_demand_design(old_flg: bool = False) -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        Uniformly sampled from (p,t,s).
    """
    price = np.linspace(10, 25, 20)
    time = np.linspace(0.0, 10, 20)
    emotion = np.array([1, 2, 3, 4, 5, 6, 7])
    data = []
    target = []
    for p, t, s in product(price, time, emotion):
        data.append([p, t, s])
        target.append(f(p, t, s))
    features = np.array(data)
    targets: np.ndarray = np.array(target)[:, np.newaxis]
    if old_flg:
        test_data = TestDataSet(treatment=features,
                                structural=targets,
                                covariate=None)
    else:
        test_data = TestDataSet(treatment=features[:, 0:1],
                                covariate=features[:, 1:],
                                structural=targets)
    return test_data


def generate_train_demand_design(data_size: int,
                                 rho: float,
                                 rand_seed: int = 42,
                                 old_flg: bool = False) -> TrainDataSet:
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
    emotion = rng.choice(list(range(1, 8)), data_size)
    time = rng.uniform(0, 10, data_size)
    cost = rng.normal(0, 1.0, data_size)
    noise_price = rng.normal(0, 1.0, data_size)
    noise_demand = rho * noise_price + rng.normal(0, np.sqrt(1 - rho ** 2), data_size)
    price = 25 + (cost + 3) * psi(time) + noise_price
    structural: np.ndarray = f(price, time, emotion).astype(float)
    outcome: np.ndarray = (structural + noise_demand).astype(float)
    if old_flg:
        treatment = np.c_[price, time, emotion]
        instrumental = np.c_[cost, time, emotion]
        train_data = TrainDataSet(treatment=treatment,
                                  instrumental=instrumental,
                                  covariate=None,
                                  outcome=outcome[:, np.newaxis],
                                  structural=structural[:, np.newaxis])
    else:
        treatment: np.ndarray = price[:, np.newaxis]
        covariate: np.ndarray = np.c_[time, emotion]
        instrumental: np.ndarray = np.c_[cost, time, emotion]
        train_data = TrainDataSet(treatment=treatment,
                                  instrumental=instrumental,
                                  covariate=covariate,
                                  outcome=outcome[:, np.newaxis],
                                  structural=structural[:, np.newaxis])
    return train_data
