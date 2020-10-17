from typing import Tuple, Sequence
import numpy as np
from numpy.random import default_rng
from torchvision.datasets import MNIST
from torchvision import transforms
from filelock import FileLock

from ..data.demand_design import generate_test_demand_design, generate_train_demand_design
from ..data.data_class import TrainDataSet, TestDataSet


def attach_image(num_array: Sequence[int], train_flg: bool, seed: int = 42) -> np.ndarray:
    """
    Randomly samples number from MNIST datasets

    Parameters
    ----------
    num_array : Array[int]
        Array of numbers that we sample for
    train_flg : bool
        Whether to sample from train/test in MNIST dataset
    seed : int
        Random seed
    Returns
    -------
    result : (len(num_array), 28*28) array
        ndarray for sampled images
    """
    rng = default_rng(seed)
    with FileLock("./data.lock"):
        mnist = MNIST("./data/", train=train_flg, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), target_transform=None, download=True)
    img_data = mnist.data.numpy()
    img_target = mnist.targets.numpy()

    def select_one(idx):
        sub = img_data[img_target == idx]
        return sub[rng.choice(sub.shape[0])].reshape((1, -1))

    return np.concatenate([select_one(num) for num in num_array], axis=0)


def generate_test_demand_design_image() -> TestDataSet:
    """
    Returns
    -------
    test_data : TestDataSet
        2800 points of test data, uniformly sampled from (price, time, emotion). Emotion is transformed into img.
    """
    org_test: TestDataSet = generate_test_demand_design(False)
    treatment = org_test.treatment
    covariate = org_test.covariate
    target = org_test.structural
    emotion_arr = covariate[:, 1].astype(int)
    emotion_img = attach_image(emotion_arr, False, 42)
    covariate_img = np.concatenate([covariate[:, 0:1], emotion_img], axis=1)
    return TestDataSet(treatment=treatment,
                       covariate=covariate_img,
                       structural=target)


def generate_train_demand_design_image(data_size: int,
                                       rho: float,
                                       rand_seed: int = 42):
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
    org_train: TrainDataSet = generate_train_demand_design(data_size, rho, rand_seed, False)
    covariate = org_train.covariate
    emotion_arr = covariate[:, -1]
    emotion_img = attach_image(emotion_arr, True, rand_seed)
    covariate_img = np.concatenate([covariate[:, 0:1], emotion_img], axis=1)
    instrument_img = np.concatenate([org_train.instrumental[:, 0:2], emotion_img], axis=1)
    return TrainDataSet(treatment=org_train.treatment,
                        instrumental= instrument_img,
                        covariate=covariate_img,
                        outcome=org_train.outcome,
                        structural=org_train.structural)
