from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import ray
import logging
import torch

from src.utils import grid_search_dict
from src.models.DFIV.trainer import DFIVTrainer
from src.models.DeepIV.trainer import DeepIVTrainer
from src.models.KernelIV.trainer import KernelIVTrainer
from src.models.RFFKernelIV.trainer import RFFKernelIVTrainer
from src.models.DeepGMM.trainer import DeepGMMTrainer


logger = logging.getLogger()

def get_trainer(alg_name: str):
    if alg_name == "DeepIV":
        return DeepIVTrainer
    elif alg_name == "DFIV":
        return DFIVTrainer
    elif alg_name == "KernelIV":
        return KernelIVTrainer
    elif alg_name == "RFFKernelIV":
        return RFFKernelIVTrainer
    elif alg_name == "deepGMM":
        return DeepGMMTrainer
    else:
        raise ValueError(f"invalid algorithm name {alg_name}")


def run_one(alg_name: str, data_param: Dict[str, Any], train_config: Dict[str, Any],
            use_gpu: bool, dump_dir_root: Optional[Path], experiment_id: int, verbose: int):
    Trainer_cls = get_trainer(alg_name)
    one_dump_dir = None
    if dump_dir_root is not None:
        one_dump_dir = dump_dir_root.joinpath(f"{experiment_id}/")
        os.mkdir(one_dump_dir)
    trainer = Trainer_cls(data_param, train_config, use_gpu, one_dump_dir)
    return trainer.train(experiment_id, verbose)


def experiments(alg_name: str,
                configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int, num_gpu: Optional[int]):
    train_config = configs["train_params"]
    org_data_config = configs["data"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1:
        ray.init(local_mode=True, num_gpus=num_gpu)
        verbose: int = 2
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpu)
        verbose: int = 0

    use_gpu: bool = (num_gpu is not None)

    if use_gpu and torch.cuda.is_available():
        remote_run = ray.remote(num_gpus=1, max_calls=1)(run_one)
    else:
        remote_run = ray.remote(run_one)

    for dump_name, data_param in grid_search_dict(org_data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        tasks = [remote_run.remote(alg_name, data_param, train_config,
                                   use_gpu, one_dump_dir, idx, verbose) for idx in range(n_repeat)]
        res = ray.get(tasks)

        np.savetxt(one_dump_dir.joinpath("result.csv"), np.array(res))
        logger.critical(f"{dump_name} ended")

    ray.shutdown()