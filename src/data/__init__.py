from typing import Tuple, NamedTuple, Optional
import numpy as np

from .demand_design_image import generate_test_demand_design_image, generate_train_demand_design_image
from .demand_design import generate_test_demand_design, generate_train_demand_design
from .dsprine import generate_train_dsprite, generate_test_dsprite
from .data_class import TrainDataSet, TestDataSet


def generate_train_data(data_name: str, rand_seed: int, **args) -> TrainDataSet:
    if data_name == "demand":
        return generate_train_demand_design(args["data_size"], args["rho"], rand_seed, False)
    elif data_name == "demand_old":
        # Demand design for no covariate (deprecated)
        return generate_train_demand_design(args["data_size"], args["rho"], rand_seed, True)
    elif data_name == "demand_image":
        return generate_train_demand_design_image(args["data_size"], args["rho"], rand_seed)
    elif data_name == "dsprite":
        return generate_train_dsprite(args["data_size"], rand_seed)
    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_name: str, **args) -> TestDataSet:
    if data_name == "demand":
        return generate_test_demand_design(False)
    elif data_name == "demand_old":
        # Demand design for no covariate (deprecated)
        return generate_test_demand_design(True)
    elif data_name == "demand_image":
        return generate_test_demand_design_image()
    elif data_name == "dsprite":
        return generate_test_dsprite()
    else:
        raise ValueError(f"data name {data_name} is not valid")

