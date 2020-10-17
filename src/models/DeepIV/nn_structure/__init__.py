from typing import Tuple, Optional

import torch
from torch import nn

from .nn_structure_for_demand import build_net_for_demand
from .nn_structure_for_demand_image import build_net_for_demand_image
from .nn_structure_for_dsprite import build_net_for_dsprite

import logging

logger = logging.getLogger()


def build_extractor(data_name: str, **args) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    if data_name == "demand":
        logger.info("build without image")
        return build_net_for_demand(**args)
    elif data_name == "demand_image":
        logger.info("build with image")
        return build_net_for_demand_image(**args)
    elif data_name == "demand_old":
        raise ValueError("Cannot specify demand_old for DeepIV")
    elif data_name == "dsprite":
        logger.info("build dsprite model")
        return build_net_for_dsprite()
    else:
        raise ValueError(f"data name {data_name} is not valid")
