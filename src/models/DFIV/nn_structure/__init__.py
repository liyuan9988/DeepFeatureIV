from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .nn_structure_for_demand import build_net_for_demand
from .nn_structure_for_demand_old import build_net_for_demand_old
from .nn_structure_for_demand_image import build_net_for_demand_image
from .nn_structure_for_dsprite import build_net_for_dsprite

import logging

logger = logging.getLogger()


def build_extractor(data_name: str) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    if data_name == "demand":
        logger.info("build without image")
        return build_net_for_demand()
    elif data_name == "demand_image":
        logger.info("build with image")
        return build_net_for_demand_image()
    elif data_name == "demand_old":
        logger.info("build old model without image")
        return build_net_for_demand_old()
    elif data_name == "dsprite":
        logger.info("build dsprite model")
        return build_net_for_dsprite()
    else:
        raise ValueError(f"data name {data_name} is not valid")
