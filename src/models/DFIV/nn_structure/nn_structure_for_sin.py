import torch
from torch import nn
from typing import Tuple, Optional


def build_net_for_sin() -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    treatment_net = nn.Sequential(nn.Linear(1, 20),
                                  nn.LeakyReLU(),
                                  nn.Linear(20, 3),
                                  nn.LeakyReLU())

    instrumental_net = nn.Sequential(nn.Linear(2, 20),
                                     nn.LeakyReLU())

    return treatment_net, instrumental_net, None
