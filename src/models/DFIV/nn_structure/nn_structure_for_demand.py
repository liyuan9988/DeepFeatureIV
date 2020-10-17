import torch
from torch import nn


def build_net_for_demand():
    treatment_net = nn.Sequential(nn.Linear(1, 16),
                                  nn.ReLU(),
                                  nn.Linear(16, 1))

    instrumental_net = nn.Sequential(nn.Linear(3, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    covariate_net = nn.Sequential(nn.Linear(2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 32),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(),
                                  nn.Linear(32, 16),
                                  nn.ReLU())
    return treatment_net, instrumental_net, covariate_net
