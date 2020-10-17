import torch
from torch import nn


def build_net_for_demand_old():
    treatment_net = nn.Sequential(nn.Linear(3, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 32),
                                  nn.Tanh())

    instrumental_net = nn.Sequential(nn.Linear(3, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32))

    covariate_net = None
    return treatment_net, instrumental_net, covariate_net
